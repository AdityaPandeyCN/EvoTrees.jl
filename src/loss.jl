# utility for softmax
struct OneHotVector <: AbstractVector{Bool}
    ix::UInt32
    of::UInt32
end

Base.size(xs::OneHotVector) = (Int64(xs.of),)
Base.getindex(xs::OneHotVector, i::Integer) = i == xs.ix
Base.getindex(xs::OneHotVector, ::Colon) = OneHotVector(xs.ix, xs.of)

function onehot(l, labels)
    i = something(findfirst(isequal(l), labels), 0)
    i > 0 || error("Value $l is not in labels")
    OneHotVector(i, length(labels))
end

# linear
function update_grads!(::Linear, α::T, pred::Vector{SVector{L,T}}, target::AbstractVector{T}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat,L}
    @inbounds for i in eachindex(δ)
        δ[i] = SVector(2 * (pred[i][1] - target[i]) * 𝑤[i][1])
        δ²[i] = SVector(2 * 𝑤[i][1])
    end
end

# logistic - on linear predictor
function update_grads!(::Logistic, α::T, pred::Vector{SVector{L,T}}, target::AbstractVector{T}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat,L}
    ϵ = T(2e-7)
    @inbounds for i in eachindex(δ)
        p = clamp(sigmoid(pred[i][1]), ϵ, 1-ϵ)
        δ[i] = SVector((p - target[i]) * 𝑤[i][1])
        δ²[i] = SVector(max(ϵ, p * (1 - p)) * 𝑤[i][1])
    end
end

# Poisson
function update_grads!(::Poisson, α::T, pred::Vector{SVector{L,T}}, target::AbstractVector{T}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat,L}
    @inbounds for i in eachindex(δ)
        δ[i] = SVector((exp(pred[i][1]) .- target[i]) * 𝑤[i][1])
        δ²[i] = SVector(exp(pred[i][1]) * 𝑤[i][1])
    end
end

# L1
function update_grads!(::L1, α::T, pred::Vector{SVector{L,T}}, target::AbstractArray{T,1}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat,L}
    @inbounds for i in eachindex(δ)
        δ[i] =  SVector((α * max(target[i] - pred[i][1], 0) - (1 - α) * max(pred[i][1] - target[i], 0)) * 𝑤[i][1])
    end
end

# Softmax
function update_grads!(::Softmax, α::T, pred::Vector{SVector{L,T}}, target::AbstractVector{S}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat,L,S <: Integer}
    ϵ = T(2e-7)
    @inbounds for i in 1:size(pred, 1)
        # pred[i] = SVector{L,T}(pred[i] .- maximum(pred[i]))
        # p = exp.(pred[i])
        p = exp.(pred[i] .- maximum(pred[i]))
        p = clamp.(p / sum(p), ϵ, 1-ϵ)
        # δ[i] = SVector{L,T}(min.(1 - ϵ, max.(ϵ, p - (onehot(target[i], 1:L)))) .* 𝑤[i][1])
        # δ²[i] = SVector{L,T}(min.(1 - ϵ, max.(ϵ, 2 .* p .* (1 .- p))) .* 𝑤[i][1])
        δ[i] = SVector{L,T}(p - (onehot(target[i], 1:L)) .* 𝑤[i][1])
        δ²[i] = SVector{L,T}(max.(ϵ, p .* (1 .- p)) .* 𝑤[i][1])
        # δ²[i] = SVector{L,T}(p .* (1 .- p) .* 𝑤[i][1])

        # δ[i] = SVector{L,T}((exp.(pred[i]) / sums - (onehot(target[i], 1:L))) .* 𝑤[i][1])
        # δ²[i] = SVector{L,T}(1 / sums .* (1 .- exp.(pred[i]) / sums) .* 𝑤[i][1])
    end
end

# Quantile
function update_grads!(::Quantile, α::T, pred::Vector{SVector{L,T}}, target::AbstractVector{T}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat,L}
    @inbounds for i in eachindex(δ)
        δ[i] = target[i] > pred[i][1] ? SVector(α * 𝑤[i][1]) : SVector((α - 1) * 𝑤[i][1])
        δ²[i] = SVector(target[i] - pred[i][1]) # δ² serves to calculate the quantile value - hence no weighting on δ²
    end
end

# Gaussian - http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
# pred[i][1] = μ
# pred[i][2] = log(σ)
function update_grads!(::Gaussian, α, pred::Vector{SVector{L,T}}, target::AbstractArray{T,1}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat,L}
    ϵ = T(2e-7)
    @inbounds @threads for i in eachindex(δ)
        δ[i] = SVector((pred[i][1] - target[i]) / max(ϵ, exp(2 * pred[i][2])) * 𝑤[i][1], (1 - (pred[i][1] - target[i])^2 / max(ϵ, exp(2 * pred[i][2]))) * 𝑤[i][1])
        δ²[i] = SVector(𝑤[i][1] / max(ϵ, exp(2 * pred[i][2])),  2 * 𝑤[i][1] / max(ϵ, exp(2 * pred[i][2])) * (pred[i][1] - target[i])^2)
    end
end

# utility functions
function logit(x::AbstractArray{T,1}) where T <: AbstractFloat
    logit.(x)
end

function logit(x::T) where T <: AbstractFloat
    x = log(x / (1 - x))
    return x
end

function sigmoid(x::AbstractArray{T,1}) where T <: AbstractFloat
    sigmoid.(x)
end

function sigmoid(x::T) where T <: AbstractFloat
    x = 1 / (1 + exp(-x))
    return x
end

function softmax(x::AbstractVector{T}) where T <: AbstractFloat
    x .-= maximum(x)
    x = exp.(x) ./ sum(exp.(x))
    return x
end


##############################
# get the gain metric
##############################
# GradientRegression
function get_gain(::S, ∑δ::SVector{L,T}, ∑δ²::SVector{L,T}, ∑𝑤::SVector{1,T}, λ::T) where {S <: GradientRegression,T <: AbstractFloat,L}
    gain = sum((∑δ.^2 ./ (∑δ² .+ λ .* ∑𝑤)) ./ 2)
return gain
end

# MultiClassRegression
function get_gain(::S, ∑δ::SVector{L,T}, ∑δ²::SVector{L,T}, ∑𝑤::SVector{1,T}, λ::T) where {S <: MultiClassRegression,T <: AbstractFloat,L}
    gain = sum((∑δ.^2 ./ (∑δ² .+ λ .* ∑𝑤)) ./ 2)
    return gain
end

# L1 Regression
function get_gain(::S, ∑δ::SVector{L,T}, ∑δ²::SVector{L,T}, ∑𝑤::SVector{1,T}, λ::T) where {S <: L1Regression,T <: AbstractFloat,L}
    gain = sum(abs.(∑δ))
    return gain
end

# QuantileRegression
function get_gain(::S, ∑δ::SVector{L,T}, ∑δ²::SVector{L,T}, ∑𝑤::SVector{1,T}, λ::T) where {S <: QuantileRegression,T <: AbstractFloat,L}
    gain = sum(abs.(∑δ) ./ (1 .+ λ))
    return gain
end

# GaussianRegression
function get_gain(::S, ∑δ::SVector{L,T}, ∑δ²::SVector{L,T}, ∑𝑤::SVector{1,T}, λ::T) where {S <: GaussianRegression,T <: AbstractFloat,L}
    gain = sum((∑δ.^2 ./ (∑δ² .+ λ .* ∑𝑤)) ./ 2)
    return gain
end
