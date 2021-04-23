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
function update_grads!(::Linear, δ𝑤::Matrix{T}, p::Matrix{T}, y::Vector{T}) where {T <: AbstractFloat}
    @inbounds @simd for i in 1:size(δ𝑤, 2)
        δ𝑤[1,i] = 2 * (p[1,i] - y[i]) * δ𝑤[3,i]
        δ𝑤[2,i] = 2 * δ𝑤[3,i]
    end
end

# logistic - on linear predictor
function update_grads!(::Logistic, δ𝑤::Matrix{T}, p::Matrix{T}, y::Vector{T}) where {T <: AbstractFloat}
    @inbounds @simd for i in 1:size(δ𝑤, 2)
        δ𝑤[1,i] = (p[1,i] * (1 - y[i]) - (1 - p[1,i]) * y[i]) * δ𝑤[3,i]
        δ𝑤[2,i] = p[1,i] * (1 - p[1,i]) * δ𝑤[3,i]
    end
end

# Poisson
function update_grads!(::Poisson, α::T, pred::Vector{SVector{L,T}}, target::AbstractVector{T}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat,L}
    @inbounds for i in eachindex(δ)
        δ[i] = (exp(pred[i][1]) .- target[i]) * 𝑤[i]
        δ²[i] = exp(pred[i][1]) * 𝑤[i]
    end
end

# L1
# function update_grads!(::L1, α::T, pred::Vector{SVector{L,T}}, target::AbstractArray{T,1}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat,L}
#     @inbounds for i in eachindex(δ)
#         δ[i] =  (α * max(target[i] - pred[i][1], 0) - (1 - α) * max(pred[i][1] - target[i], 0)) * 𝑤[i]
#     end
# end

# Softmax
function update_grads!(::Softmax, α::T, pred::Vector{SVector{L,T}}, target::AbstractVector{S}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat,L,S <: Integer}
    # pred = pred - maximum.(pred)
    # sums = sum(exp.(pred), dims=2)
    @inbounds for i in 1:size(pred, 1)
        pred[i] = SVector{L,T}(pred[i] .- maximum(pred[i]))
        sums = sum(exp.(pred[i]))
        δ[i] = SVector{L,T}((exp.(pred[i]) / sums - (onehot(target[i], 1:L))) .* 𝑤[i][1])
        δ²[i] = SVector{L,T}(1 / sums .* (1 .- exp.(pred[i]) / sums) .* 𝑤[i][1])
    end
end

# Quantile
# function update_grads!(::Quantile, α::T, pred::Vector{SVector{L,T}}, target::AbstractVector{T}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑤::Vector{SVector{1,T}}) where {T <: AbstractFloat,L}
#     @inbounds for i in eachindex(δ)
#         δ[i] = target[i] > pred[i][1] ? SVector(α * 𝑤[i][1]) : SVector((α - 1) * 𝑤[i][1])
#         δ²[i] = SVector(target[i] - pred[i][1]) # δ² serves to calculate the quantile value - hence no weighting on δ²
#     end
# end

# Gaussian - http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
# pred[i][1] = μ
# pred[i][2] = log(σ)
function update_grads!(::Gaussian, δ𝑤::Matrix{T}, p::Matrix{T}, y::Vector{T}) where {T <: AbstractFloat}
    @inbounds @simd for i in 1:size(δ𝑤, 2)

        # first order
        δ𝑤[1,i] = (p[1,i] - y[i]) / max(1e-8, exp(2 * p[2,i])) * δ𝑤[5,i]
        δ𝑤[2,i] = (1 - (p[1,i] - y[i])^2 / max(1e-8, exp(2 * p[2,i]))) * δ𝑤[5,i]

        # second order
        δ𝑤[3,i] = δ𝑤[5,i] / max(1e-8, exp(2 * p[2,i]))
        δ𝑤[4,i] = 2 * δ𝑤[5,i] / max(1e-8, exp(2 * p[2,i])) * (p[1,i] - y[i])^2
    end
end

# utility functions
function logit(x::AbstractArray{T,1}) where T <: AbstractFloat
    @. x = log(x / (1 - x))
    return x
end

function logit(x::T) where T <: AbstractFloat
    x = log(x / (1 - x))
    return x
end

function sigmoid(x::AbstractArray{T,1}) where T <: AbstractFloat
    @. x = 1 / (1 + exp(-x))
    return x
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
function get_gain(::S, ∑::Vector{T}, λ::T) where {S <: GradientRegression,T <: AbstractFloat}
    ∑[1]^2 / (∑[2] + λ * ∑[3]) / 2
end

# GaussianRegression
function get_gain(::S, ∑::Vector{T}, λ::T) where {S <: GaussianRegression,T <: AbstractFloat}
    (∑[1]^2 / (∑[3] + λ * ∑[5]) + ∑[2]^2 / (∑[4] + λ * ∑[5])) / 2
end

# GradientRegression
# function get_gain(loss::S, ∑δ::SVector{L,T}, ∑δ²::SVector{L,T}, ∑𝑤::SVector{1,T}, λ::T) where {S <: GradientRegression,T <: AbstractFloat,L}
#     gain = sum((∑δ.^2 ./ (∑δ² .+ λ .* ∑𝑤)) ./ 2)
# return gain
# end
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
# function get_gain(::S, ∑δ::SVector{L,T}, ∑δ²::SVector{L,T}, ∑𝑤::SVector{1,T}, λ::T) where {S <: GaussianRegression,T <: AbstractFloat,L}
#     gain = sum((∑δ.^2 ./ (∑δ² .+ λ .* ∑𝑤)) ./ 2)
#     return gain
# end

function update_childs_∑!(::L, nodes, n, bin, feat) where {L <: GradientRegression}
    nodes[n << 1].∑ .= nodes[n].hL[feat][(3 * bin - 2):(3 * bin)]
    nodes[n << 1 + 1].∑ .= nodes[n].hR[feat][(3 * bin - 2):(3 * bin)]
    return nothing
end

function update_childs_∑!(::L, nodes, n, bin, feat) where {L <: GaussianRegression}
    nodes[n << 1].∑ .= nodes[n].hL[feat][(5 * bin - 4):(5 * bin)]
    nodes[n << 1 + 1].∑ .= nodes[n].hR[feat][(5 * bin - 4):(5 * bin)]
    return nothing
end