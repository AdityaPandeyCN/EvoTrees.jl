
# compute the gradient and hessian given target and predict
# linear
function update_grads!(::Val{:linear}, pred::AbstractArray{T, 1}, target::AbstractArray{T, 1}, δ::AbstractArray{T, 1}, δ²::AbstractArray{T, 1}, 𝑤::AbstractArray{T, 1}) where T <: AbstractFloat
    @. δ = 2 * (pred - target) * 𝑤
    @. δ² = 2.0 * 𝑤
end

# compute the gradient and hessian given target and predict
# logistic - on linear predictor
function update_grads!(::Val{:logistic}, pred::AbstractArray{T, 1}, target::AbstractArray{T, 1}, δ::AbstractArray{T, 1}, δ²::AbstractArray{T, 1}, 𝑤::AbstractArray{T, 1}) where T <: AbstractFloat
    @. δ = (sigmoid(pred) * (1 - target) - (1 - sigmoid(pred)) * target) * 𝑤
    @. δ² = sigmoid(pred) * (1 - sigmoid(pred)) * 𝑤
end


# compute the gradient and hessian given target and predict
# logistic - directly on probs
# function update_grads!(::Val{:logistic}, pred::AbstractArray{T, 1}, target::AbstractArray{T, 1}, δ::AbstractArray{T, 1}, δ²::AbstractArray{T, 1}) where T <: AbstractFloat
#     @. δ = (1 - target) / (1 - pred) - target / pred
#     @. δ² = (1 - target) / (1 - pred) ^ 2 + target / pred ^ 2
# end

function logit(x::AbstractArray{T, 1}) where T <: AbstractFloat
    @. x = x / (1 - x)
    return δ, δ²
end

function sigmoid(x::AbstractArray{T, 1}) where T <: AbstractFloat
    @. x = 1 / (1 + exp(-x))
    return x
end

function sigmoid(x::T) where T <: AbstractFloat
    x = 1 / (1 + exp(-x))
    return x
end

# update the performance tracker
function update_track!(track::SplitTrack{T}, λ::T) where T <: AbstractFloat
    track.gainL = (track.∑δL ^ 2 / (track.∑δ²L + λ .* track.∑𝑤L)) / 2.0
    track.gainR = (track.∑δR ^ 2 / (track.∑δ²R + λ .* track.∑𝑤R)) / 2.0
    track.gain = track.gainL + track.gainR
end

# Calculate the gain for a given split
function get_gain(∑δ::T, ∑δ²::T, ∑𝑤::T, λ::T) where T <: AbstractFloat
    gain = (∑δ ^ 2 / (∑δ² + λ * ∑𝑤)) / 2.0
    return gain
end
