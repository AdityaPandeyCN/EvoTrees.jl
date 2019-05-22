# compute the gradient and hessian given target and predict
# linear
function update_grads!(loss::Linear, α, pred::AbstractArray{T, 1}, target::AbstractArray{T, 1}, δ::AbstractArray{T, 1}, δ²::AbstractArray{T, 1}, 𝑤::AbstractArray{T, 1}) where T <: AbstractFloat
    @. δ = 2 * (pred - target) * 𝑤
    @. δ² = 2 * 𝑤
end

# compute the gradient and hessian given target and predict
# logistic - on linear predictor
function update_grads!(loss::Logistic, α, pred::AbstractArray{T, 1}, target::AbstractArray{T, 1}, δ::AbstractArray{T, 1}, δ²::AbstractArray{T, 1}, 𝑤::AbstractArray{T, 1}) where T <: AbstractFloat
    @. δ = (sigmoid(pred) * (1 - target) - (1 - sigmoid(pred)) * target) * 𝑤
    @. δ² = sigmoid(pred) * (1 - sigmoid(pred)) * 𝑤
end

# compute the gradient and hessian given target and predict
# poisson
# Reference: https://isaacchanghau.github.io/post/loss_functions/
function update_grads!(loss::Poisson, α, pred::AbstractArray{T, 1}, target::AbstractArray{T, 1}, δ::AbstractArray{T, 1}, δ²::AbstractArray{T, 1}, 𝑤::AbstractArray{T, 1}) where T <: AbstractFloat
    @. δ = (exp(pred) - target) * 𝑤
    @. δ² = exp(pred) * 𝑤
end

# compute the gradient and hessian given target and predict
# quantile
function update_grads!(loss::Quantile, α, pred::AbstractArray{T, 1}, target::AbstractArray{T, 1}, δ::AbstractArray{T, 1}, δ²::AbstractArray{T, 1}, 𝑤::AbstractArray{T, 1}) where T <: AbstractFloat
    @. δ =  (α * max(target - pred, 0) - (1-α) * max(pred - target, 0)) * 𝑤
end

function logit(x::AbstractArray{T, 1}) where T <: AbstractFloat
    @. x = log(x / (1 - x))
    return x
end

function logit(x::T) where T <: AbstractFloat
    x = log(x / (1 - x))
    return x
end

function sigmoid(x::AbstractArray{T, 1}) where T <: AbstractFloat
    @. x = 1 / (1 + exp(-x))
    return x
end

function sigmoid(x::T) where T <: AbstractFloat
    x = 1 / (1 + exp(-x))
    return x
end

# update the performance tracker - GradientRegression
function update_track!(loss::S, track::SplitTrack{T}, λ::T) where {S <: GradientRegression, T <: AbstractFloat}
    track.gainL = (track.∑δL ^ 2 / (track.∑δ²L + λ .* track.∑𝑤L)) / 2
    track.gainR = (track.∑δR ^ 2 / (track.∑δ²R + λ .* track.∑𝑤R)) / 2
    track.gain = track.gainL + track.gainR
end

# update the performance tracker - QuantileRegression
function update_track!(loss::S, track::SplitTrack{T}, λ::T) where {S <: QuantileRegression, T <: AbstractFloat}
    track.gainL = abs(track.∑δL)
    track.gainR = abs(track.∑δR)
    track.gain = track.gainL + track.gainR
end

# Calculate the gain for a given split - GradientRegression
function get_gain(loss::S, ∑δ::T, ∑δ²::T, ∑𝑤::T, λ::T) where {S <: GradientRegression, T <: AbstractFloat}
    gain = (∑δ ^ 2 / (∑δ² + λ * ∑𝑤)) / 2
    return gain
end

# Calculate the gain for a given split - QuantileRegression
function get_gain(loss::S, ∑δ::T, ∑δ²::T, ∑𝑤::T, λ::T) where {S <: QuantileRegression, T <: AbstractFloat}
    gain = abs(∑δ)
    return gain
end
