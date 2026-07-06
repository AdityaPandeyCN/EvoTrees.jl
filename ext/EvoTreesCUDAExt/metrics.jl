@inline metric_obs(::Val{:mse}, pk, yk, alpha) = (pk - yk)^2
@inline metric_obs(::Val{:mae}, pk, yk, alpha) = abs(pk - yk)
@inline function metric_obs(::Val{:wmae}, pk, yk, alpha)
    return alpha * max(yk - pk, zero(pk)) + (1 - alpha) * max(pk - yk, zero(pk))
end
@inline function metric_obs(::Val{:logloss}, pk, yk, alpha)
    pred = EvoTrees.sigmoid(pk)
    return -yk * log(pred) + (yk - 1) * log(1 - pred)
end
@inline function metric_obs(::Val{:poisson}, pk, yk, alpha)
    ϵ = eps(oftype(pk, 1e-7))
    pred = exp(pk)
    return 2 * (yk * log(yk / pred + ϵ) + pred - yk)
end
@inline function metric_obs(::Val{:gamma}, pk, yk, alpha)
    pred = exp(pk)
    return 2 * (log(pred / yk) + yk / pred - 1)
end
@inline function metric_obs(::Val{:tweedie}, pk, yk, alpha)
    rho = oftype(pk, 1.5)
    pred = exp(pk)
    return 2 * (
        yk^(2 - rho) / (1 - rho) / (2 - rho) -
        yk * pred^(1 - rho) / (1 - rho) +
        pred^(2 - rho) / (2 - rho)
    )
end

function eval_metric_kernel!(
    eval::CuDeviceVector{T},
    p::CuDeviceMatrix{T},
    y::CuDeviceVector{T},
    w::CuDeviceVector{T},
    metric,
    alpha::T,
) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds eval[i] = w[i] * metric_obs(metric, p[1, i], y[i], alpha)
    end
    return nothing
end

function eval_metric_mt_kernel!(
    eval::CuDeviceVector{T},
    p::CuDeviceMatrix{T},
    y::CuDeviceMatrix{T},
    w::CuDeviceVector{T},
    metric,
    alpha::T,
) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size(y, 2)
        K = size(p, 1)
        acc = zero(T)
        @inbounds for k in 1:K
            acc += metric_obs(metric, p[k, i], y[k, i], alpha)
        end
        @inbounds eval[i] = w[i] * acc / K
    end
    return nothing
end

function eval_metric_gpu(metric::Val, p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, alpha=0.5, kwargs...) where {T<:AbstractFloat}
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads eval_metric_kernel!(eval, p, y, w, metric, T(alpha))
    CUDA.synchronize()
    return sum(eval) / sum(w)
end

function eval_metric_gpu(metric::Val, p::CuMatrix{T}, y::CuMatrix{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, alpha=0.5, kwargs...) where {T<:AbstractFloat}
    threads = min(MAX_THREADS, size(y, 2))
    blocks = cld(size(y, 2), threads)
    @cuda blocks = blocks threads = threads eval_metric_mt_kernel!(eval, p, y, w, metric, T(alpha))
    CUDA.synchronize()
    return sum(eval) / sum(w)
end

########################
# MSE
########################
function EvoTrees.mse(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    return eval_metric_gpu(Val(:mse), p, y, w, eval; MAX_THREADS, kwargs...)
end

function EvoTrees.mse(p::CuMatrix{T}, y::CuMatrix{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    return eval_metric_gpu(Val(:mse), p, y, w, eval; MAX_THREADS, kwargs...)
end

########################
# RMSE
########################
EvoTrees.rmse(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat} =
    sqrt(EvoTrees.mse(p, y, w, eval; MAX_THREADS, kwargs...))
EvoTrees.rmse(p::CuMatrix{T}, y::CuMatrix{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat} =
    sqrt(EvoTrees.mse(p, y, w, eval; MAX_THREADS, kwargs...))

########################
# MAE
########################
function EvoTrees.mae(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    return eval_metric_gpu(Val(:mae), p, y, w, eval; MAX_THREADS, kwargs...)
end

function EvoTrees.mae(p::CuMatrix{T}, y::CuMatrix{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    return eval_metric_gpu(Val(:mae), p, y, w, eval; MAX_THREADS, kwargs...)
end

########################
# WMAE
########################
function EvoTrees.wmae(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, alpha=0.5, kwargs...) where {T<:AbstractFloat}
    return eval_metric_gpu(Val(:wmae), p, y, w, eval; MAX_THREADS, alpha, kwargs...)
end

function EvoTrees.wmae(p::CuMatrix{T}, y::CuMatrix{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, alpha=0.5, kwargs...) where {T<:AbstractFloat}
    return eval_metric_gpu(Val(:wmae), p, y, w, eval; MAX_THREADS, alpha, kwargs...)
end

########################
# MultiQuantile
########################
function eval_multiquantile_kernel!(
    eval::CuDeviceVector{T},
    p::CuDeviceMatrix{T},
    y::CuDeviceVector{T},
    w::CuDeviceVector{T},
    alphas::CuDeviceVector{T},
    K::Int,
) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        yi = y[i]
        acc = zero(T)
        @inbounds for k in 1:K
            diff = yi - p[k, i]
            alpha = alphas[k]
            acc += alpha * max(diff, zero(T)) + (1 - alpha) * max(-diff, zero(T))
        end
        @inbounds eval[i] = w[i] * acc / K
    end
    return nothing
end

function EvoTrees.multiquantile(
    p::CuMatrix{T},
    y::CuVector{T},
    w::CuVector{T},
    eval::CuVector{T};
    MAX_THREADS=1024,
    alphas,
    kwargs...
) where {T<:AbstractFloat}
    K = length(alphas)
    alphas_dev = alphas isa CuVector ? alphas : CuArray(T.(alphas))
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads eval_multiquantile_kernel!(eval, p, y, w, alphas_dev, K)
    CUDA.synchronize()
    return sum(eval) / sum(w)
end

########################
# Logloss
########################
function EvoTrees.logloss(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    return eval_metric_gpu(Val(:logloss), p, y, w, eval; MAX_THREADS, kwargs...)
end

function EvoTrees.logloss(p::CuMatrix{T}, y::CuMatrix{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    return eval_metric_gpu(Val(:logloss), p, y, w, eval; MAX_THREADS, kwargs...)
end

########################
# Gaussian
########################
function eval_gaussian_kernel!(eval::CuDeviceVector{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector{T}, w::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(y)
        @inbounds eval[i] = -w[i] * (p[2, i] + (y[i] - p[1, i])^2 / (2 * exp(2 * p[2, i])))
    end
    return nothing
end
function EvoTrees.gaussian_mle(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads eval_gaussian_kernel!(eval, p, y, w)
    CUDA.synchronize()
    return sum(eval) / sum(w)
end

function eval_gaussian_mt_kernel!(eval, p, y, w)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size(y, 2)
        Y = size(p, 1) ÷ 2
        acc = zero(eltype(eval))
        @inbounds for t in 1:Y
            μ = p[2t-1, i]
            ls = p[2t, i]
            yt = y[t, i]
            acc += -(ls + (yt - μ)^2 / (2 * exp(2 * ls)))
        end
        @inbounds eval[i] = w[i] * acc / Y
    end
    return nothing
end
function EvoTrees.gaussian_mle(p::CuMatrix{T}, y::CuMatrix{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    threads = min(MAX_THREADS, size(y, 2))
    blocks = cld(size(y, 2), threads)
    @cuda blocks = blocks threads = threads eval_gaussian_mt_kernel!(eval, p, y, w)
    CUDA.synchronize()
    return sum(eval) / sum(w)
end

function eval_logistic_mt_kernel!(eval, p, y, w)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= size(y, 2)
        Y = size(p, 1) ÷ 2
        acc = zero(eltype(eval))
        @inbounds for t in 1:Y
            μ = p[2t-1, i]
            ls = p[2t, i]
            yt = y[t, i]
            acc += log(1 / 4 * sech(exp(-ls) * (yt - μ))^2) - ls
        end
        @inbounds eval[i] = w[i] * acc / Y
    end
    return nothing
end
function EvoTrees.logistic_mle(p::CuMatrix{T}, y::CuMatrix{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    threads = min(MAX_THREADS, size(y, 2))
    blocks = cld(size(y, 2), threads)
    @cuda blocks = blocks threads = threads eval_logistic_mt_kernel!(eval, p, y, w)
    CUDA.synchronize()
    return sum(eval) / sum(w)
end

########################
# Poisson Deviance
########################
function EvoTrees.poisson(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    return eval_metric_gpu(Val(:poisson), p, y, w, eval; MAX_THREADS, kwargs...)
end

function EvoTrees.poisson(p::CuMatrix{T}, y::CuMatrix{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    return eval_metric_gpu(Val(:poisson), p, y, w, eval; MAX_THREADS, kwargs...)
end

########################
# Gamma Deviance
########################
function EvoTrees.gamma(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    return eval_metric_gpu(Val(:gamma), p, y, w, eval; MAX_THREADS, kwargs...)
end

function EvoTrees.gamma(p::CuMatrix{T}, y::CuMatrix{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    return eval_metric_gpu(Val(:gamma), p, y, w, eval; MAX_THREADS, kwargs...)
end

########################
# Tweedie Deviance
########################
function EvoTrees.tweedie(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    return eval_metric_gpu(Val(:tweedie), p, y, w, eval; MAX_THREADS, kwargs...)
end

function EvoTrees.tweedie(p::CuMatrix{T}, y::CuMatrix{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    return eval_metric_gpu(Val(:tweedie), p, y, w, eval; MAX_THREADS, kwargs...)
end

########################
# mlogloss
########################
function eval_mlogloss_kernel!(eval::CuDeviceVector{T}, p::CuDeviceMatrix{T}, y::CuDeviceVector, w::CuDeviceVector{T}) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    K = size(p, 1)
    if i <= length(y)
        isum = zero(T)
        @inbounds for k in 1:K
            isum += exp(p[k, i])
        end
        @inbounds eval[i] = w[i] * (log(isum) - p[y[i], i])
    end
    return nothing
end

function EvoTrees.mlogloss(p::CuMatrix{T}, y::CuVector, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    threads = min(MAX_THREADS, length(y))
    blocks = cld(length(y), threads)
    @cuda blocks = blocks threads = threads eval_mlogloss_kernel!(eval, p, y, w)
    CUDA.synchronize()
    return sum(eval) / sum(w)
end

