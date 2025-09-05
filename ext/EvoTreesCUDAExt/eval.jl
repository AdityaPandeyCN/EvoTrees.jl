using KernelAbstractions

########################
# MSE
########################
@kernel function eval_mse_kernel!(eval, p, y, w)
    i = @index(Global)
    if i <= length(y)
        @inbounds eval[i] = w[i] * (p[1, i] - y[i])^2
    end
end

function EvoTrees.mse(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    eval_mse_kernel!(backend)(eval, p, y, w; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return sum(eval) / sum(w)
end

########################
# RMSE - FIXED BUG
########################
EvoTrees.rmse(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat} =
    sqrt(EvoTrees.mse(p, y, w, eval; MAX_THREADS, kwargs...))  # Fixed: call mse not rmse

########################
# MAE
########################
@kernel function eval_mae_kernel!(eval, p, y, w)
    i = @index(Global)
    if i <= length(y)
        @inbounds eval[i] = w[i] * abs(p[1, i] - y[i])
    end
end

function EvoTrees.mae(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    eval_mae_kernel!(backend)(eval, p, y, w; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return sum(eval) / sum(w)
end

########################
# WMAE
########################
@kernel function eval_wmae_kernel!(eval, p, y, w, alpha)
    i = @index(Global)
    if i <= length(y)
        @inbounds eval[i] = w[i] * (
            alpha * max(y[i] - p[1, i], zero(eltype(p))) +
            (1 - alpha) * max(p[1, i] - y[i], zero(eltype(p)))
        )
    end
end

function EvoTrees.wmae(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; alpha=0.5, MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    eval_wmae_kernel!(backend)(eval, p, y, w, T(alpha); ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return sum(eval) / sum(w)
end

########################
# Logloss
########################
@kernel function eval_logloss_kernel!(eval, p, y, w)
    i = @index(Global)
    ϵ = eps(eltype(p))
    if i <= length(y)
        @inbounds pred = clamp(EvoTrees.sigmoid(p[1, i]), ϵ, 1 - ϵ)  # Added numerical stability
        @inbounds eval[i] = w[i] * (-y[i] * log(pred) - (1 - y[i]) * log(1 - pred))
    end
end

function EvoTrees.logloss(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    eval_logloss_kernel!(backend)(eval, p, y, w; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return sum(eval) / sum(w)
end

########################
# Gaussian - WITH NUMERICAL STABILITY
########################
@kernel function eval_gaussian_kernel!(eval, p, y, w)
    i = @index(Global)
    if i <= length(y)
        @inbounds log_sigma = clamp(p[2, i], eltype(p)(-10), eltype(p)(10))  # Added clamping
        @inbounds sigma2 = exp(2 * log_sigma)
        @inbounds eval[i] = -w[i] * (log_sigma + 0.5 * (y[i] - p[1, i])^2 / sigma2)
    end
end

function EvoTrees.gaussian_mle(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    eval_gaussian_kernel!(backend)(eval, p, y, w; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return sum(eval) / sum(w)
end

########################
# Poisson Deviance
########################
@kernel function eval_poisson_kernel!(eval, p, y, w)
    i = @index(Global)
    ϵ = eps(eltype(p)(1e-7))
    if i <= length(y)
        @inbounds pred = exp(p[1, i])
        @inbounds eval[i] = w[i] * 2 * (y[i] * log(y[i] / pred + ϵ) + pred - y[i])
    end
end

function EvoTrees.poisson(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    eval_poisson_kernel!(backend)(eval, p, y, w; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return sum(eval) / sum(w)
end

########################
# Gamma Deviance
########################
@kernel function eval_gamma_kernel!(eval, p, y, w)
    i = @index(Global)
    if i <= length(y)
        @inbounds pred = exp(p[1, i])
        @inbounds eval[i] = w[i] * 2 * (log(pred / y[i]) + y[i] / pred - 1)
    end
end

function EvoTrees.gamma(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    eval_gamma_kernel!(backend)(eval, p, y, w; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return sum(eval) / sum(w)
end

########################
# Tweedie Deviance
########################
@kernel function eval_tweedie_kernel!(eval, p, y, w)
    i = @index(Global)
    rho = eltype(p)(1.5)
    if i <= length(y)
        pred = exp(p[1, i])
        @inbounds eval[i] = w[i] * 2 * (y[i]^(2 - rho) / (1 - rho) / (2 - rho) - y[i] * pred^(1 - rho) / (1 - rho) + pred^(2 - rho) / (2 - rho))
    end
end

function EvoTrees.tweedie(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    eval_tweedie_kernel!(backend)(eval, p, y, w; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return sum(eval) / sum(w)
end

########################
# mlogloss
########################
@kernel function eval_mlogloss_kernel!(eval, p, y, w)
    i = @index(Global)
    K = size(p, 1)
    if i <= length(y)
        isum = zero(eltype(p))
        @inbounds for k in 1:K
            isum += exp(p[k, i])
        end
        @inbounds eval[i] = w[i] * (log(isum) - p[y[i], i])
    end
end

function EvoTrees.mlogloss(p::CuMatrix{T}, y::CuVector, w::CuVector{T}, eval::CuVector{T}; MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    eval_mlogloss_kernel!(backend)(eval, p, y, w; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return sum(eval) / sum(w)
end

########################
# MISSING: Quantile (ADD THIS)
########################
@kernel function eval_quantile_kernel!(eval, p, y, w, alpha)
    i = @index(Global)
    if i <= length(y)
        @inbounds diff = y[i] - p[1, i]
        @inbounds eval[i] = w[i] * abs(diff) * (diff > 0 ? alpha : (1 - alpha))
    end
end

function EvoTrees.quantile(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; 
                          alpha=0.5, MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    eval_quantile_kernel!(backend)(eval, p, y, w, T(alpha); ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return sum(eval) / sum(w)
end

########################
# MISSING: Credibility Variance
########################
@kernel function eval_cred_var_kernel!(eval, p, y, w, lambda)
    i = @index(Global)
    if i <= length(y)
        @inbounds eval[i] = w[i] * ((y[i] - p[1, i])^2 + lambda * p[2, i])
    end
end

function EvoTrees.cred_var(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; 
                           lambda=1.0, MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    eval_cred_var_kernel!(backend)(eval, p, y, w, T(lambda); ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return sum(eval) / sum(w)
end

########################
# MISSING: Credibility Std
########################
@kernel function eval_cred_std_kernel!(eval, p, y, w, lambda)
    i = @index(Global)
    if i <= length(y)
        @inbounds sigma = exp(p[2, i])
        @inbounds eval[i] = w[i] * ((y[i] - p[1, i])^2 / sigma + lambda * sigma)
    end
end

function EvoTrees.cred_std(p::CuMatrix{T}, y::CuVector{T}, w::CuVector{T}, eval::CuVector{T}; 
                           lambda=1.0, MAX_THREADS=1024, kwargs...) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    eval_cred_std_kernel!(backend)(eval, p, y, w, T(lambda); ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return sum(eval) / sum(w)
end

