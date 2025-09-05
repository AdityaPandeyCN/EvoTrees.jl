using KernelAbstractions

#####################
# MSE
#####################
@kernel function kernel_mse_∇!(∇, p, y)
    i = @index(Global)
    if i <= length(y)
        @inbounds ∇[1, i] = 2 * (p[1, i] - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = 2 * ∇[3, i]
    end
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.MSE},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    kernel_mse_∇!(backend)(∇, p, y; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# MAE - FIXED
#####################
@kernel function kernel_mae_∇!(∇, p, y)
    i = @index(Global)
    if i <= length(y)
        @inbounds diff = y[i] - p[1, i]
        @inbounds ∇[1, i] = sign(diff) * ∇[3, i]  # Fixed: use sign function
        @inbounds ∇[2, i] = eltype(∇)(1e-4) * ∇[3, i]  # Fixed: small constant for Hessian
    end
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.MAE},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    kernel_mae_∇!(backend)(∇, p, y; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# Quantile - FIXED
#####################
@kernel function kernel_quantile_∇!(∇, p, y, alpha)
    i = @index(Global)
    if i <= length(y)
        @inbounds diff = y[i] - p[1, i]
        @inbounds ∇[1, i] = (diff > 0 ? alpha : (alpha - 1)) * ∇[3, i]
        @inbounds ∇[2, i] = eltype(∇)(1e-4) * ∇[3, i]  # Fixed: small constant instead of diff
    end
end

function EvoTrees.update_grads!(
    ∇::CuMatrix{T},
    p::CuMatrix{T},
    y::CuVector{T},
    ::Type{EvoTrees.Quantile},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    kernel_quantile_∇!(backend)(∇, p, y, T(params.alpha); ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# LogLoss
#####################
@kernel function kernel_logloss_∇!(∇, p, y)
    i = @index(Global)
    ϵ = eps(eltype(p))
    if i <= length(y)
        @inbounds pred = clamp(EvoTrees.sigmoid(p[1, i]), ϵ, 1 - ϵ)  # Added stability
        @inbounds ∇[1, i] = (pred - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = pred * (1 - pred) * ∇[3, i]
    end
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.LogLoss},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    kernel_logloss_∇!(backend)(∇, p, y; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# Poisson
#####################
@kernel function kernel_poisson_∇!(∇, p, y)
    i = @index(Global)
    if i <= length(y)
        @inbounds pred = exp(p[1, i])
        @inbounds ∇[1, i] = (pred - y[i]) * ∇[3, i]
        @inbounds ∇[2, i] = pred * ∇[3, i]
    end
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.Poisson},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    kernel_poisson_∇!(backend)(∇, p, y; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# Gamma
#####################
@kernel function kernel_gamma_∇!(∇, p, y)
    i = @index(Global)
    if i <= length(y)
        @inbounds pred = exp(p[1, i])
        @inbounds ∇[1, i] = 2 * (1 - y[i] / pred) * ∇[3, i]
        @inbounds ∇[2, i] = 2 * y[i] / pred * ∇[3, i]
    end
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.Gamma},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    kernel_gamma_∇!(backend)(∇, p, y; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# Tweedie
#####################
@kernel function kernel_tweedie_∇!(∇, p, y)
    i = @index(Global)
    rho = eltype(p)(1.5)
    if i <= length(y)
        @inbounds pred = exp(p[1, i])
        @inbounds ∇[1, i] = 2 * (pred^(2 - rho) - y[i] * pred^(1 - rho)) * ∇[3, i]
        @inbounds ∇[2, i] = 2 * ((2 - rho) * pred^(2 - rho) - (1 - rho) * y[i] * pred^(1 - rho)) * ∇[3, i]
    end
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.Tweedie},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    kernel_tweedie_∇!(backend)(∇, p, y; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# Softmax
#####################
@kernel function kernel_mlogloss_∇!(∇, p, y)
    i = @index(Global)
    K = size(p, 1)
    if i <= length(y)
        isum = zero(eltype(p))
        @inbounds for k in 1:K
            isum += exp(p[k, i])
        end
        @inbounds for k in 1:K
            iexp = exp(p[k, i])
            if k == y[i]
                ∇[k, i] = (iexp / isum - 1) * ∇[end, i]
            else
                ∇[k, i] = iexp / isum * ∇[end, i]
            end
            ∇[k+K, i] = iexp / isum * (1 - iexp / isum) * ∇[end, i]
        end
    end
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.MLogLoss},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    kernel_mlogloss_∇!(backend)(∇, p, y; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return
end

################################################################################
# Gaussian - FIXED with numerical stability
################################################################################
@kernel function kernel_gauss_∇!(∇, p, y)
    i = @index(Global)
    if i <= length(y)
        # Clamp log_sigma to prevent numerical issues
        @inbounds log_sigma = clamp(p[2, i], eltype(p)(-10), eltype(p)(10))
        @inbounds sigma2 = exp(2 * log_sigma)
        @inbounds diff = p[1, i] - y[i]
        
        # first order gradients
        @inbounds ∇[1, i] = diff / sigma2 * ∇[5, i]
        @inbounds ∇[2, i] = (1 - diff^2 / sigma2) * ∇[5, i]
        # second order gradients
        @inbounds ∇[3, i] = ∇[5, i] / sigma2
        @inbounds ∇[4, i] = 2 * ∇[5, i] * diff^2 / sigma2
    end
end

function EvoTrees.update_grads!(
    ∇::CuMatrix,
    p::CuMatrix,
    y::CuVector,
    ::Type{EvoTrees.GaussianMLE},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
)
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    kernel_gauss_∇!(backend)(∇, p, y; ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# Credibility Variance - NEW CORRECT IMPLEMENTATION
#####################
@kernel function kernel_cred_var_∇!(∇, p, y, lambda)
    i = @index(Global)
    if i <= length(y)
        @inbounds ∇[1, i] = -2 * (y[i] - p[1, i]) * ∇[5, i]
        @inbounds ∇[2, i] = lambda * ∇[5, i]
        @inbounds ∇[3, i] = 2 * ∇[5, i]
        @inbounds ∇[4, i] = zero(eltype(∇)) * ∇[5, i]
    end
end

function EvoTrees.update_grads!(
    ∇::CuMatrix{T},
    p::CuMatrix{T},
    y::CuVector{T},
    ::Type{EvoTrees.CredVar},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    kernel_cred_var_∇!(backend)(∇, p, y, T(params.lambda); ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return
end

#####################
# Credibility Std - NEW CORRECT IMPLEMENTATION
#####################
@kernel function kernel_cred_std_∇!(∇, p, y, lambda)
    i = @index(Global)
    if i <= length(y)
        @inbounds sigma = exp(p[2, i])
        @inbounds diff = y[i] - p[1, i]
        @inbounds ∇[1, i] = -2 * diff / sigma * ∇[5, i]
        @inbounds ∇[2, i] = (-diff^2 / sigma + lambda * sigma) * ∇[5, i]
        @inbounds ∇[3, i] = 2 / sigma * ∇[5, i]
        @inbounds ∇[4, i] = (2 * diff^2 / sigma + lambda * sigma) * ∇[5, i]
    end
end

function EvoTrees.update_grads!(
    ∇::CuMatrix{T},
    p::CuMatrix{T},
    y::CuVector{T},
    ::Type{EvoTrees.CredStd},
    params::EvoTrees.EvoTypes;
    MAX_THREADS=1024
) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(p)
    n = length(y)
    workgroupsize = min(256, n)
    kernel_cred_std_∇!(backend)(∇, p, y, T(params.lambda); ndrange=n, workgroupsize=workgroupsize)
    KernelAbstractions.synchronize(backend)
    return
end

