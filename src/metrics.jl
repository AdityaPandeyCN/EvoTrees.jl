function mse end
function mae end
function logloss end
function poisson end
function gamma end
function tweedie end
function wmae end

@inline obs_metric(::typeof(mse), pk, yk; kwargs...) = (pk - yk)^2
@inline obs_metric(::typeof(mae), pk, yk; kwargs...) = abs(pk - yk)
@inline function obs_metric(::typeof(logloss), pk, yk; kwargs...)
    pred = sigmoid(pk)
    return -yk * log(pred) + (yk - 1) * log(1 - pred)
end
@inline function obs_metric(::typeof(poisson), pk, yk; kwargs...)
    pred = exp(pk)
    return 2 * (yk * (log(yk) - log(pred)) + pred - yk)
end
@inline function obs_metric(::typeof(gamma), pk, yk; kwargs...)
    pred = exp(pk)
    return 2 * (log(pred / yk) + yk / pred - 1)
end
@inline function obs_metric(::typeof(tweedie), pk, yk; kwargs...)
    rho = oftype(pk, 1.5)
    pred = exp(pk)
    return 2 * (
        yk^(2 - rho) / (1 - rho) / (2 - rho) -
        yk * pred^(1 - rho) / (1 - rho) +
        pred^(2 - rho) / (2 - rho)
    )
end
@inline function obs_metric(::typeof(wmae), pk, yk; alpha=0.5, kwargs...)
    return alpha * max(yk - pk, zero(pk)) + (1 - alpha) * max(pk - yk, zero(pk))
end

function apply_metric(
    metric,
    p::AbstractMatrix{T},
    y::AbstractVecOrMat{T},
    w::AbstractVector{T},
    eval::AbstractVector{T};
    kwargs...
) where {T}
    K = size(p, 1)
    @threads for i in eachindex(w)
        acc = zero(T)
        @inbounds for k in 1:K
            acc += obs_metric(metric, p[k, i], _target(y, k, i); kwargs...)
        end
        eval[i] = w[i] * acc / K
    end
    return sum(Float64, eval) / sum(Float64, w)
end

function mse(p::AbstractMatrix{T}, y::AbstractVecOrMat{T}, w::AbstractVector{T}, eval::AbstractVector{T}; kwargs...) where {T}
    return apply_metric(mse, p, y, w, eval; kwargs...)
end
rmse(p::AbstractMatrix{T}, y::AbstractVecOrMat, w::AbstractVector, eval::AbstractVector; kwargs...) where {T} =
    sqrt(mse(p, y, w, eval::AbstractVector; kwargs...))

function mae(
    p::AbstractMatrix{T},
    y::AbstractVecOrMat{T},
    w::AbstractVector{T},
    eval::AbstractVector{T};
    kwargs...
) where {T}
    return apply_metric(mae, p, y, w, eval; kwargs...)
end

function logloss(
    p::AbstractMatrix{T},
    y::AbstractVecOrMat{T},
    w::AbstractVector{T},
    eval::AbstractVector{T};
    kwargs...
) where {T}
    return apply_metric(logloss, p, y, w, eval; kwargs...)
end

function mlogloss(
    p::AbstractMatrix{T},
    y::AbstractVector{<:Integer},
    w::AbstractVector{T},
    eval::AbstractVector{T};
    kwargs...
) where {T}
    K = size(p, 1)
    @threads for i in eachindex(y)
        isum = zero(T)
        @inbounds for k in 1:K
            isum += exp(p[k, i])
        end
        @inbounds eval[i] = w[i] * (log(isum) - p[y[i], i])
    end
    return sum(Float64, eval) / sum(Float64, w)
end

function poisson(
    p::AbstractMatrix{T},
    y::AbstractVecOrMat{T},
    w::AbstractVector{T},
    eval::AbstractVector{T};
    kwargs...
) where {T}
    return apply_metric(poisson, p, y, w, eval; kwargs...)
end

function gamma(
    p::AbstractMatrix{T},
    y::AbstractVecOrMat{T},
    w::AbstractVector{T},
    eval::AbstractVector{T};
    kwargs...
) where {T}
    return apply_metric(gamma, p, y, w, eval; kwargs...)
end

function tweedie(
    p::AbstractMatrix{T},
    y::AbstractVecOrMat{T},
    w::AbstractVector{T},
    eval::AbstractVector{T};
    kwargs...
) where {T}
    return apply_metric(tweedie, p, y, w, eval; kwargs...)
end

function gaussian_mle(
    p::AbstractMatrix{T},
    y::AbstractVecOrMat{T},
    w::AbstractVector{T},
    eval::AbstractVector{T};
    kwargs...
) where {T}
    Y = size(p, 1) ÷ 2
    @threads for i in eachindex(w)
        acc = zero(T)
        @inbounds for t in 1:Y
            μ = p[2t-1, i]
            ls = p[2t, i]
            yt = y isa AbstractVector ? y[i] : y[t, i]
            acc += -(ls + (yt - μ)^2 / (2 * exp(2 * ls)))
        end
        eval[i] = w[i] * acc / Y
    end
    return sum(Float64, eval) / sum(Float64, w)
end

function logistic_mle(
    p::AbstractMatrix{T},
    y::AbstractVecOrMat{T},
    w::AbstractVector{T},
    eval::AbstractVector{T};
    kwargs...
) where {T}
    Y = size(p, 1) ÷ 2
    @threads for i in eachindex(w)
        acc = zero(T)
        @inbounds for t in 1:Y
            μ = p[2t-1, i]
            ls = p[2t, i]
            yt = y isa AbstractVector ? y[i] : y[t, i]
            acc += log(1 / 4 * sech(exp(-ls) * (yt - μ))^2) - ls
        end
        eval[i] = w[i] * acc / Y
    end
    return sum(Float64, eval) / sum(Float64, w)
end

function wmae(
    p::AbstractMatrix{T},
    y::AbstractVecOrMat{T},
    w::AbstractVector{T},
    eval::AbstractVector{T};
    alpha=0.5,
    kwargs...
) where {T}
    return apply_metric(wmae, p, y, w, eval; alpha, kwargs...)
end

function multiquantile(
    p::AbstractMatrix{T},
    y::AbstractVector{T},
    w::AbstractVector{T},
    eval::AbstractVector{T};
    alphas,
    kwargs...
) where {T}
    K = length(alphas)
    @assert size(p, 1) == K
    @threads for i in eachindex(y)
        yi = y[i]
        wi = w[i]
        acc = zero(T)
        @inbounds for k in 1:K
            diff = yi - p[k, i]
            alpha = alphas[k]
            acc += alpha * max(diff, zero(T)) + (1 - alpha) * max(-diff, zero(T))
        end
        eval[i] = wi * acc / K
    end
    return sum(Float64, eval) / sum(Float64, w)
end


function gini_raw(p::V, y::V) where {V<:AbstractVector}
    _y = y .- minimum(y)
    if length(_y) < 2
        return 0.0
    end
    random = cumsum(ones(length(p)) ./ length(p)^2)
    y_sort = _y[sortperm(p)]
    y_cum = cumsum(y_sort) ./ sum(_y) ./ length(p)
    gini = sum(Float64, random .- y_cum)
    return gini
end

function gini_norm(p::AbstractVector, y::AbstractVector)
    if length(y) < 2
        return 0.0
    end
    return gini_raw(y, p) / gini_raw(y, y)
end

function gini(
    p::AbstractMatrix{T},
    y::AbstractVector{T},
    w::AbstractVector{T},
    eval::AbstractVector{T};
    kwargs...
) where {T}
    return gini_norm(view(p, 1, :), y)
end

const metric_dict = Dict(
    :mse => mse,
    :rmse => rmse,
    :mae => mae,
    :logloss => logloss,
    :mlogloss => mlogloss,
    :poisson_deviance => poisson,
    :poisson => poisson,
    :gamma_deviance => gamma,
    :gamma => gamma,
    :tweedie_deviance => tweedie,
    :tweedie => tweedie,
    :gaussian_mle => gaussian_mle,
    :gaussian => gaussian_mle,
    :logistic_mle => logistic_mle,
    :wmae => wmae,
    :quantile => wmae,
    :multiquantile => multiquantile,
    :gini => gini,
)

is_maximise(::typeof(mse)) = false
is_maximise(::typeof(rmse)) = false
is_maximise(::typeof(mae)) = false
is_maximise(::typeof(logloss)) = false
is_maximise(::typeof(mlogloss)) = false
is_maximise(::typeof(poisson)) = false
is_maximise(::typeof(gamma)) = false
is_maximise(::typeof(tweedie)) = false
is_maximise(::typeof(gaussian_mle)) = true
is_maximise(::typeof(logistic_mle)) = true
is_maximise(::typeof(wmae)) = false
is_maximise(::typeof(multiquantile)) = false
is_maximise(::typeof(gini)) = true