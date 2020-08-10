using Statistics
using StatsBase: sample
using Distributions
using Random
using CUDA
using Revise
using EvoTrees
using EvoTrees: sigmoid, logit
using Plots

# prepare a dataset
Random.seed!(123)
features = rand(10_000) .* 5
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
𝑖 = collect(1:size(X,1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

################################
# linear
################################
params1 = EvoTreeRegressor(T=Float32,
    loss=:linear, metric=:none,
    nrounds=200, nbins = 64,
    λ = 0.5, γ=0.1, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=0.5, colsample=1.0)

@time model = EvoTrees.fit_evotree_gpu(params1, X_train, Y_train);
@time pred_train_linear = EvoTrees.predict_gpu(model, X_train)

x_perm = sortperm(X_train[:,1])
plot(X_train, Y_train, msize = 1, mcolor = "gray", mswidth=0, background_color = RGB(1, 1, 1), seriestype=:scatter, xaxis = ("feature"), yaxis = ("target"), legend = true, label = "")
plot!(X_train[:,1][x_perm], pred_train_linear[x_perm], color = "navy", linewidth = 1.5, label = "Linear")
# savefig("figures/regression_sinus_gpu.png")


################################
# Gaussian
################################
params1 = EvoTreeGaussian(T=Float32,
    loss=:gaussian, metric=:none,
    nrounds=200, nbins=64,
    λ = 1.0, γ=0.1, η=0.1,
    max_depth = 5, min_weight = 100.0,
    rowsample=0.5, colsample=1.0, rng=123)

@time model = EvoTrees.fit_evotree_gpu(params1, X_train, Y_train);
@time pred_train_gauss = EvoTrees.predict_gpu(model, X_train)

pred_gauss = [Distributions.Normal(pred_train_gauss[i,1], pred_train_gauss[i,2]) for i in 1:size(pred_train_gauss,1)]
pred_q80 = quantile.(pred_gauss, 0.8)
pred_q20 = quantile.(pred_gauss, 0.2)

mean(Y_train .< pred_q80)
mean(Y_train .< pred_q20)

x_perm = sortperm(X_train[:,1])
plot(X_train[:, 1], Y_train, ms = 1, mcolor = "gray", mswidth=0, background_color = RGB(1, 1, 1), seriestype=:scatter, xaxis = ("feature"), yaxis = ("target"), legend = true, label = "")
plot!(X_train[:,1][x_perm], pred_train_gauss[x_perm, 1], color = "navy", linewidth = 1.5, label = "mu")
plot!(X_train[:,1][x_perm], pred_train_gauss[x_perm, 2], color = "darkred", linewidth = 1.5, label = "sigma")
plot!(X_train[:,1][x_perm], pred_q20[x_perm, 1], color = "green", linewidth = 1.5, label = "q20")
plot!(X_train[:,1][x_perm], pred_q80[x_perm, 1], color = "green", linewidth = 1.5, label = "q80")
# savefig("figures/gaussian_sinus_gpu.png")
