using BenchmarkTools
using Statistics
using StatsBase: sample, quantile
using Distributions
using Random
using CairoMakie
using CUDA
using EvoTrees
using EvoTrees: fit, predict, sigmoid, logit

# using ProfileView

# prepare a dataset
tree_type = :binary # binary/oblivious
_device = :gpu

Random.seed!(123)
features = rand(10_000) .* 5
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
is = collect(1:size(X, 1))

# train-eval split
i_sample = sample(is, size(is, 1), replace=false)
train_size = 0.8
i_train = i_sample[1:floor(Int, train_size * size(is, 1))]
i_eval = i_sample[floor(Int, train_size * size(is, 1))+1:end]

x_train, x_eval = X[i_train, :], X[i_eval, :]
y_train, y_eval = Y[i_train], Y[i_eval]

# mse
config = EvoTreeRegressor(;
    nrounds=500,
    early_stopping_rounds=50,
    nbins=64,
    L2=0.0,
    eta=0.1,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    device=_device
)

@time model = fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
);

@time pred_train_linear = predict(model, x_train)
mean(abs.(pred_train_linear .- y_train))
sqrt(mean((pred_train_linear .- y_train) .^ 2))

# logistic / cross-entropy
config = EvoTreeRegressor(;
    loss=:logloss,
    nrounds=500,
    early_stopping_rounds=50,
    nbins=64,
    L2=1.0,
    eta=0.1,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    device=_device
)

@time model = fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
);
@time pred_train_logistic = model(x_train; device=_device)
sqrt(mean((pred_train_logistic .- y_train) .^ 2))

# poisson
config = EvoTreeCount(;
    nrounds=500,
    early_stopping_rounds=50,
    nbins=64,
    L2=1.0,
    eta=0.1,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    device=_device
)

@time model = fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
);
@time pred_train_poisson = model(x_train; device=_device)
sqrt(mean((pred_train_poisson .- y_train) .^ 2))

# gamma
config = EvoTreeRegressor(;
    loss=:gamma,
    nrounds=500,
    early_stopping_rounds=50,
    nbins=64,
    L2=1.0,
    eta=0.1,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    device=_device
)

@time model = fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
);
@time pred_train_gamma = model(x_train; device=_device)
sqrt(mean((pred_train_gamma .- y_train) .^ 2))

# tweedie
config = EvoTreeRegressor(;
    loss=:tweedie,
    nrounds=500,
    early_stopping_rounds=50,
    nbins=64,
    L2=1.0,
    eta=0.1,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    device=_device
)

@time model = fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
);
@time pred_train_tweedie = model(x_train; device=_device)
sqrt(mean((pred_train_tweedie .- y_train) .^ 2))

# MAE
config = EvoTreeRegressor(;
    loss=:mae,
    nrounds=500,
    early_stopping_rounds=50,
    nbins=64,
    L2=0.0,
    eta=0.1,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    device=_device
)

@time model = fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
);
@time pred_train_mae = model(x_train; device=_device)
sqrt(mean((pred_train_mae .- y_train) .^ 2))

###########################################
# plot
###########################################
x_perm = sortperm(x_train[:, 1])
f = Figure()
ax = Axis(f[1, 1], xlabel="feature", ylabel="target")
scatter!(ax,
    x_train[x_perm, 1],
    y_train[x_perm],
    color="#BBB",
    markersize=2)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_linear[x_perm],
    color="navy",
    linewidth=1,
    label="mse",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_logistic[x_perm],
    color="darkred",
    linewidth=1,
    label="logloss",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_poisson[x_perm],
    color="green",
    linewidth=1,
    label="poisson",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_gamma[x_perm],
    color="pink",
    linewidth=1,
    label="gamma",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_tweedie[x_perm],
    color="orange",
    linewidth=1,
    label="tweedie",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_mae[x_perm],
    color="lightblue",
    linewidth=1,
    label="mae",
)
Legend(f[2, 1], ax; halign=:left, orientation=:horizontal)
f
save("docs/src/assets/regression-sinus-$tree_type-$_device.png", f)

###############################
## gaussian
###############################
println("\n" * "="^60)
println("STARTING GAUSSIAN MLE SECTION")
println("="^60)
config = EvoTreeGaussian(;
    nrounds=500,
    early_stopping_rounds=50,
    nbins=64,
    L2=1.0,
    # gamma=0.1,
    eta=0.1,
    max_depth=6,
    min_weight=8,
    rowsample=0.5,
    colsample=1.0,
    rng=123,
    tree_type,
    device=_device
)

println("=== GAUSSIAN MLE DEBUG ===")
println("Device: $_device, Tree type: $tree_type")
println("Config: nrounds=$(config.nrounds), eta=$(config.eta), L2=$(config.L2), min_weight=$(config.min_weight)")
println("Training data: x_train size=$(size(x_train)), y_train size=$(size(y_train))")
println("y_train stats: min=$(minimum(y_train)), max=$(maximum(y_train)), mean=$(mean(y_train)), std=$(std(y_train))")

# Let's also try CPU for comparison
println("\n--- TESTING CPU VERSION FOR COMPARISON ---")
config_cpu = EvoTreeGaussian(;
    nrounds=100,
    early_stopping_rounds=25,
    nbins=64,
    L2=1.0,
    eta=0.1,
    max_depth=6,
    min_weight=8,
    rowsample=0.5,
    colsample=1.0,
    rng=123,
    tree_type,
    device=:cpu
)

@time model_cpu = fit(
    config_cpu;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
);
pred_cpu = model_cpu(x_train; device=:cpu)
println("CPU μ stats: min=$(minimum(pred_cpu[:, 1])), max=$(maximum(pred_cpu[:, 1])), mean=$(mean(pred_cpu[:, 1]))")
println("CPU σ stats: min=$(minimum(pred_cpu[:, 2])), max=$(maximum(pred_cpu[:, 2])), mean=$(mean(pred_cpu[:, 2]))")
println("--- END CPU COMPARISON ---\n")

@time model = fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
);

# Check if model has logger info
if haskey(model.info, :logger) && !isnothing(model.info[:logger])
    logger = model.info[:logger]
    println("Final metric: $(logger[:metrics][end]) (should be increasing for Gaussian MLE)")
    println("Best metric: $(logger[:best_metric]) at iteration $(logger[:best_iter])")
    println("Total rounds: $(logger[:nrounds])")
else
    println("No logger information available")
end

@time pred_train_gaussian = model(x_train; device=_device)

println("Predictions shape: $(size(pred_train_gaussian))")
println("μ (col 1) stats: min=$(minimum(pred_train_gaussian[:, 1])), max=$(maximum(pred_train_gaussian[:, 1])), mean=$(mean(pred_train_gaussian[:, 1]))")
println("σ (col 2) stats: min=$(minimum(pred_train_gaussian[:, 2])), max=$(maximum(pred_train_gaussian[:, 2])), mean=$(mean(pred_train_gaussian[:, 2]))")
println("σ exp stats: min=$(minimum(exp.(pred_train_gaussian[:, 2]))), max=$(maximum(exp.(pred_train_gaussian[:, 2]))), mean=$(mean(exp.(pred_train_gaussian[:, 2])))")

pred_gauss = [
    Distributions.Normal(pred_train_gaussian[i, 1], pred_train_gaussian[i, 2]) for
    i in axes(pred_train_gaussian, 1)
]
pred_q80 = quantile.(pred_gauss, 0.8)
pred_q20 = quantile.(pred_gauss, 0.2)

# Validate quantile coverage
coverage_q80 = mean(y_train .< pred_q80)
coverage_q20 = mean(y_train .< pred_q20)
println("Q80 coverage: $(coverage_q80) (should be ~0.8)")
println("Q20 coverage: $(coverage_q20) (should be ~0.2)")

# Check if predictions make sense
println("Sample predictions (first 5):")
for i in 1:min(5, length(y_train))
    μ, σ = pred_train_gaussian[i, 1], exp(pred_train_gaussian[i, 2])
    println("  y[$i]=$(y_train[i]), μ=$μ, σ=$σ, q20=$(pred_q20[i]), q80=$(pred_q80[i])")
end

println("=== END GAUSSIAN DEBUG ===\n")

###########################################
# plot
###########################################
x_perm = sortperm(x_train[:, 1])
f = Figure()
ax = Axis(f[1, 1], xlabel="feature", ylabel="target")
scatter!(ax,
    x_train[x_perm, 1],
    y_train[x_perm],
    color="#BBB",
    markersize=2)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_gaussian[x_perm, 1],
    color="navy",
    linewidth=1,
    label="mu",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_gaussian[x_perm, 2],
    color="darkred",
    linewidth=1,
    label="sigma",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_q20[x_perm, 1],
    color="green",
    linewidth=1,
    label="q20",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_q80[x_perm, 1],
    color="green",
    linewidth=1,
    label="q80",
)
Legend(f[2, 1], ax; halign=:left, orientation=:horizontal)
save("docs/src/assets/gaussian-sinus-$tree_type-$_device.png", f)

###############################
## Quantiles
###############################
# q50
params1 = EvoTreeRegressor(;
    loss=:quantile,
    alpha=0.5,
    nrounds=500,
    nbins=64,
    eta=0.1,
    # L2=1.0,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    early_stopping_rounds=50,
    device=_device
)
@time model = fit(
    params1;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
);
# 116.822 ms (74496 allocations: 36.41 MiB) for 100 iterations
# @btime model = grow_gbtree($X_train, $Y_train, $params1, X_eval = $X_eval, Y_eval = $Y_eval)
@time pred_train_q50 = model(x_train)
@info sum(pred_train_q50 .< y_train) / length(y_train)

# q20
params1 = EvoTreeRegressor(;
    loss=:quantile,
    alpha=0.2,
    nrounds=500,
    nbins=64,
    eta=0.1,
    L2=1.0,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    early_stopping_rounds=50,
    device=_device
)
@time model = fit(params1; x_train, y_train, x_eval, y_eval, print_every_n=25);
@time pred_train_q20 = model(x_train)
@info sum(pred_train_q20 .> y_train) / length(y_train)

# q80
params1 = EvoTreeRegressor(;
    loss=:quantile,
    alpha=0.8,
    nrounds=500,
    nbins=64,
    L2=1.0,
    eta=0.2,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    early_stopping_rounds=50,
    device=_device
)
@time model = fit(params1; x_train, y_train, x_eval, y_eval, print_every_n=25)
@time pred_train_q80 = model(x_train)
@info sum(pred_train_q80 .> y_train) / length(y_train)

x_perm = sortperm(x_train[:, 1])
f = Figure()
ax = Axis(f[1, 1], xlabel="feature", ylabel="target")
scatter!(ax,
    x_train[x_perm, 1],
    y_train[x_perm],
    color="#BBB",
    markersize=2)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_q50[x_perm],
    color="navy",
    linewidth=1,
    label="Median",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_q20[x_perm],
    color="darkred",
    linewidth=1,
    label="Q20",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_q80[x_perm],
    color="darkgreen",
    linewidth=1,
    label="Q80",
)
Legend(f[2, 1], ax; halign=:left, orientation=:horizontal)
f
save("docs/src/assets/quantiles-sinus-$tree_type-$_device.png", f)

###############################
# credibility losses
###############################
# cred_var
config = EvoTreeRegressor(;
    loss=:cred_var,
    nrounds=500,
    early_stopping_rounds=50,
    nbins=64,
    L2=1.0,
    lambda=1.0,
    eta=0.1,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    device=_device
)

@time model = fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
);
@time pred_train_cred_var = model(x_train; device=_device)
sqrt(mean((pred_train_cred_var .- y_train) .^ 2))

# cred_std
config = EvoTreeRegressor(;
    loss=:cred_std,
    nrounds=500,
    early_stopping_rounds=50,
    nbins=64,
    L2=1.0,
    lambda=1.0,
    eta=0.1,
    max_depth=6,
    min_weight=1.0,
    rowsample=0.5,
    colsample=1.0,
    tree_type,
    device=_device
)

@time model = fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    print_every_n=25,
);
@time pred_train_cred_std = model(x_train; device=_device)
sqrt(mean((pred_train_cred_std .- y_train) .^ 2))

###########################################
# plot credibility
###########################################
x_perm = sortperm(x_train[:, 1])
f = Figure()
ax = Axis(f[1, 1], xlabel="feature", ylabel="target")
scatter!(ax,
    x_train[x_perm, 1],
    y_train[x_perm],
    color="#BBB",
    markersize=2)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_cred_var[x_perm],
    color="navy",
    linewidth=1,
    label="cred_var",
)
lines!(ax,
    x_train[x_perm, 1],
    pred_train_cred_std[x_perm],
    color="darkred",
    linewidth=1,
    label="cred_std",
)
Legend(f[2, 1], ax; halign=:left, orientation=:horizontal)
save("docs/src/assets/credibility-sinus-$tree_type-$_device.png", f)

