using DataFrames
using CSV
using Statistics
using Base.Threads: @threads
using BenchmarkTools
using Profile
using StatsBase: sample

using Revise
using Traceur
using EvoTrees
using EvoTrees: get_gain, update_gains!, get_max_gain, update_grads!, grow_tree!, grow_gbtree, SplitInfo2, Tree, Node, Params, predict, find_split!, SplitTrack, update_track!, sigmoid

# prepare a dataset
data = CSV.read("./data/performance_tot_v2_perc.csv", allowmissing = :auto)
names(data)

features = data[1:53]
X = convert(Array, features)
Y = data[54]
Y = convert(Array{Float64}, Y)
𝑖 = collect(1:size(X,1))

# train-eval split
𝑖_sample = StatsBase.sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]


# idx
X_perm = zeros(Int, size(X))
@threads for feat in 1:size(X, 2)
    X_perm[:, feat] = sortperm(X[:, feat]) # returns gain value and idx split
    # idx[:, feat] = sortperm(view(X, :, feat)) # returns gain value and idx split
end

# placeholder for sort perm
perm_ini = zeros(Int, size(X))

# set parameters
nrounds = 1
λ = 1.0
γ = 1e-3
η = 0.5
max_depth = 5
min_weight = 5.0
rowsample = 1.0
colsample = 1.0

# params1 = Params(nrounds, λ, γ, η, max_depth, min_weight, :linear)
params1 = Params(:linear, 1, λ, γ, 1.0, 5, min_weight, rowsample, colsample)

# initial info
δ, δ² = zeros(size(X, 1)), zeros(size(X, 1))
pred = zeros(size(Y, 1))
@time update_grads!(Val{params1.loss}(), pred, Y, δ, δ²)
∑δ, ∑δ² = sum(δ), sum(δ²)

gain = get_gain(∑δ, ∑δ², params1.λ)
𝑖 = collect(1:size(X,1))
root = Node(1, ∑δ, ∑δ², gain, 0, 0.0, 0, 0, - ∑δ / (∑δ² + params1.λ), 𝑖)
tree = Tree([root])
grow_tree!(tree, X, δ, δ², params1, perm_ini)
# tree = Tree([root])
# @btime grow_tree!(tree, X, δ, δ², params1, perm_ini)

# predict - map a sample to tree-leaf prediction
tree
pred = predict(tree, X)
# pred = sigmoid(pred)
mean((pred .- Y) .^ 2)
# println(sort(unique(pred)))

function test_grow(n, X, δ, δ², perm_ini, params)
    for i in 1:n
        tree = Tree([Node(1, ∑δ, ∑δ², gain, 0, 0.0, 0, 0, - ∑δ / (∑δ² + params1.λ), 𝑖)])
        grow_tree!(tree, X, δ, δ², params, perm_ini)
        # grow_tree!(tree, view(X, :, :), view(δ, :), view(δ², :), params1)
    end
end

@time test_grow(1, X, δ, δ², perm_ini, params1)
@time test_grow(10, X, δ, δ², perm_ini, params1)
@time test_grow(100, X, δ, δ², perm_ini, params1)

# full model
params1 = Params(:linear, 100, λ, γ, 0.05, 5, min_weight, rowsample, colsample)
@time model = grow_gbtree(X, Y, params1)

# predict - map a sample to tree-leaf prediction
pred = predict(model, X)
# pred = sigmoid(pred)
mean((pred .- Y) .^ 2)


# train model
params1 = Params(:linear, 100, λ, γ, 0.05, 5, min_weight, rowsample, colsample)

@time model = grow_gbtree(X_train, Y_train, params2, X_eval = X_eval, Y_eval = Y_eval)
pred = predict(model, X)
mean((pred .- Y) .^ 2)


X_bin = convert(Array{UInt8}, round.(X*64))
@time test_grow(1, X_bin, δ, δ², perm_ini, params1)
@time test_grow(10, X_bin, δ, δ², perm_ini, params1)
@time test_grow(100, X_bin, δ, δ², perm_ini, params1)
@time model = grow_gbtree(X_bin, Y, params1)

X_train_bin = convert(Array{UInt8}, round.(X_train*64))
X_eval_bin = convert(Array{UInt8}, round.(X_eval*64))

@time model = grow_gbtree(X_train_bin, Y_train, params1, X_eval = X_eval_bin, Y_eval = Y_eval)
# predict - map a sample to tree-leaf prediction
pred = predict(model, X_eval_bin)
mean((pred .- Y_eval) .^ 2)
