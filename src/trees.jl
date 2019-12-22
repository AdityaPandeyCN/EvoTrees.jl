# initialize train_nodes
function grow_tree(δ, δ², 𝑤,
    hist_δ, hist_δ², hist_𝑤,
    params::Union{EvoTreeRegressor,EvoTreeCount,EvoTreeClassifier,EvoTreeGaussian},
    train_nodes::Vector{TrainNode{L,T,S}},
    splits::Vector{SplitInfo{L,T,Int}},
    edges, X_bin) where {R<:Real, T<:AbstractFloat, S<:Int, L}

    active_id = ones(Int, 1)
    leaf_count = one(Int)
    tree_depth = one(Int)
    tree = Tree(Vector{TreeNode{params.K, T, Int, Bool}}())

    # grow while there are remaining active nodes
    while size(active_id, 1) > 0 && tree_depth <= params.max_depth
        next_active_id = ones(Int, 0)
        # grow nodes
        for id in active_id
            node = train_nodes[id]
            if tree_depth == params.max_depth || node.∑𝑤[1] <= params.min_weight
                push!(tree.nodes, TreeNode(pred_leaf(params.loss, node, params, δ²)))
            else
                @threads for feat in node.𝑗
                    splits[feat].gain = node.gain
                    find_split_static!(hist_δ[feat], hist_δ²[feat], hist_𝑤[feat], view(X_bin,:,feat), δ, δ², 𝑤, node.∑δ, node.∑δ², node.∑𝑤, params, splits[feat], edges[feat], node.𝑖)
                    # update_hist!(hist_δ, hist_δ², hist_𝑤, X_bin, δ, δ², 𝑤, set, feat)
                    # find_split!(hist_δ[feat], hist_δ²[feat], hist_𝑤[feat], node.∑δ, node.∑δ², node.∑𝑤, params, splits[feat], edges[feat], feat)
                end
                # assign best split
                best = get_max_gain(splits)
                # grow node if best split improve gain
                if best.gain > node.gain + params.γ
                    # Node: depth, ∑δ, ∑δ², gain, 𝑖, 𝑗
                    # set = BitSet(node.𝑖)
                    left, right = update_set(node.𝑖, best.𝑖, view(X_bin,:,best.feat))
                    train_nodes[leaf_count + 1] = TrainNode(node.depth + 1, best.∑δL, best.∑δ²L, best.∑𝑤L, best.gainL, left, node.𝑗)
                    train_nodes[leaf_count + 2] = TrainNode(node.depth + 1, best.∑δR, best.∑δ²R, best.∑𝑤R, best.gainR, right, node.𝑗)
                    # push split Node
                    push!(tree.nodes, TreeNode(leaf_count + 1, leaf_count + 2, best.feat, best.cond, params.K))
                    push!(next_active_id, leaf_count + 1)
                    push!(next_active_id, leaf_count + 2)
                    leaf_count += 2
                else
                    push!(tree.nodes, TreeNode(pred_leaf(params.loss, node, params, δ²)))
                end # end of single node split search
            end
        end # end of loop over active ids for a given depth
        active_id = next_active_id
        tree_depth += 1
    end # end of tree growth
    return tree
end

# extract the gain value from the vector of best splits and return the split info associated with best split
function get_max_gain(splits::Vector{SplitInfo{L,T,S}}) where {L,T,S}
    gains = (x -> x.gain).(splits)
    feat = findmax(gains)[2]
    best = splits[feat]
    return best
end

# grow_gbtree
function grow_gbtree(X::AbstractArray{R, 2}, Y::AbstractVector{S}, params::Union{EvoTreeRegressor,EvoTreeCount,EvoTreeClassifier,EvoTreeGaussian};
    X_eval::AbstractArray{R, 2} = Array{R, 2}(undef, (0,0)), Y_eval::AbstractVector{S} = Vector{S}(undef, 0),
    early_stopping_rounds=Int(1e5), print_every_n=100, verbosity=1) where {R<:Real, S<:Real}

    seed!(params.seed)

    μ = ones(params.K)
    μ .*= mean(Y)
    if typeof(params.loss) == Logistic
        μ .= logit.(μ)
    elseif typeof(params.loss) == Poisson
        μ .= log.(μ)
    elseif typeof(params.loss) == Softmax
        μ .*= 0.0
    elseif typeof(params.loss) == Gaussian
        μ = SVector{2}([mean(Y), log(var(Y))])
    end

    # initialize preds
    pred = zeros(SVector{params.K,Float64}, size(X,1))
    for i in eachindex(pred)
        pred[i] += μ
    end

    # eval init
    if size(Y_eval, 1) > 0
        # pred_eval = ones(size(Y_eval, 1), params.K) .* μ'
        pred_eval = zeros(SVector{params.K,Float64}, size(X_eval,1))
        for i in eachindex(pred_eval)
            pred_eval[i] += μ
        end
    end

    # bias = Tree([TreeNode(SVector{1, Float64}(μ))])
    bias = Tree([TreeNode(SVector{params.K,Float64}(μ))])
    gbtree = GBTree([bias], params, Metric())

    X_size = size(X)
    𝑖_ = collect(1:X_size[1])
    𝑗_ = collect(1:X_size[2])

    # initialize gradients and weights
    δ = zeros(SVector{params.K, Float64}, X_size[1])
    δ² = zeros(SVector{params.K, Float64}, X_size[1])
    𝑤 = zeros(SVector{1, Float64}, X_size[1])
    for i in eachindex(𝑤)
        𝑤[i] += ones(1)
    end

    edges = get_edges(X, params.nbins)
    X_bin = binarize(X, edges)

    # initialize train nodes
    train_nodes = Vector{TrainNode{params.K, Float64, Int64}}(undef, 2^params.max_depth-1)
    for node in 1:2^params.max_depth-1
        train_nodes[node] = TrainNode(0, SVector{params.K, Float64}(fill(-Inf, params.K)), SVector{params.K, Float64}(fill(-Inf, params.K)), SVector{1, Float64}(fill(-Inf, 1)), -Inf, [0], [0])
    end

    # initializde node splits info and tracks - colsample size (𝑗)
    splits = Vector{SplitInfo{params.K, Float64, Int64}}(undef, X_size[2])
    hist_δ = Vector{Vector{SVector{params.K, Float64}}}(undef, X_size[2])
    hist_δ² = Vector{Vector{SVector{params.K, Float64}}}(undef, X_size[2])
    hist_𝑤 = Vector{Vector{SVector{1, Float64}}}(undef, X_size[2])
    for feat in 𝑗_
        splits[feat] = SplitInfo{params.K, Float64, Int}(-Inf, SVector{params.K, Float64}(zeros(params.K)), SVector{params.K, Float64}(zeros(params.K)), SVector{1, Float64}(zeros(1)), SVector{params.K, Float64}(zeros(params.K)), SVector{params.K, Float64}(zeros(params.K)), SVector{1, Float64}(zeros(1)), -Inf, -Inf, 0, feat, 0.0)
        hist_δ[feat] = zeros(SVector{params.K, Float64}, length(edges[feat]))
        hist_δ²[feat] = zeros(SVector{params.K, Float64}, length(edges[feat]))
        hist_𝑤[feat] = zeros(SVector{1, Float64}, length(edges[feat]))
    end

    # initialize metric
    if params.metric != :none
        metric_track = Metric()
        metric_best = Metric()
        iter_since_best = 0
    end

    # loop over nrounds
    for i in 1:params.nrounds
        # select random rows and cols
        𝑖 = 𝑖_[sample(𝑖_, ceil(Int, params.rowsample * X_size[1]), replace=false, ordered=true)]
        𝑗 = 𝑗_[sample(𝑗_, ceil(Int, params.colsample * X_size[2]), replace=false, ordered=true)]

        # reset gain to -Inf
        for feat in 𝑗_
            splits[feat].gain = -Inf
        end

        # get gradients
        update_grads!(params.loss, params.α, pred, Y, δ, δ², 𝑤)
        ∑δ, ∑δ², ∑𝑤 = sum(δ[𝑖]), sum(δ²[𝑖]), sum(𝑤[𝑖])
        gain = get_gain(params.loss, ∑δ, ∑δ², ∑𝑤, params.λ)

        # assign a root and grow tree
        train_nodes[1] = TrainNode(1, ∑δ, ∑δ², ∑𝑤, gain, 𝑖, 𝑗)
        tree = grow_tree(δ, δ², 𝑤, hist_δ, hist_δ², hist_𝑤, params, train_nodes, splits, edges, X_bin)
        # push new tree to model
        push!(gbtree.trees, tree)
        # update predictions
        predict!(pred, tree, X)
        # eval predictions
        if size(Y_eval, 1) > 0
            predict!(pred_eval, tree, X_eval)
        end

        # callback function
        if params.metric != :none
            if size(Y_eval, 1) > 0
                metric_track.metric .= eval_metric(Val{params.metric}(), pred_eval, Y_eval, params.α)
            else
                metric_track.metric .= eval_metric(Val{params.metric}(), pred, Y, params.α)
            end

            if metric_track.metric < metric_best.metric
                metric_best.metric .=  metric_track.metric
                metric_best.iter .=  i
            else
                iter_since_best += 1
            end

            if mod(i, print_every_n) == 0 && verbosity > 0
                display(string("iter:", i, ", eval: ", metric_track.metric))
            end
            iter_since_best >= early_stopping_rounds ? break : nothing
        end # end of callback

    end #end of nrounds

    if params.metric != :none
        gbtree.metric.iter .= metric_best.iter
        gbtree.metric.metric .= metric_best.metric
    end
    return gbtree
end

# grow_gbtree - continue training
function grow_gbtree!(model::GBTree, X::AbstractArray{R, 2}, Y::AbstractVector{S};
    X_eval::AbstractArray{R, 2} = Array{R, 2}(undef, (0,0)), Y_eval::AbstractVector{S} = Vector{S}(undef, 0),
    early_stopping_rounds=Int(1e5), print_every_n=100, verbosity=1) where {R<:Real, S<:Real}

    params = model.params
    seed!(params.seed)

    # initialize predictions - efficiency to be improved
    pred = zeros(SVector{params.K,Float64}, size(X,1))
    pred_ = predict(model, X)
    for i in eachindex(pred)
        pred[i] = SVector{params.K,Float64}(pred_[i])
    end
    # eval init
    if size(Y_eval, 1) > 0
        pred_eval = zeros(SVector{params.K,Float64}, size(X_eval,1))
        pred_eval_ = predict(model, X_eval)
        for i in eachindex(pred_eval)
            pred_eval[i] = SVector{params.K,Float64}(pred_eval_[i])
        end
    end

    X_size = size(X)
    𝑖_ = collect(1:X_size[1])
    𝑗_ = collect(1:X_size[2])

    # initialize gradients and weights
    δ, δ² = zeros(SVector{params.K, Float64}, X_size[1]), zeros(SVector{params.K, Float64}, X_size[1])
    𝑤 = zeros(SVector{1, Float64}, X_size[1]) .+ 1

    edges = get_edges(X, params.nbins)
    X_bin = binarize(X, edges)

    # initialize train nodes
    train_nodes = Vector{TrainNode{params.K, Float64, Int64}}(undef, 2^params.max_depth-1)
    for node in 1:2^params.max_depth-1
        train_nodes[node] = TrainNode(0, SVector{params.K, Float64}(fill(-Inf, params.K)), SVector{params.K, Float64}(fill(-Inf, params.K)), SVector{1, Float64}(fill(-Inf, 1)), -Inf, [0], [0])
    end

    # initializde node splits info and tracks - colsample size (𝑗)
    splits = Vector{SplitInfo{params.K, Float64, Int64}}(undef, X_size[2])
    hist_δ = Vector{Vector{SVector{params.K, Float64}}}(undef, X_size[2])
    hist_δ² = Vector{Vector{SVector{params.K, Float64}}}(undef, X_size[2])
    hist_𝑤 = Vector{Vector{SVector{1, Float64}}}(undef, X_size[2])
    for feat in 𝑗_
        splits[feat] = SplitInfo{params.K, Float64, Int}(-Inf, SVector{params.K, Float64}(zeros(params.K)), SVector{params.K, Float64}(zeros(params.K)), SVector{1, Float64}(zeros(1)), SVector{params.K, Float64}(zeros(params.K)), SVector{params.K, Float64}(zeros(params.K)), SVector{1, Float64}(zeros(1)), -Inf, -Inf, 0, feat, 0.0)
        hist_δ[feat] = zeros(SVector{params.K, Float64}, length(edges[feat]))
        hist_δ²[feat] = zeros(SVector{params.K, Float64}, length(edges[feat]))
        hist_𝑤[feat] = zeros(SVector{1, Float64}, length(edges[feat]))
    end

    # initialize metric
    if params.metric != :none
        metric_track = model.metric
        metric_best = model.metric
        iter_since_best = 0
    end

    # loop over nrounds
    for i in 1:params.nrounds
        # select random rows and cols
        𝑖 = 𝑖_[sample(𝑖_, ceil(Int, params.rowsample * X_size[1]), replace=false, ordered=true)]
        𝑗 = 𝑗_[sample(𝑗_, ceil(Int, params.colsample * X_size[2]), replace=false, ordered=true)]

        # reset gain to -Inf
        for feat in 𝑗_
            splits[feat].gain = -Inf
        end

        # get gradients
        update_grads!(params.loss, params.α, pred, Y, δ, δ², 𝑤)
        ∑δ, ∑δ², ∑𝑤 = sum(δ[𝑖]), sum(δ²[𝑖]), sum(𝑤[𝑖])
        gain = get_gain(params.loss, ∑δ, ∑δ², ∑𝑤, params.λ)

        # assign a root and grow tree
        train_nodes[1] = TrainNode(1, ∑δ, ∑δ², ∑𝑤, gain, 𝑖, 𝑗)
        tree = grow_tree(δ, δ², 𝑤, hist_δ, hist_δ², hist_𝑤, params, train_nodes, splits, edges, X_bin)

        # update push tree to model
        push!(model.trees, tree)

        # get update predictions
        predict!(pred, tree, X)
        # eval predictions
        if size(Y_eval, 1) > 0
            predict!(pred_eval, tree, X_eval)
        end

        # callback function
        if params.metric != :none

            if size(Y_eval, 1) > 0
                metric_track.metric .= eval_metric(Val{params.metric}(), pred_eval, Y_eval, params.α)
            else
                metric_track.metric .= eval_metric(Val{params.metric}(), pred, Y, params.α)
            end

            if metric_track.metric < metric_best.metric
                metric_best.metric .=  metric_track.metric
                metric_best.iter .=  i
            else
                iter_since_best += 1
            end

            if mod(i, print_every_n) == 0 && verbosity > 0
                display(string("iter:", i, ", eval: ", metric_track.metric))
            end
            iter_since_best >= early_stopping_rounds ? break : nothing
        end
    end #end of nrounds

    if params.metric != :none
        model.metric.iter .= metric_best.iter
        model.metric.metric .= metric_best.metric
    end
    return model
end




# grow_gbtree
function grow_gbtree_MLJ(X::AbstractMatrix{R}, Y::AbstractVector{S}, params::Union{EvoTreeRegressor,EvoTreeCount,EvoTreeClassifier,EvoTreeGaussian}; verbosity=1) where {R<:Real, S<:Real}

    seed!(params.seed)

    μ = ones(params.K)
    μ .*= mean(Y)
    if typeof(params.loss) == Logistic
        μ .= logit.(μ)
    elseif typeof(params.loss) == Poisson
        μ .= log.(μ)
    elseif typeof(params.loss) == Softmax
        μ .*= 0.0
    elseif typeof(params.loss) == Gaussian
        μ = SVector{2}([mean(Y), log(var(Y))])
    end

    # initialize preds
    pred = zeros(SVector{params.K,Float64}, size(X,1))
    for i in eachindex(pred)
        pred[i] += μ
    end

    # bias = Tree([TreeNode(SVector{1, Float64}(μ))])
    bias = Tree([TreeNode(SVector{params.K,Float64}(μ))])
    gbtree = GBTree([bias], params, Metric())

    X_size = size(X)
    𝑖_ = collect(1:X_size[1])
    𝑗_ = collect(1:X_size[2])

    # initialize gradients and weights
    δ, δ² = zeros(SVector{params.K, Float64}, X_size[1]), zeros(SVector{params.K, Float64}, X_size[1])
    𝑤 = zeros(SVector{1, Float64}, X_size[1]) .+ 1

    edges = get_edges(X, params.nbins)
    X_bin = binarize(X, edges)

    # initialize train nodes
    train_nodes = Vector{TrainNode{params.K, Float64, Int64}}(undef, 2^params.max_depth-1)
    for node in 1:2^params.max_depth-1
        train_nodes[node] = TrainNode(0, SVector{params.K, Float64}(fill(-Inf, params.K)), SVector{params.K, Float64}(fill(-Inf, params.K)), SVector{1, Float64}(fill(-Inf, 1)), -Inf, [0], [0])
    end

    # initializde node splits info and tracks - colsample size (𝑗)
    splits = Vector{SplitInfo{params.K, Float64, Int64}}(undef, X_size[2])
    hist_δ = Vector{Vector{SVector{params.K, Float64}}}(undef, X_size[2])
    hist_δ² = Vector{Vector{SVector{params.K, Float64}}}(undef, X_size[2])
    hist_𝑤 = Vector{Vector{SVector{1, Float64}}}(undef, X_size[2])
    for feat in 𝑗_
        splits[feat] = SplitInfo{params.K, Float64, Int}(-Inf, SVector{params.K, Float64}(zeros(params.K)), SVector{params.K, Float64}(zeros(params.K)), SVector{1, Float64}(zeros(1)), SVector{params.K, Float64}(zeros(params.K)), SVector{params.K, Float64}(zeros(params.K)), SVector{1, Float64}(zeros(1)), -Inf, -Inf, 0, feat, 0.0)
        hist_δ[feat] = zeros(SVector{params.K, Float64}, length(edges[feat]))
        hist_δ²[feat] = zeros(SVector{params.K, Float64}, length(edges[feat]))
        hist_𝑤[feat] = zeros(SVector{1, Float64}, length(edges[feat]))
    end

    # loop over nrounds
    for i in 1:params.nrounds
        # select random rows and cols
        𝑖 = 𝑖_[sample(𝑖_, ceil(Int, params.rowsample * X_size[1]), replace=false, ordered=true)]
        𝑗 = 𝑗_[sample(𝑗_, ceil(Int, params.colsample * X_size[2]), replace=false, ordered=true)]

        # reset gain to -Inf
        for feat in 𝑗_
            splits[feat].gain = -Inf
        end

        # get gradients
        update_grads!(params.loss, params.α, pred, Y, δ, δ², 𝑤)
        ∑δ, ∑δ², ∑𝑤 = sum(δ[𝑖]), sum(δ²[𝑖]), sum(𝑤[𝑖])
        gain = get_gain(params.loss, ∑δ, ∑δ², ∑𝑤, params.λ)

        # assign a root and grow tree
        train_nodes[1] = TrainNode(1, ∑δ, ∑δ², ∑𝑤, gain, 𝑖, 𝑗)
        tree = grow_tree(δ, δ², 𝑤, hist_δ, hist_δ², hist_𝑤, params, train_nodes, splits, edges, X_bin)
        # push new tree to model
        push!(gbtree.trees, tree)

        # get update predictions
        predict!(pred, tree, X)

    end #end of nrounds

    cache = (params=deepcopy(params), X=X, Y=Y, pred=pred, 𝑖_=𝑖_, 𝑗_=𝑗_, δ=δ, δ²=δ², 𝑤=𝑤, edges=edges, X_bin=X_bin,
        train_nodes=train_nodes, splits=splits, hist_δ=hist_δ, hist_δ²=hist_δ², hist_𝑤=hist_𝑤)
    # cache = (deepcopy(params), X, Y, pred, 𝑖_, 𝑗_, δ, δ², 𝑤, edges, X_bin, train_nodes, splits, hist_δ, hist_δ², hist_𝑤)
    return gbtree, cache
end

# continue training for MLJ - continue training from same dataset - all preprocessed elements passed as cache
function grow_gbtree_MLJ!(model::GBTree, cache; verbosity=1)

    params = model.params

    # initialize predictions
    # cache_params, X, Y, pred, 𝑖_, 𝑗_, δ, δ², 𝑤, edges, X_bin, train_nodes, splits, hist_δ, hist_δ², hist_𝑤 = cache
    train_nodes = cache.train_nodes
    splits = cache.splits

    X_size = size(cache.X_bin)
    δnrounds = params.nrounds - cache.params.nrounds
    # println("MLJ! δnrounds: ", δnrounds)

    # loop over nrounds
    for i in 1:δnrounds

        # select random rows and cols
        𝑖 = cache.𝑖_[sample(cache.𝑖_, ceil(Int, params.rowsample * X_size[1]), replace=false, ordered=true)]
        𝑗 = cache.𝑗_[sample(cache.𝑗_, ceil(Int, params.colsample * X_size[2]), replace=false, ordered=true)]

        # reset gain to -Inf
        for feat in cache.𝑗_
            splits[feat].gain = -Inf
        end

        # get gradients
        update_grads!(params.loss, params.α, cache.pred, cache.Y, cache.δ, cache.δ², cache.𝑤)
        ∑δ, ∑δ², ∑𝑤 = sum(cache.δ[𝑖]), sum(cache.δ²[𝑖]), sum(cache.𝑤[𝑖])
        gain = get_gain(params.loss, ∑δ, ∑δ², ∑𝑤, params.λ)

        # assign a root and grow tree
        train_nodes[1] = TrainNode(1, ∑δ, ∑δ², ∑𝑤, gain, 𝑖, 𝑗)
        tree = grow_tree(cache.δ, cache.δ², cache.𝑤, cache.hist_δ, cache.hist_δ², cache.hist_𝑤, params, train_nodes, splits, cache.edges, cache.X_bin)

        # update push tree to model
        push!(model.trees, tree)

        # get update predictions
        predict!(cache.pred, tree, cache.X)

    end #end of nrounds


    cache.params.nrounds = params.nrounds
    # cache = (deepcopy(params), X, Y, pred, 𝑖_, 𝑗_, δ, δ², 𝑤, edges, X_bin, train_nodes, splits, hist_δ, hist_δ², hist_𝑤)

    # return model, cache
    return model
end
