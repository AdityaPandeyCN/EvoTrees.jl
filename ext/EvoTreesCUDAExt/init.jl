function EvoTrees.init_core(params::EvoTrees.EvoTypes{L}, ::Type{<:EvoTrees.GPU}, data, fnames, y_train, w, offset) where {L}

    # binarize data into quantiles
    edges, featbins, feattypes = EvoTrees.get_edges(data; fnames, nbins=params.nbins, rng=params.rng)
    x_bin = CuArray(EvoTrees.binarize(data; fnames, edges))
    nobs, nfeats = size(x_bin)
    T = Float32

    # Standard target variable processing...
    target_levels = nothing
    if L == EvoTrees.Logistic
        K = 1; y = T.(y_train); μ = [EvoTrees.logit(EvoTrees.mean(y))]
        !isnothing(offset) && (offset .= EvoTrees.logit.(offset))
    elseif L in [EvoTrees.Poisson, EvoTrees.Gamma, EvoTrees.Tweedie]
        K = 1; y = T.(y_train); μ = fill(log(EvoTrees.mean(y)), 1)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == EvoTrees.MLogLoss
        if eltype(y_train) <: EvoTrees.CategoricalValue
            target_levels = EvoTrees.CategoricalArrays.levels(y_train)
            y = UInt32.(EvoTrees.CategoricalArrays.levelcode.(y_train))
        else
            target_levels = sort(unique(y_train))
            yc = EvoTrees.CategoricalVector(y_train, levels=target_levels)
            y = UInt32.(EvoTrees.CategoricalArrays.levelcode.(yc))
        end
        K = length(target_levels)
        μ = T.(log.(EvoTrees.proportions(y, UInt32(1):UInt32(K))))
        μ .-= maximum(μ)
        !isnothing(offset) && (offset .= log.(offset))
    else # MSE, Quantile, etc.
        K = 1; y = T.(y_train); μ = [EvoTrees.mean(y)]
    end
    y = CuArray(y)
    μ = T.(μ)
    !isnothing(offset) && (μ .= 0)

    # initialize preds
    pred = CUDA.zeros(T, K, nobs)
    pred .= CuArray(μ)
    !isnothing(offset) && (pred .+= CuArray(offset'))

    # initialize gradients
    ∇ = CUDA.zeros(T, 2 * K + 1, nobs)
    @assert (length(y) == length(w) && minimum(w) > 0)
    ∇[end, :] .= w

    # initialize indexes
    nidx = CUDA.ones(UInt32, nobs)
    is_in = CUDA.zeros(UInt32, nobs)
    is_out = CUDA.zeros(UInt32, nobs)
    mask = CUDA.zeros(UInt8, nobs)
    js_ = UInt32.(collect(1:nfeats))
    js = zeros(eltype(js_), ceil(Int, params.colsample * nfeats))

    # assign monotone contraints
    monotone_constraints = zeros(Int32, nfeats)
    hasproperty(params, :monotone_constraints) && for (k, v) in params.monotone_constraints
        monotone_constraints[k] = v
    end

    # model info
    info = Dict(
        :fnames => fnames,
        :target_levels => target_levels,
        :edges => edges,
        :feattypes => feattypes,
    )
    bias = [EvoTrees.Tree{L,K}(μ)]
    m = EvoTree{L,K}(bias, info)

    feattypes_gpu = CuArray(feattypes)
    monotone_constraints_gpu = CuArray(monotone_constraints)

    ### PRE-ALLOCATE ALL GPU ARRAYS FOR THE TRAINING LOOP ###
    
    # Safe sizing for all potential nodes in a full binary tree
    max_nodes = 2^(params.max_depth + 1)

    # Histograms
    h∇ = CUDA.zeros(Float32, 2 * K + 1, params.nbins, nfeats, max_nodes)

    # Final tree structure arrays
    tree_split_gpu = CUDA.zeros(Bool, max_nodes)
    tree_cond_bin_gpu = CUDA.zeros(UInt32, max_nodes)
    tree_feat_gpu = CUDA.zeros(Int32, max_nodes)
    tree_gain_gpu = CUDA.zeros(T, max_nodes)
    tree_pred_gpu = CUDA.zeros(T, max_nodes)

    # Node-level statistics and tracking
    nodes_sum_gpu = CUDA.zeros(Float64, 3, max_nodes)
    nodes_gain_gpu = CUDA.zeros(T, max_nodes)
    anodes_gpu = CUDA.zeros(Int32, max_nodes) # Active nodes
    n_next_gpu = CUDA.zeros(Int32, max_nodes) # Next level's active nodes
    n_next_active_gpu = CUDA.zeros(Int32, 1) # Counter

    # Best split found for each active node
    best_gain_gpu = CUDA.zeros(T, max_nodes)
    best_bin_gpu = CUDA.zeros(Int32, max_nodes)
    best_feat_gpu = CUDA.zeros(Int32, max_nodes)

    # Temporary arrays for the map-reduce split finding strategy
    gains_feats = CUDA.zeros(T, max_nodes, nfeats)
    bins_feats = CUDA.zeros(Int32, max_nodes, nfeats)

    # build cache
    cache = (
        info=Dict(:nrounds => 0), x_bin=x_bin, y=y, w=w, K=K,
        pred=pred, nidx=nidx, is_in=is_in, is_out=is_out, mask=mask,
        js_=js_, js=js, ∇=∇, h∇=h∇, fnames=fnames, edges=edges,
        feattypes_gpu=feattypes_gpu, monotone_constraints_gpu=monotone_constraints_gpu,
        
        # Add all pre-allocated arrays to the cache
        tree_split_gpu=tree_split_gpu, tree_cond_bin_gpu=tree_cond_bin_gpu,
        tree_feat_gpu=tree_feat_gpu, tree_gain_gpu=tree_gain_gpu,
        tree_pred_gpu=tree_pred_gpu, nodes_sum_gpu=nodes_sum_gpu,
        nodes_gain_gpu=nodes_gain_gpu, anodes_gpu=anodes_gpu,
        n_next_gpu=n_next_gpu, n_next_active_gpu=n_next_active_gpu,
        best_gain_gpu=best_gain_gpu, best_bin_gpu=best_bin_gpu,
        best_feat_gpu=best_feat_gpu, gains_feats=gains_feats,
        bins_feats=bins_feats
    )
    return m, cache
end

