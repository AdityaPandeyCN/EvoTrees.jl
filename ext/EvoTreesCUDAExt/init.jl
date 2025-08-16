function EvoTrees.init_core(params::EvoTrees.EvoTypes{L}, ::Type{<:EvoTrees.GPU}, data, fnames, y_train, w, offset) where {L}

    # binarize data into quantiles
    edges, featbins, feattypes = EvoTrees.get_edges(data; fnames, nbins=params.nbins, rng=params.rng)
    x_bin = CuArray(EvoTrees.binarize(data; fnames, edges))
    nobs, nfeats = size(x_bin)
    T = Float32  # GPU computation type

    # Standard target variable processing
    target_levels = nothing
    if L == EvoTrees.Logistic
        K = 1
        y = T.(y_train)
        μ = [T(EvoTrees.logit(EvoTrees.mean(y)))]
        !isnothing(offset) && (offset .= T.(EvoTrees.logit.(offset)))
    elseif L in [EvoTrees.Poisson, EvoTrees.Gamma, EvoTrees.Tweedie]
        K = 1
        y = T.(y_train)
        μ = [T(log(EvoTrees.mean(y)))]
        !isnothing(offset) && (offset .= T.(log.(offset)))
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
        !isnothing(offset) && (offset .= T.(log.(offset)))
    else # MSE, Quantile, etc.
        K = 1
        y = T.(y_train)
        μ = [T(EvoTrees.mean(y))]
    end
    
    # Transfer to GPU with correct types
    y = CuArray(y)
    μ = T.(μ)
    !isnothing(offset) && (μ .= T(0))

    # Initialize predictions
    pred = CUDA.zeros(T, K, nobs)
    pred .= CuArray(μ)
    !isnothing(offset) && (pred .+= CuArray(T.(offset')))

    # Initialize gradients with correct type
    ∇ = CUDA.zeros(T, 2 * K + 1, nobs)
    @assert (length(y) == length(w) && minimum(w) > 0) "Invalid weights"
    ∇[end, :] .= CuArray(T.(w))

    # Initialize indexes
    nidx = CUDA.ones(UInt32, nobs)
    is_in = CUDA.zeros(UInt32, nobs)
    is_out = CUDA.zeros(UInt32, nobs)
    mask = CUDA.zeros(UInt8, nobs)
    js_ = UInt32.(collect(1:nfeats))
    js = zeros(UInt32, ceil(Int, params.colsample * nfeats))

    # Assign monotone constraints
    monotone_constraints = zeros(Int32, nfeats)
    if hasproperty(params, :monotone_constraints) && !isnothing(params.monotone_constraints)
        for (k, v) in params.monotone_constraints
            if k <= nfeats && k >= 1
                monotone_constraints[k] = Int32(v)
            end
        end
    end

    # Model info
    info = Dict(
        :fnames => fnames,
        :target_levels => target_levels,
        :edges => edges,
        :feattypes => feattypes,
    )
    bias = [EvoTrees.Tree{L,K}(μ)]
    m = EvoTree{L,K}(bias, info)

    # GPU arrays
    feattypes_gpu = CuArray(feattypes)
    monotone_constraints_gpu = CuArray(monotone_constraints)

    # PRE-CONVERT ALL PARAMETERS TO GPU TYPES
    gpu_params = (
        lambda = T(params.lambda),
        min_weight = T(params.min_weight),
        gamma = T(params.gamma),
        alpha = T(get(params, :alpha, 0.5)),
        eta = T(params.eta),
        max_depth = Int32(params.max_depth),
        nbins = Int32(params.nbins),
        rowsample = T(params.rowsample),
        colsample = T(params.colsample),
    )

    ### PRE-ALLOCATE ALL GPU ARRAYS ###
    max_nodes = 2^(params.max_depth + 1) - 1
    @assert max_nodes <= 2^20 "max_depth too large"

    # Histograms - use T type consistently
    h∇ = CUDA.zeros(T, 2 * K + 1, params.nbins, nfeats, max_nodes)

    # Tree structure arrays
    tree_split_gpu = CUDA.zeros(Bool, max_nodes)
    tree_cond_bin_gpu = CUDA.zeros(UInt32, max_nodes)
    tree_feat_gpu = CUDA.zeros(Int32, max_nodes)
    tree_gain_gpu = CUDA.zeros(T, max_nodes)
    tree_pred_gpu = CUDA.zeros(T, max_nodes)

    # Node statistics - use T for consistency
    nodes_sum_gpu = CUDA.zeros(T, 3, max_nodes)
    nodes_gain_gpu = CUDA.zeros(T, max_nodes)
    
    # Active node tracking
    anodes_gpu = CUDA.zeros(Int32, max_nodes)
    n_next_gpu = CUDA.zeros(Int32, max_nodes)
    n_next_active_gpu = CUDA.zeros(Int32, 1)

    # Best split tracking
    best_gain_gpu = CUDA.zeros(T, max_nodes)
    best_bin_gpu = CUDA.zeros(Int32, max_nodes)
    best_feat_gpu = CUDA.zeros(Int32, max_nodes)

    # Temporary arrays
    max_active_nodes = min(max_nodes, 2^params.max_depth)
    gains_feats = CUDA.zeros(T, max_active_nodes, nfeats)
    bins_feats = CUDA.zeros(Int32, max_active_nodes, nfeats)

    # Build cache
    cache = (
        info = info,
        nrounds = Ref(0),
        x_bin = x_bin,
        y = y,
        w = CuArray(T.(w)),
        K = K,
        pred = pred,
        nidx = nidx,
        is_in = is_in,
        is_out = is_out,
        mask = mask,
        js_ = js_,
        js = js,
        ∇ = ∇,
        h∇ = h∇,
        fnames = fnames,
        edges = edges,
        feattypes_gpu = feattypes_gpu,
        monotone_constraints_gpu = monotone_constraints_gpu,
        gpu_params = gpu_params,  # Pre-converted parameters!
        tree_split_gpu = tree_split_gpu,
        tree_cond_bin_gpu = tree_cond_bin_gpu,
        tree_feat_gpu = tree_feat_gpu,
        tree_gain_gpu = tree_gain_gpu,
        tree_pred_gpu = tree_pred_gpu,
        nodes_sum_gpu = nodes_sum_gpu,
        nodes_gain_gpu = nodes_gain_gpu,
        anodes_gpu = anodes_gpu,
        n_next_gpu = n_next_gpu,
        n_next_active_gpu = n_next_active_gpu,
        best_gain_gpu = best_gain_gpu,
        best_bin_gpu = best_bin_gpu,
        best_feat_gpu = best_feat_gpu,
        gains_feats = gains_feats,
        bins_feats = bins_feats
    )
    
    return m, cache
end

