function EvoTrees.init_core(params::EvoTrees.EvoTypes, ::Type{<:EvoTrees.GPU}, data, fnames, y_train, w, offset)

    edges, featbins, feattypes = EvoTrees.get_edges(data; feature_names=fnames, nbins=params.nbins, rng=params.rng)
    xb = EvoTrees.binarize(data; feature_names=fnames, edges)
    x_bin = CuArray(xb)
    nobs, nfeats = size(x_bin)
    T = Float32
    L = EvoTrees._loss2type_dict[params.loss]

    target_levels = nothing
    target_isordered = false
    if L == EvoTrees.LogLoss
        @assert eltype(y_train) <: Real && minimum(y_train) >= 0 && maximum(y_train) <= 1
        K = 1
        y = T.(y_train)
        μ = [EvoTrees.logit(EvoTrees.mean(y))]
        !isnothing(offset) && (offset .= EvoTrees.logit.(offset))
    elseif L in [EvoTrees.Poisson, EvoTrees.Gamma, EvoTrees.Tweedie]
        @assert eltype(y_train) <: Real
        K = 1
        y = T.(y_train)
        μ = fill(log(EvoTrees.mean(y)), 1)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == EvoTrees.MLogLoss
        if eltype(y_train) <: EvoTrees.CategoricalValue
            target_levels = EvoTrees.CategoricalArrays.levels(y_train)
            target_isordered = EvoTrees.isordered(y_train)
            y = UInt32.(EvoTrees.CategoricalArrays.levelcode.(y_train))
        else
            target_levels = sort(unique(y_train))
            target_isordered = false
            yc = EvoTrees.CategoricalVector(y_train, levels=target_levels)
            y = UInt32.(EvoTrees.CategoricalArrays.levelcode.(yc))
        end
        K = length(target_levels)
        μ = T.(log.(EvoTrees.proportions(y, UInt32(1):UInt32(K))))
        μ .-= maximum(μ)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == EvoTrees.GaussianMLE
        @assert eltype(y_train) <: Real
        K = 2
        y = T.(y_train)
        μ = [EvoTrees.mean(y), log(EvoTrees.std(y))]
        !isnothing(offset) && (offset[:, 2] .= log.(offset[:, 2]))
    else # MSE, MAE, Quantile, etc.
        @assert eltype(y_train) <: Real
        K = 1
        y = T.(y_train)
        μ = [EvoTrees.mean(y)]
    end
    y = CuArray(y)
    μ = T.(μ)
    !isnothing(offset) && (μ .= 0)

    backend = KernelAbstractions.get_backend(x_bin)
    pred = KernelAbstractions.zeros(backend, T, K, nobs)
    mu_dev = KernelAbstractions.zeros(backend, T, K, 1)
    copyto!(mu_dev, reshape(T.(μ), K, 1))
    pred .= mu_dev
    if !isnothing(offset)
        offT = T.(offset')
        off_dev = KernelAbstractions.zeros(backend, T, size(offT, 1), size(offT, 2))
        copyto!(off_dev, offT)
        pred .+= off_dev
    end

    ∇ = KernelAbstractions.zeros(backend, T, 2 * K + 1, nobs)
    h∇ = KernelAbstractions.zeros(backend, Float32, 2 * K + 1, params.nbins, nfeats, 2^params.max_depth - 1)
    
    @assert (length(y) == length(w) && minimum(w) > 0)
    ∇[end, :] .= w

    nidx = KernelAbstractions.ones(backend, UInt32, nobs)
    is_in = KernelAbstractions.zeros(backend, UInt32, nobs)
    is_out = KernelAbstractions.zeros(backend, UInt32, nobs)
    mask = KernelAbstractions.zeros(backend, UInt8, nobs)
    js_ = UInt32.(collect(1:nfeats))
    js = KernelAbstractions.zeros(backend, UInt32, ceil(Int, params.colsample * nfeats))

    monotone_constraints = zeros(Int32, nfeats)
    hasproperty(params, :monotone_constraints) && for (k, v) in params.monotone_constraints
        monotone_constraints[k] = v
    end

    info = Dict(
        :nrounds => 0,
        :feature_names => fnames,
        :target_levels => target_levels,
        :target_isordered => target_isordered,
        :edges => edges,
        :featbins => featbins,
        :feattypes => feattypes,
    )

    m = EvoTree{L,K}(L, K, [EvoTrees.Tree{L,K}(μ)], info)

    feattypes_gpu = CuArray(feattypes)
    monotone_constraints_gpu = CuArray(monotone_constraints)

    max_nodes_level = 2^params.max_depth
    left_nodes_buf = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    right_nodes_buf = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    target_mask_buf = KernelAbstractions.zeros(backend, UInt8, 2^(params.max_depth + 1))

    max_tree_nodes = 2^params.max_depth - 1
    tree_split_gpu = KernelAbstractions.zeros(backend, Bool, max_tree_nodes)
    tree_cond_bin_gpu = KernelAbstractions.zeros(backend, UInt8, max_tree_nodes)
    tree_feat_gpu = KernelAbstractions.zeros(backend, Int32, max_tree_nodes)
    tree_gain_gpu = KernelAbstractions.zeros(backend, Float64, max_tree_nodes)
    tree_pred_gpu = KernelAbstractions.zeros(backend, Float32, K, max_tree_nodes)
    
    max_nodes_total = 2^(params.max_depth + 1)
    # Allocate nodes_sum as 2D to match kernels indexing nodes_sum[k, node]
    nodes_sum_gpu = KernelAbstractions.zeros(backend, Float32, 2*K+1, max_nodes_total)
    
    nodes_gain_gpu = KernelAbstractions.zeros(backend, Float32, max_nodes_total)
    anodes_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    n_next_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level * 2)
    n_next_active_gpu = KernelAbstractions.zeros(backend, Int32, 1)
    best_gain_gpu = KernelAbstractions.zeros(backend, Float32, max_nodes_level)
    best_bin_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    best_feat_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    build_nodes_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    subtract_nodes_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    build_count = KernelAbstractions.zeros(backend, Int32, 1)
    subtract_count = KernelAbstractions.zeros(backend, Int32, 1)

    cache = CacheGPU(
        info, x_bin, y, CuArray(w), K, nothing, pred, nidx, is_in, is_out, mask,
        js_, js, ∇, h∇, nothing, nothing, fnames, edges, featbins, feattypes_gpu,
        nothing, nothing, nothing, nothing, monotone_constraints_gpu,
        left_nodes_buf, right_nodes_buf, target_mask_buf, tree_split_gpu,
        tree_cond_bin_gpu, tree_feat_gpu, tree_gain_gpu, tree_pred_gpu,
        nodes_sum_gpu, nodes_gain_gpu, anodes_gpu, n_next_gpu,
        n_next_active_gpu, best_gain_gpu, best_bin_gpu, best_feat_gpu,
        build_nodes_gpu, subtract_nodes_gpu, build_count, subtract_count
    )
    
    return m, cache
end

