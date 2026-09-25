function EvoTrees.init_core(params::EvoTrees.EvoTypes, device::Type{<:EvoTrees.GPU}, data, feature_names, y_train, w, offset, group=nothing)

    rng = Xoshiro(params.seed)
    edges, featbins, feattypes = EvoTrees.get_edges(data; feature_names, nbins=params.nbins, rng)
    backend = _gpu_backend(device)
    x_bin = EvoTrees.binarize(device, data; feature_names, edges)
    nobs, nfeats = size(x_bin)
    T = Float32
    L = EvoTrees._loss2type_dict[params.loss]

    K, y_cpu, μ, target_levels, target_isordered = EvoTrees._init_target(L, y_train, params, offset, T)
    y = _to_device(backend, y_cpu)
    μ = T.(μ)
    !isnothing(offset) && (μ .= 0)

    pred = KernelAbstractions.zeros(backend, T, K, nobs)
    pred .= _to_device(backend, μ)
    !isnothing(offset) && (pred .+= _to_device(backend, collect(offset')))

    ∇ = KernelAbstractions.zeros(backend, T, 2 * K + 1, nobs)
    h∇ = KernelAbstractions.zeros(backend, Float64, 2 * K + 1, maximum(featbins), length(featbins), 2^params.max_depth - 1)
    @assert (size(y, ndims(y)) == length(w) && minimum(w) > 0)
    ∇[end, :] .= w

    is_full = _to_device(backend, collect(UInt32, 1:nobs))
    mask_cpu = zeros(UInt8, nobs)
    mask_gpu = KernelAbstractions.zeros(backend, UInt8, nobs)
    js_ = UInt32.(collect(1:nfeats))
    n_sampled_feats = max(1, ceil(Int, params.colsample * nfeats))
    js = KernelAbstractions.zeros(backend, UInt32, n_sampled_feats)

    monotone_constraints = zeros(Int32, nfeats)
    hasproperty(params, :monotone_constraints) && for (k, v) in params.monotone_constraints
        monotone_constraints[k] = v
    end

    info = Dict(
        :nrounds => 0,
        :feature_names => feature_names,
        :target_levels => target_levels,
        :target_isordered => target_isordered,
        :edges => edges,
        :featbins => featbins,
        :feattypes => feattypes,
    )

    m = EvoTree{L,K}(L, K, μ, info)

    max_tree_nodes = 2^(params.max_depth + 1) - 1
    zeros_gpu(T, dims...) = KernelAbstractions.zeros(backend, T, dims...)

    group_cache = if isnothing(group)
        nothing
    else
        ng = EvoTrees.ngroups(group)
        GroupCacheGPU(
            group,
            _to_device(backend, group.group),
            zeros(UInt8, ng),
            KernelAbstractions.zeros(backend, UInt8, ng),
        )
    end

    cache = CacheBaseGPU{typeof(y),typeof(group_cache)}(
        rng,
        K,
        x_bin,
        y,
        w,
        pred,
        is_full,
        mask_cpu,
        mask_gpu,
        js_,
        js,
        ∇,
        h∇,
        feature_names,
        edges,
        featbins,
        _to_device(backend, feattypes),
        _to_device(backend, monotone_constraints),
        zeros_gpu(Bool, max_tree_nodes),
        zeros_gpu(UInt8, max_tree_nodes),
        zeros_gpu(Int32, max_tree_nodes),
        zeros_gpu(Float64, max_tree_nodes),
        zeros_gpu(Float64, 2 * K + 1, max_tree_nodes),
        zeros_gpu(Int32, max_tree_nodes),
        zeros_gpu(Float64, max_tree_nodes),
        zeros_gpu(Int32, max_tree_nodes),
        zeros_gpu(Int32, max_tree_nodes),
        zeros_gpu(Float64, n_sampled_feats, max_tree_nodes),
        zeros_gpu(Int32, n_sampled_feats, max_tree_nodes),
        zeros_gpu(Float64, 2 * K + 1, n_sampled_feats * max_tree_nodes),
        zeros_gpu(Float64, params.nbins, n_sampled_feats),
        zeros_gpu(Int32, params.nbins, n_sampled_feats),
        group_cache,
    )

    return m, cache
end

function EvoTrees.binarize(device::Type{<:EvoTrees.GPU}, X::Matrix{T}; feature_names, edges) where {T<:Real}
    backend = _gpu_backend(device)
    nobs, nfeats = size(X)
    x_bin = KernelAbstractions.zeros(backend, UInt8, nobs, nfeats)
    col = KernelAbstractions.allocate(backend, T, nobs)
    for j in 1:nfeats
        copyto!(col, 1, X, (j - 1) * nobs + 1, nobs)
        view(x_bin, :, j) .= UInt8.(searchsortedfirst.(Ref(_to_device(backend, edges[j])), col))
    end
    KernelAbstractions.synchronize(backend)
    return x_bin
end

EvoTrees.binarize(device::Type{<:EvoTrees.GPU}, data; feature_names, edges) =
    _to_device(_gpu_backend(device), EvoTrees.binarize(data; feature_names, edges))
