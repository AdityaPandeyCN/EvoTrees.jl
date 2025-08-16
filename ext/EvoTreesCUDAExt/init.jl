using CUDA
using KernelAbstractions

function EvoTrees.init_core(
    params::EvoTrees.EvoTypes{L}, 
    ::Type{<:EvoTrees.GPU}, 
    data, 
    fnames, 
    y_train, 
    w, 
    offset
) where {L}
    
    # Binarize data into quantiles
    edges, featbins, feattypes = EvoTrees.get_edges(
        data; 
        fnames=fnames, 
        nbins=params.nbins, 
        rng=params.rng
    )
    x_bin = CuArray(EvoTrees.binarize(data; fnames=fnames, edges=edges))
    nobs, nfeats = size(x_bin)
    T = Float32  # GPU computation type
    
    # Process target variable based on loss type
    target_levels = nothing
    if L == EvoTrees.MLogLoss
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
    else  # Regression losses (MSE, Logistic, Quantile, etc.)
        K = 1
        y = T.(y_train)
        μ = [EvoTrees.mean(y)]
    end
    
    # Transfer to GPU
    y = CuArray(y)
    μ = T.(μ)
    !isnothing(offset) && (μ .= T(0))
    
    # Initialize predictions
    pred = CUDA.zeros(T, K, nobs)
    pred .= CuArray(μ)
    !isnothing(offset) && (pred .+= CuArray(T.(offset')))
    
    # Initialize gradients (2K for gradients/hessians, 1 for weights)
    ∇ = CUDA.zeros(T, 2 * K + 1, nobs)
    
    # Set weights
    if isnothing(w)
        ∇[end, :] .= T(1)
    else
        ∇[end, :] .= CuArray(T.(w))
    end
    
    # Handle monotone constraints
    monotone_constraints = hasfield(typeof(params), :monotone_constraints) ? 
                          params.monotone_constraints : Dict{Int,Int}()
    monotone_constraints_array = zeros(Int32, nfeats)
    for (feat, direction) in monotone_constraints
        if feat <= nfeats
            monotone_constraints_array[feat] = Int32(direction)
        end
    end
    
    # Package GPU parameters
    gpu_params = (
        lambda = T(params.lambda),
        min_weight = T(params.min_weight),
        gamma = T(params.gamma),
        alpha = T(hasfield(typeof(params), :alpha) ? params.alpha : 0.5),
        eta = T(params.eta),
        max_depth = Int32(params.max_depth),
        nbins = Int32(params.nbins),
        rowsample = T(params.rowsample),
        colsample = T(params.colsample),
    )
    
    # Pre-allocate GPU arrays
    max_nodes = 2^(params.max_depth + 1) - 1
    max_active_nodes = 2^params.max_depth
    
    # Histogram arrays
    h∇ = CUDA.zeros(T, 3, params.nbins, nfeats, max_nodes)
    h∇L = CUDA.zeros(T, 3, params.nbins, nfeats, max_nodes)
    
    # Tree structure arrays
    tree_split_gpu = CUDA.zeros(Bool, max_nodes)
    tree_cond_bin_gpu = CUDA.zeros(UInt32, max_nodes)
    tree_feat_gpu = CUDA.zeros(Int32, max_nodes)
    tree_gain_gpu = CUDA.zeros(T, max_nodes)
    tree_pred_gpu = CUDA.zeros(T, max_nodes)
    
    # Node statistics
    nodes_sum_gpu = CUDA.zeros(T, 3, max_nodes)
    nodes_gain_gpu = CUDA.zeros(T, max_nodes)
    
    # Active nodes tracking
    anodes_gpu = CUDA.zeros(Int32, max_active_nodes)
    n_next_gpu = CUDA.zeros(Int32, max_active_nodes * 2)
    n_next_active_gpu = CUDA.zeros(Int32, 1)
    
    # Best split buffers
    best_gain_gpu = CUDA.zeros(T, max_active_nodes)
    best_bin_gpu = CUDA.zeros(Int32, max_active_nodes)
    best_feat_gpu = CUDA.zeros(Int32, max_active_nodes)
    
    # Temporary arrays for split finding
    gains_feats = CUDA.zeros(T, max_active_nodes, nfeats)
    bins_feats = CUDA.zeros(Int32, max_active_nodes, nfeats)
    
    # Build cache
    cache = (
        info = Dict(:nrounds => Ref(0)),
        x_bin = x_bin,
        y = y,
        w = isnothing(w) ? CUDA.ones(T, nobs) : CuArray(T.(w)),
        K = K,
        pred = pred,
        ∇ = ∇,
        edges = edges,
        feattypes_gpu = CuArray(feattypes),
        monotone_constraints_gpu = CuArray(monotone_constraints_array),
        gpu_params = gpu_params,
        
        # Sampling indices
        nidx = CUDA.ones(UInt32, nobs),
        is_in = CUDA.zeros(UInt32, nobs),
        is_out = CUDA.zeros(UInt32, nobs),
        mask = CUDA.zeros(UInt8, nobs),
        js_ = UInt32.(1:nfeats),
        js = zeros(UInt32, ceil(Int, params.colsample * nfeats)),
        
        # GPU arrays
        h∇ = h∇,
        h∇L = h∇L,
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
    
    # Initialize model
    info = Dict(:fnames => fnames, :target_levels => target_levels)
    bias = [EvoTrees.Tree{L,K}(μ)]
    m = EvoTree{L,K}(bias, info)
    
    return m, cache
end

