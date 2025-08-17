function EvoTrees.grow_evotree!(evotree::EvoTree{L,K}, cache, params::EvoTrees.EvoTypes{L}, ::Type{<:EvoTrees.GPU}) where {L,K}
    EvoTrees.update_grads!(cache.∇, cache.pred, cache.y, params)
    is = EvoTrees.subsample(cache.is_in, cache.is_out, cache.mask, params.rowsample, params.rng)
    EvoTrees.sample!(params.rng, cache.js_, cache.js, replace=false, ordered=true)

    tree = EvoTrees.Tree{L,K}(params.max_depth)
    grow! = params.tree_type == "oblivious" ? grow_otree! : grow_tree!
    grow!(
        tree,
        params,
        cache.∇,
        cache.edges,
        cache.nidx,
        is,
        cache.js,
        cache.h∇,
        cache.h∇L,
        cache.h∇R,
        cache.x_bin,
        cache.feattypes_gpu,
    )
    push!(evotree.trees, tree)
    EvoTrees.predict!(cache.pred, tree, cache.x_bin, cache.feattypes_gpu)
    cache[:info][:nrounds] += 1
    return nothing
end

function grow_tree!(
    tree::EvoTrees.Tree{L,K},
    params::EvoTrees.EvoTypes{L},
    ∇::CuMatrix,
    edges,
    nidx::CuVector,
    is,
    js,
    h∇::CuArray,
    h∇L::CuArray,
    h∇R::CuArray,
    x_bin::CuMatrix,
    feattypes_gpu::CuVector{Bool},
) where {L,K}

    backend = KernelAbstractions.get_backend(x_bin)
    js_gpu = KernelAbstractions.adapt(backend, js)
    is_gpu = KernelAbstractions.adapt(backend, is)

    tree_split_gpu = KernelAbstractions.zeros(backend, Bool, length(tree.split))
    tree_cond_bin_gpu = KernelAbstractions.zeros(backend, UInt32, length(tree.cond_bin))
    tree_feat_gpu = KernelAbstractions.zeros(backend, Int32, length(tree.feat))
    tree_gain_gpu = KernelAbstractions.zeros(backend, Float64, length(tree.gain))
    tree_pred_gpu = KernelAbstractions.zeros(backend, Float32, length(tree.pred))

    max_nodes_total = 2^(params.max_depth + 1)
    nodes_sum_gpu = KernelAbstractions.zeros(backend, Float64, 3, max_nodes_total)
    nodes_gain_gpu = KernelAbstractions.zeros(backend, Float64, max_nodes_total)

    max_nodes_level = 2^params.max_depth
    anodes_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    n_next_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level * 2)
    n_next_active_gpu = CuArray([0])

    best_gain_gpu = KernelAbstractions.zeros(backend, Float64, max_nodes_level)
    best_bin_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    best_feat_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    
    nidx .= 1
    
    nsamples = Float32(length(is_gpu))
    root_sums_cpu = zeros(Float64, 3)
    root_sums_cpu[1] = sum(view(∇, 1, is_gpu))
    root_sums_cpu[2] = sum(view(∇, 2, is_gpu))
    root_sums_cpu[3] = nsamples
    copyto!(view(nodes_sum_gpu, :, 1), root_sums_cpu)

    get_gain_gpu!(backend)(nodes_gain_gpu, nodes_sum_gpu, CuArray([1]), params.lambda; ndrange=1)
    
    copyto!(view(anodes_gpu, 1:1), [1])
    
    n_active = 1

    for depth in 1:params.max_depth
        !iszero(n_active) || break
        
        n_nodes_level = 2^(depth - 1)
        active_nodes_full = view(anodes_gpu, 1:n_nodes_level)
        
        if n_active < n_nodes_level
            view(anodes_gpu, n_active+1:n_nodes_level) .= 0
        end

        view_gain = view(best_gain_gpu, 1:n_nodes_level)
        view_bin  = view(best_bin_gpu, 1:n_nodes_level)
        view_feat = view(best_feat_gpu, 1:n_nodes_level)
        
        update_hist_gpu!(
            h∇, h∇L, h∇R,
            view_gain, view_bin, view_feat,
            ∇, x_bin, nidx, js_gpu,
            depth, active_nodes_full, nodes_sum_gpu, params
        )

        n_next_active_gpu .= 0
        view_gain_act  = view(view_gain, 1:n_active)
        view_bin_act   = view(view_bin, 1:n_active)
        view_feat_act  = view(view_feat, 1:n_active)

        active_nodes_act = view(active_nodes_full, 1:n_active)

        apply_splits_kernel!(backend)(
            tree_split_gpu, tree_cond_bin_gpu, tree_feat_gpu, tree_gain_gpu, tree_pred_gpu,
            nodes_sum_gpu, nodes_gain_gpu,
            n_next_gpu, n_next_active_gpu,
            view_gain_act, view_bin_act, view_feat_act,
            h∇L,
            active_nodes_act,
            depth, params.max_depth, params.lambda, params.gamma;
            ndrange = n_active
        )
        
        n_active = Int(Array(n_next_active_gpu)[1])
        if n_active > 0
            copyto!(view(anodes_gpu, 1:n_active), view(n_next_gpu, 1:n_active))
        end

        if depth < params.max_depth && n_active > 0
            update_nodes_idx_kernel!(backend)(
                nidx, is_gpu, x_bin, tree_feat_gpu, tree_cond_bin_gpu, feattypes_gpu;
                ndrange = length(is_gpu)
            )
        end
    end

    copyto!(tree.split, Array(tree_split_gpu))
    copyto!(tree.cond_bin, Array(tree_cond_bin_gpu))
    copyto!(tree.feat, Array(tree_feat_gpu))
    copyto!(tree.gain, Array(tree_gain_gpu))
    copyto!(tree.pred, Array(tree_pred_gpu))
    
    for i in eachindex(tree.split)
        if tree.split[i]
            tree.cond_float[i] = edges[tree.feat[i]][tree.cond_bin[i]]
        end
    end

    return nothing
end

@kernel function apply_splits_kernel!(
    tree_split, tree_cond_bin, tree_feat, tree_gain, tree_pred,
    nodes_sum, nodes_gain,
    n_next, n_next_active,
    best_gain, best_bin, best_feat,
    h∇L,
    active_nodes,
    depth, max_depth, lambda, gamma
)
    n_idx = @index(Global)
    node = active_nodes[n_idx]

    @inbounds if depth < max_depth && best_gain[n_idx] > nodes_gain[node] + gamma
        tree_split[node] = true
        tree_cond_bin[node] = best_bin[n_idx]
        tree_feat[node] = best_feat[n_idx]
        tree_gain[node] = best_gain[n_idx]

        child_l, child_r = node << 1, (node << 1) + 1
        feat, bin = Int(tree_feat[node]), Int(tree_cond_bin[node])

        nodes_sum[1, child_l] = h∇L[1, bin, feat, node]
        nodes_sum[2, child_l] = h∇L[2, bin, feat, node]
        nodes_sum[3, child_l] = h∇L[3, bin, feat, node]
        
        nodes_sum[1, child_r] = nodes_sum[1, node] - nodes_sum[1, child_l]
        nodes_sum[2, child_r] = nodes_sum[2, node] - nodes_sum[2, child_l]
        nodes_sum[3, child_r] = nodes_sum[3, node] - nodes_sum[3, child_l]

        p1_l, p2_l = nodes_sum[1, child_l], nodes_sum[2, child_l]
        nodes_gain[child_l] = p1_l^2 / (p2_l + lambda + 1e-8)
        p1_r, p2_r = nodes_sum[1, child_r], nodes_sum[2, child_r]
        nodes_gain[child_r] = p1_r^2 / (p2_r + lambda + 1e-8)
        
        idx_base = Atomix.@atomic n_next_active[1] += 2
        n_next[idx_base - 1] = child_l
        n_next[idx_base] = child_r
    else
        g, h, w = nodes_sum[1, node], nodes_sum[2, node], nodes_sum[3, node]
        if w <= 0.0 || h + lambda <= 0.0
            tree_pred[node] = 0.0f0
        else
            tree_pred[node] = -g / (h + lambda + 1e-8)
        end
    end
end

@kernel function get_gain_gpu!(nodes_gain, nodes_sum, nodes, lambda)
    n_idx = @index(Global)
    node = nodes[n_idx]
    @inbounds p1 = nodes_sum[1, node]
    @inbounds p2 = nodes_sum[2, node]
    @inbounds nodes_gain[node] = p1^2 / (p2 + lambda + 1e-8)
end

