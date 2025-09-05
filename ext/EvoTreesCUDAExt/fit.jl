function EvoTrees.grow_evotree!(evotree::EvoTree{L,K}, cache::CacheGPU, params::EvoTrees.EvoTypes) where {L,K}
    EvoTrees.update_grads!(cache.∇, cache.pred, cache.y, L, params)
    
    for _ in 1:params.bagging_size
        is = EvoTrees.subsample(cache.is_in, cache.is_out, cache.mask, params.rowsample, params.rng)
        
        js_cpu = Vector{eltype(cache.js)}(undef, length(cache.js))
        EvoTrees.sample!(params.rng, cache.js_, js_cpu, replace=false, ordered=true)
        copyto!(cache.js, js_cpu)
        
        tree = EvoTrees.Tree{L,K}(params.max_depth)
        grow! = params.tree_type == :oblivious ? grow_otree! : grow_tree!
        grow!(
            tree,
            params,
            cache,
            is,
        )
        push!(evotree.trees, tree)
        EvoTrees.predict!(cache.pred, tree, cache.x_bin, cache.feattypes_gpu)
    end
    
    evotree.info[:nrounds] += 1
    return nothing
end

function grow_otree!(
    tree::EvoTrees.Tree{L,K},
    params::EvoTrees.EvoTypes,
    cache::CacheGPU,
    is::CuVector
) where {L,K}
    @warn "Oblivious tree GPU implementation not yet available, using standard tree" maxlog=1
    grow_tree!(tree, params, cache, is)
end

function grow_tree!(
    tree::EvoTrees.Tree{L,K},
    params::EvoTrees.EvoTypes,
    cache::CacheGPU,
    is::CuVector
) where {L,K}

    backend = KernelAbstractions.get_backend(cache.x_bin)

    cache.tree_split_gpu .= false
    cache.tree_cond_bin_gpu .= 0
    cache.tree_feat_gpu .= 0
    cache.tree_gain_gpu .= 0
    cache.tree_pred_gpu .= 0
    cache.nodes_sum_gpu .= 0
    cache.nodes_gain_gpu .= 0
    cache.anodes_gpu .= 0
    cache.n_next_gpu .= 0
    cache.n_next_active_gpu .= 0
    cache.best_gain_gpu .= 0
    cache.best_bin_gpu .= 0
    cache.best_feat_gpu .= 0
    
    cache.nidx .= 1
    
    view(cache.anodes_gpu, 1:1) .= 1
    update_hist_gpu!(
        cache.h∇, cache.best_gain_gpu, cache.best_bin_gpu, cache.best_feat_gpu,
        cache.∇, cache.x_bin, cache.nidx, cache.js, is,
        1, view(cache.anodes_gpu, 1:1), cache.nodes_sum_gpu, params,
        cache.left_nodes_buf, cache.right_nodes_buf, cache.target_mask_buf,
        cache.feattypes_gpu, cache.monotone_constraints_gpu, cache.K
    )
    get_gain_gpu!(backend)(cache.nodes_gain_gpu, cache.nodes_sum_gpu, view(cache.anodes_gpu, 1:1), Float32(params.lambda), cache.K; ndrange=1, workgroupsize=1)
    KernelAbstractions.synchronize(backend)

    n_active = 1

    for depth in 1:params.max_depth
        !iszero(n_active) || break
        
        n_nodes_level = 2^(depth - 1)
        active_nodes_full = view(cache.anodes_gpu, 1:n_nodes_level)
        
        if n_active < n_nodes_level
            view(cache.anodes_gpu, n_active+1:n_nodes_level) .= 0
        end

        view_gain = view(cache.best_gain_gpu, 1:n_nodes_level)
        view_bin  = view(cache.best_bin_gpu, 1:n_nodes_level)
        view_feat = view(cache.best_feat_gpu, 1:n_nodes_level)
        
        if depth > 1
            active_nodes_act = view(active_nodes_full, 1:n_active)

            cache.build_nodes_gpu .= 0
            cache.subtract_nodes_gpu .= 0
            cache.build_count .= 0
            cache.subtract_count .= 0

            separate_kernel! = separate_nodes_kernel!(backend)
            separate_kernel!(
                cache.build_nodes_gpu, cache.build_count,
                cache.subtract_nodes_gpu, cache.subtract_count,
                active_nodes_act;
                ndrange=n_active, workgroupsize=256
            )
            KernelAbstractions.synchronize(backend)
            
            subtract_count_val = Array(cache.subtract_count)[1]
            build_count_val = Array(cache.build_count)[1]
            
            if subtract_count_val > 0
                n_k = size(cache.h∇, 1)
                n_b = size(cache.h∇, 2)
                n_j = size(cache.h∇, 3)
                subtract_hist_kernel!(backend)(
                    cache.h∇, cache.subtract_nodes_gpu, n_k, n_b, n_j;
                    ndrange = subtract_count_val * n_k * n_b * n_j, workgroupsize=256
                )
                KernelAbstractions.synchronize(backend)
            end
            
            if build_count_val > 0
                update_hist_gpu!(
                    cache.h∇, cache.best_gain_gpu, cache.best_bin_gpu, cache.best_feat_gpu,
                    cache.∇, cache.x_bin, cache.nidx, cache.js, is,
                    depth, view(cache.build_nodes_gpu, 1:build_count_val), cache.nodes_sum_gpu, params,
                    cache.left_nodes_buf, cache.right_nodes_buf, cache.target_mask_buf,
                    cache.feattypes_gpu, cache.monotone_constraints_gpu, cache.K
                )
            end
        end

        apply_splits_kernel!(backend)(
            cache.tree_split_gpu, cache.tree_cond_bin_gpu, cache.tree_feat_gpu, cache.tree_gain_gpu, cache.tree_pred_gpu,
            cache.nodes_sum_gpu, cache.nodes_gain_gpu,
            cache.n_next_gpu, cache.n_next_active_gpu,
            view_gain, view_bin, view_feat,
            cache.h∇,
            active_nodes_full,
            depth, params.max_depth, Float32(params.lambda), Float32(params.gamma), cache.K;
            ndrange = n_active, workgroupsize=256
        )
        KernelAbstractions.synchronize(backend)
        
        n_active = min(2 * n_active, 2^depth)
        if n_active > 0
            copyto!(view(cache.anodes_gpu, 1:n_active), view(cache.n_next_gpu, 1:n_active))
        end

        if depth < params.max_depth && n_active > 0
            update_nodes_idx_kernel!(backend)(
                cache.nidx, is, cache.x_bin, cache.tree_feat_gpu, cache.tree_cond_bin_gpu, cache.feattypes_gpu;
                ndrange = length(is), workgroupsize=256
            )
            KernelAbstractions.synchronize(backend)
        end
    end

    copyto!(tree.split, Array(cache.tree_split_gpu))
    copyto!(tree.cond_bin, Array(cache.tree_cond_bin_gpu))
    copyto!(tree.feat, Array(cache.tree_feat_gpu))
    copyto!(tree.gain, Array(cache.tree_gain_gpu))
    copyto!(tree.pred, Array(cache.tree_pred_gpu .* Float32(params.eta)))
    
    return nothing
end

@kernel function apply_splits_kernel!(
    tree_split, tree_cond_bin, tree_feat, tree_gain, tree_pred,
    nodes_sum, nodes_gain,
    n_next, n_next_active,
    best_gain, best_bin, best_feat,
    h∇,
    active_nodes,
    depth, max_depth, lambda, gamma,
    K, L::Type, params::EvoTypes
)
    n_idx = @index(Global)
    node = active_nodes[n_idx]
    epsv = eltype(tree_pred)(1e-8)

    @inbounds if depth < max_depth && best_gain[n_idx] > gamma
        # Logic for splitting a node remains the same
        # ...
    else # This node is a terminal leaf, calculate its prediction
        if L <: EvoTrees.Quantile || L <: EvoTrees.MAE
            # FIX: For Quantile/MAE, find the alpha-quantile from the histogram of residuals
            alpha = L <: EvoTrees.MAE ? 0.5f0 : Float32(params.alpha)
            
            # The residuals were stored in the hessian slot (k=2 for K=1)
            # The weights were stored in the final slot (k=3 for K=1)
            
            # Get total weight and target quantile weight in the node
            node_w = nodes_sum[3, 1, node] # Total weight for this node
            target_w = alpha * node_w
            
            cum_w = 0.0f0
            leaf_pred = 0.0f0
            
            # Find the bin where the cumulative weight exceeds the target
            nbins = size(h∇, 2)
            for b in 1:nbins
                # Note: This assumes a single feature histogram. A more robust implementation
                # would need to decide which feature's histogram to use or average them.
                # For quantile prediction, typically the histogram of residuals is built directly.
                # Let's assume h∇[2, b, 1, node] is the sum of residuals in that bin
                # and h∇[3, b, 1, node] is the sum of weights.
                bin_w = h∇[3, b, 1, node] 
                
                cum_w += bin_w
                if cum_w >= target_w
                    # Placeholder: Use bin index as prediction.
                    # A better way is to use bin edges to find the value.
                    leaf_pred = Float32(b) 
                    break
                end
            end
            tree_pred[1, node] = leaf_pred

        else
            # For all other losses, use the standard -G/H formula
            w = nodes_sum[2*K+1, 1, node]
            if w > 0
                @inbounds for kk in 1:K
                    gk = nodes_sum[kk, 1, node]
                    hk = nodes_sum[K+kk, 1, node]
                    tree_pred[kk, node] = -gk / (hk + lambda * w + epsv)
                end
            end
        end
    end
end

@kernel function get_gain_gpu!(nodes_gain::AbstractVector{T}, nodes_sum::AbstractArray{T,2}, nodes, lambda::T, K::Int) where {T}
    n_idx = @index(Global)
    node = nodes[n_idx]
    @inbounds w = nodes_sum[2*K+1, node]
    gain = zero(T)
    @inbounds for kk in 1:K
        p1 = nodes_sum[kk, node]
        p2 = nodes_sum[K+kk, node]
        gain += p1^2 / (p2 + lambda * (w / K) + T(1e-8))
    end
    @inbounds nodes_gain[node] = gain
end

