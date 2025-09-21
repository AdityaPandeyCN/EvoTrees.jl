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

    # Add CPU fallback for unsupported loss functions
    if L <: Union{EvoTrees.Quantile, EvoTrees.Cred}
        error("Loss function $(L) requires CPU implementation. Set device=:cpu in model config.")
    end
    
    # Warn if multi-output with potential issues (except for known working ones)
    if K > 1 && !(L <: Union{EvoTrees.GaussianMLE, EvoTrees.MLogLoss})
        @warn "Multi-output (K=$K) GPU implementation may have numerical issues" maxlog=1
    end

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
            
            build_count_val = Array(cache.build_count)[1]
            subtract_count_val = Array(cache.subtract_count)[1]
            
            if build_count_val > 0
            update_hist_gpu!(
                cache.h∇, cache.best_gain_gpu, cache.best_bin_gpu, cache.best_feat_gpu,
                cache.∇, cache.x_bin, cache.nidx, cache.js, is,
                    depth, view(cache.build_nodes_gpu, 1:build_count_val), cache.nodes_sum_gpu, params,
                cache.feattypes_gpu, cache.monotone_constraints_gpu, cache.K
            )
            end
            
            if subtract_count_val > 0
                subtract_hist_kernel!(backend)(
                    cache.h∇, view(cache.subtract_nodes_gpu, 1:subtract_count_val);
                    ndrange = subtract_count_val * size(cache.h∇, 1) * size(cache.h∇, 2) * size(cache.h∇, 3),
                    workgroupsize=256
                )
                KernelAbstractions.synchronize(backend)
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
    copyto!(tree.pred, Array(cache.tree_pred_gpu .* Float32(params.eta / params.bagging_size)))
    
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
    K
)
    n_idx = @index(Global)
    node = active_nodes[n_idx]

    epsv = eltype(tree_pred)(1e-8)

    @inbounds if depth < max_depth && best_gain[n_idx] > gamma
        tree_split[node] = true
        tree_cond_bin[node] = best_bin[n_idx]
        tree_feat[node] = best_feat[n_idx]
        tree_gain[node] = best_gain[n_idx]

        child_l, child_r = node << 1, (node << 1) + 1
        feat, bin = Int(tree_feat[node]), Int(tree_cond_bin[node])

        # FIX: Transfer ALL dimensions to children nodes
        @inbounds for kk in 1:(2*K+1)
            sum_val = zero(eltype(nodes_sum))
            for b in 1:bin
                sum_val += h∇[kk, b, feat, node]
            end
            nodes_sum[kk, child_l] = sum_val
            nodes_sum[kk, child_r] = nodes_sum[kk, node] - sum_val
        end
        
        # FIX: Calculate gain correctly for both K=1 and K>1
        w_l = nodes_sum[2*K+1, child_l]
        w_r = nodes_sum[2*K+1, child_r]
        
        if K == 1
            # Single output - original logic
            g_l = nodes_sum[1, child_l]
            h_l = nodes_sum[2, child_l]
            nodes_gain[child_l] = g_l^2 / (h_l + lambda * w_l + epsv)
            
            g_r = nodes_sum[1, child_r]
            h_r = nodes_sum[2, child_r]
            nodes_gain[child_r] = g_r^2 / (h_r + lambda * w_r + epsv)
            
            tree_pred[1, child_l] = -g_l / (h_l + lambda * w_l + epsv)
            tree_pred[1, child_r] = -g_r / (h_r + lambda * w_r + epsv)
        else
            # Multi-output: sum gains across all K dimensions
            gain_l = zero(eltype(nodes_gain))
            gain_r = zero(eltype(nodes_gain))
            
            @inbounds for k in 1:K
                # Left child
                g_l = nodes_sum[k, child_l]
                h_l = nodes_sum[K+k, child_l]
                gain_l += g_l^2 / (h_l + lambda * w_l / K + epsv)
                tree_pred[k, child_l] = -g_l / (h_l + lambda * w_l / K + epsv)
                
                # Right child
                g_r = nodes_sum[k, child_r]
                h_r = nodes_sum[K+k, child_r]
                gain_r += g_r^2 / (h_r + lambda * w_r / K + epsv)
                tree_pred[k, child_r] = -g_r / (h_r + lambda * w_r / K + epsv)
            end
            
            nodes_gain[child_l] = gain_l
            nodes_gain[child_r] = gain_r
        end
        
        # Update next active nodes
        idx_base = Atomix.@atomic n_next_active[1] += 2
        n_next[idx_base - 1] = child_l
        n_next[idx_base] = child_r

    else
        # Leaf node - calculate final predictions
        if K == 1
            g = nodes_sum[1, node]
            h = nodes_sum[2, node]
            w = nodes_sum[2*K+1, node]
            if w <= zero(w) || h + lambda * w <= zero(h)
                tree_pred[1, node] = 0.0f0
            else
                tree_pred[1, node] = -g / (h + lambda * w + epsv)
            end
        else
            # Multi-output leaf predictions
            w = nodes_sum[2*K+1, node]
            @inbounds for k in 1:K
                g = nodes_sum[k, node]
                h = nodes_sum[K+k, node]
                if w <= zero(w) || h + lambda * w / K <= zero(h)
                    tree_pred[k, node] = 0.0f0
                else
                    tree_pred[k, node] = -g / (h + lambda * w / K + epsv)
                end
            end
        end
    end
end

@kernel function get_gain_gpu!(
    nodes_gain::AbstractVector{T}, 
    nodes_sum::AbstractArray{T,2}, 
    nodes, 
    lambda::T, 
    K::Int
) where {T}
    n_idx = @index(Global)
    node = nodes[n_idx]
    
    @inbounds if node > 0
        eps = T(1e-8)
        
        if K == 1
            # Single output
            g = nodes_sum[1, node]
            h = nodes_sum[2, node]
            w = nodes_sum[2*K+1, node]
            nodes_gain[node] = g^2 / (h + lambda * w + eps)
        else
            # FIX: Multi-output - sum gains across all K dimensions
            gain_sum = zero(T)
            w = nodes_sum[2*K+1, node]
            
            @inbounds for k in 1:K
                g = nodes_sum[k, node]
                h = nodes_sum[K+k, node]
                gain_sum += g^2 / (h + lambda * w / K + eps)
            end
            
            nodes_gain[node] = gain_sum
        end
    end
end

