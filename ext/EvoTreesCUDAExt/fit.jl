function EvoTrees.grow_evotree!(evotree::EvoTree{L,K}, cache::CacheGPU, params::EvoTrees.EvoTypes) where {L,K}
    EvoTrees.update_grads!(cache.∇, cache.pred, cache.y, L, params)
    
    for _ in 1:params.bagging_size
        is = EvoTrees.subsample(cache.is_in, cache.is_out, cache.mask, params.rowsample, params.rng)
        
        js_cpu = Vector{eltype(cache.js)}(undef, length(cache.js))
        EvoTrees.sample!(params.rng, cache.js_, js_cpu, replace=false, ordered=true)
        copyto!(cache.js, js_cpu)
        
        tree = EvoTrees.Tree{L,K}(params.max_depth)
        grow_tree!(tree, params, cache, is)
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

    ∇_gpu = cache.∇
    if L <: EvoTrees.MAE
        ∇_gpu = copy(cache.∇)
        ∇_gpu[2, :] .= 1.0f0
    end

    cache.tree_split_gpu .= false
    cache.tree_cond_bin_gpu .= 0
    cache.tree_feat_gpu .= 0
    cache.tree_gain_gpu .= 0
    cache.tree_pred_gpu .= 0
    cache.nodes_sum_gpu .= 0
    cache.anodes_gpu .= 0
    cache.n_next_gpu .= 0
    cache.n_next_active_gpu .= 0
    cache.best_gain_gpu .= 0
    cache.best_bin_gpu .= 0
    cache.best_feat_gpu .= 0
    cache.nidx .= 1
    view(cache.anodes_gpu, 1:1) .= 1

    if params.max_depth == 1
        reduce_root_sums_kernel!(backend)(
            cache.nodes_sum_gpu, ∇_gpu, is; 
            ndrange=length(is), 
            workgroupsize=256
        )
        KernelAbstractions.synchronize(backend)
    else
        update_hist_gpu!(
            L, cache.h∇, cache.best_gain_gpu, cache.best_bin_gpu, cache.best_feat_gpu,
            ∇_gpu, cache.x_bin, cache.nidx, cache.js, is,
            1, view(cache.anodes_gpu, 1:1), cache.nodes_sum_gpu, params,
            cache.feattypes_gpu, cache.monotone_constraints_gpu, cache.K, 
            Float32(params.L2), view(cache.sums_temp_gpu, 1:(2*cache.K+1), 1:1),
            cache.target_mask_buf, backend
        )
    end

    n_active = params.max_depth == 1 ? 0 : 1

    for depth in 1:params.max_depth
        iszero(n_active) && break
        
        view(cache.n_next_active_gpu, 1:1) .= 0
        n_nodes = 2^(depth - 1)
        active_nodes = view(cache.anodes_gpu, 1:n_nodes)
        
        if n_active < n_nodes
            view(cache.anodes_gpu, n_active+1:n_nodes) .= 0
        end

        if depth > 1
            update_hist_gpu!(
                L, cache.h∇, cache.best_gain_gpu, cache.best_bin_gpu, cache.best_feat_gpu,
                ∇_gpu, cache.x_bin, cache.nidx, cache.js, is,
                depth, view(active_nodes, 1:n_active), cache.nodes_sum_gpu, params,
                cache.feattypes_gpu, cache.monotone_constraints_gpu, cache.K, 
                Float32(params.L2), view(cache.sums_temp_gpu, 1:(2*cache.K+1), 1:n_active),
                cache.target_mask_buf, backend
            )
        end

        apply_splits_kernel!(backend)(
            cache.tree_split_gpu, cache.tree_cond_bin_gpu, cache.tree_feat_gpu, 
            cache.tree_gain_gpu, cache.tree_pred_gpu, cache.nodes_sum_gpu,
            cache.n_next_gpu, cache.n_next_active_gpu,
            view(cache.best_gain_gpu, 1:n_nodes), 
            view(cache.best_bin_gpu, 1:n_nodes), 
            view(cache.best_feat_gpu, 1:n_nodes),
            cache.h∇, active_nodes, cache.feattypes_gpu,
            depth, params.max_depth, Float32(params.lambda), Float32(params.gamma), 
            Float32(params.L2), cache.K;
            ndrange = max(n_active, 1), 
            workgroupsize = 256
        )
        KernelAbstractions.synchronize(backend)
        
        n_active = Array(cache.n_next_active_gpu)[1]
        if n_active > 0
            copyto!(view(cache.anodes_gpu, 1:n_active), view(cache.n_next_gpu, 1:n_active))
        end

        if depth < params.max_depth && n_active > 0
            update_nodes_idx_kernel!(backend)(
                cache.nidx, is, cache.x_bin, cache.tree_feat_gpu, 
                cache.tree_cond_bin_gpu, cache.feattypes_gpu;
                ndrange = length(is), 
                workgroupsize = 256
            )
            KernelAbstractions.synchronize(backend)
        end
    end

    copyto!(tree.split, Array(cache.tree_split_gpu))
    copyto!(tree.feat, Array(cache.tree_feat_gpu))
    copyto!(tree.cond_bin, Array(cache.tree_cond_bin_gpu))
    copyto!(tree.gain, Array(cache.tree_gain_gpu))

    leaf_nodes = findall(!, tree.split)

    if L <: Union{EvoTrees.MAE, EvoTrees.Quantile}
        
        cpu_data = (
            nidx = Array(cache.nidx),
            is = Array(is),
            ∇ = Array(cache.∇),
            nodes_sum = Array(cache.nodes_sum_gpu)
        )
        
        leaf_map = Dict{Int, Vector{UInt32}}()
        sizehint!(leaf_map, length(leaf_nodes))
        for i in 1:length(cpu_data.is)
            leaf_id = cpu_data.nidx[cpu_data.is[i]]
            if !haskey(leaf_map, leaf_id)
                leaf_map[leaf_id] = UInt32[]
            end
            push!(leaf_map[leaf_id], cpu_data.is[i])
        end
        
        for n in leaf_nodes
            node_sum_view = view(cpu_data.nodes_sum, :, n)
            if L <: EvoTrees.Quantile
                node_is = get(leaf_map, n, UInt32[])
                if !isempty(node_is)
                    EvoTrees.pred_leaf_cpu!(tree.pred, n, node_sum_view, L, params, cpu_data.∇, node_is)
                else
                    EvoTrees.pred_leaf_cpu!(tree.pred, n, node_sum_view, EvoTrees.MAE, params)
                end
            else
                EvoTrees.pred_leaf_cpu!(tree.pred, n, node_sum_view, L, params)
            end
        end
    else
        nodes_sum_cpu = Array(cache.nodes_sum_gpu)
        for n in leaf_nodes
            node_sum_view = view(nodes_sum_cpu, :, n)
            EvoTrees.pred_leaf_cpu!(tree.pred, n, node_sum_view, L, params)
        end
    end
    
    return nothing
end

@kernel function apply_splits_kernel!(
    tree_split, tree_cond_bin, tree_feat, tree_gain, tree_pred,
    nodes_sum, n_next, n_next_active,
    best_gain, best_bin, best_feat, h∇, active_nodes, feattypes,
    depth, max_depth, lambda, gamma, L2, K_val
)
    n_idx = @index(Global)
    node = active_nodes[n_idx]
    eps = eltype(tree_pred)(1e-8)

    @inbounds if depth < max_depth && best_gain[n_idx] > gamma
        
        tree_split[node] = true
        tree_cond_bin[node] = best_bin[n_idx]
        tree_feat[node] = best_feat[n_idx]
        tree_gain[node] = best_gain[n_idx]

        child_l = node << 1
        child_r = (node << 1) + 1
        feat = Int(tree_feat[node])
        bin = Int(tree_cond_bin[node])
        is_numeric = feattypes[feat]

        for kk in 1:(2*K_val+1)
            sum_val = zero(eltype(nodes_sum))
            if is_numeric
                for b in 1:bin
                    sum_val += h∇[kk, b, feat, node]
                end
            else
                sum_val = h∇[kk, bin, feat, node]
            end
            nodes_sum[kk, child_l] = sum_val
            nodes_sum[kk, child_r] = nodes_sum[kk, node] - sum_val
        end
        
        w_l = nodes_sum[2*K_val+1, child_l]
        w_r = nodes_sum[2*K_val+1, child_r]
        
        if K_val == 1
            g_l = nodes_sum[1, child_l]
            h_l = nodes_sum[2, child_l]
            tree_pred[1, child_l] = -g_l / max(eps, h_l + lambda * w_l + L2)
            
            g_r = nodes_sum[1, child_r]
            h_r = nodes_sum[2, child_r]
            tree_pred[1, child_r] = -g_r / max(eps, h_r + lambda * w_r + L2)
        else
            for k in 1:K_val
                g_l = nodes_sum[k, child_l]
                h_l = nodes_sum[K_val+k, child_l]
                tree_pred[k, child_l] = -g_l / max(eps, h_l + lambda * w_l + L2)
                
                g_r = nodes_sum[k, child_r]
                h_r = nodes_sum[K_val+k, child_r]
                tree_pred[k, child_r] = -g_r / max(eps, h_r + lambda * w_r + L2)
            end
        end
        
        idx_base = Atomix.@atomic n_next_active[1] += 2
        n_next[idx_base - 1] = child_l
        n_next[idx_base] = child_r
    else
        
        w = nodes_sum[2*K_val+1, node]
        if K_val == 1
            g = nodes_sum[1, node]
            h = nodes_sum[2, node]
            tree_pred[1, node] = (w <= 0 || h <= 0) ? 0.0f0 : -g / max(eps, h + lambda * w + L2)
        else
            for k in 1:K_val
                g = nodes_sum[k, node]
                h = nodes_sum[K_val+k, node]
                tree_pred[k, node] = (w <= 0 || h <= 0) ? 0.0f0 : -g / max(eps, h + lambda * w + L2)
            end
        end
    end
end

