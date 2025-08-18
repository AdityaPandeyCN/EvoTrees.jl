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
        cache.left_nodes_buf,
        cache.right_nodes_buf,
        cache.target_mask_buf,
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
    left_nodes_buf::CuArray{Int32},
    right_nodes_buf::CuArray{Int32},
    target_mask_buf::CuArray{UInt8},
) where {L,K}

    backend = KernelAbstractions.get_backend(x_bin)
    js_gpu = KernelAbstractions.adapt(backend, js)
    is_gpu = KernelAbstractions.adapt(backend, is)

    tree_split_gpu = KernelAbstractions.zeros(backend, Bool, length(tree.split))
    tree_cond_bin_gpu = KernelAbstractions.zeros(backend, UInt8, length(tree.cond_bin))
    tree_feat_gpu = KernelAbstractions.zeros(backend, Int32, length(tree.feat))
    tree_gain_gpu = KernelAbstractions.zeros(backend, Float64, length(tree.gain))
    tree_pred_gpu = KernelAbstractions.zeros(backend, Float32, size(tree.pred, 1), size(tree.pred, 2))

    max_nodes_total = 2^(params.max_depth + 1)
    nodes_sum_gpu = KernelAbstractions.zeros(backend, Float32, 3, max_nodes_total)

    max_nodes_level = 2^params.max_depth
    anodes_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    n_next_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level * 2)
    n_next_active_gpu = KernelAbstractions.zeros(backend, Int32, 1)

    best_gain_gpu = KernelAbstractions.zeros(backend, Float32, max_nodes_level)
    best_bin_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    best_feat_gpu = KernelAbstractions.zeros(backend, Int32, max_nodes_level)
    
    nidx .= 1
    
    # Initialize root node
    copyto!(view(anodes_gpu, 1:1), Int32[1])
    n_active = 1
    
    # Compute root histograms and sums
    update_hist_gpu!(
        h∇, h∇L, h∇R,
        best_gain_gpu, best_bin_gpu, best_feat_gpu,
        ∇, x_bin, nidx, js_gpu, is_gpu,
        1, view(anodes_gpu, 1:1), nodes_sum_gpu, params,
        left_nodes_buf, right_nodes_buf, target_mask_buf
    )
    
    # Root gain not needed explicitly; proceed to depth loop

    for depth in 1:params.max_depth
        !iszero(n_active) || break
        
        n_nodes_level = 2^(depth - 1)
        active_nodes = view(anodes_gpu, 1:n_active)
        
        view_gain = view(best_gain_gpu, 1:n_active)
        view_bin  = view(best_bin_gpu, 1:n_active)
        view_feat = view(best_feat_gpu, 1:n_active)
        
        # For depth > 1, update histograms with subtraction trick
        if depth > 1
            update_hist_gpu!(
                h∇, h∇L, h∇R,
                view_gain, view_bin, view_feat,
                ∇, x_bin, nidx, js_gpu, is_gpu,
                depth, active_nodes, nodes_sum_gpu, params,
                left_nodes_buf, right_nodes_buf, target_mask_buf
            )
        end

        # Reset next active counter
        n_next_active_gpu .= 0
        
        # Apply splits for active nodes
        apply_splits_kernel!(backend)(
            tree_split_gpu, tree_cond_bin_gpu, tree_feat_gpu, tree_gain_gpu, tree_pred_gpu,
            nodes_sum_gpu,
            n_next_gpu, n_next_active_gpu,
            view_gain, view_bin, view_feat,
            h∇L,
            active_nodes,
            depth, params.max_depth, Float32(params.lambda), Float32(params.gamma);
            ndrange = n_active
        )
        KernelAbstractions.synchronize(backend)
        
        # Get number of active nodes for next iteration
        n_active = Int(Array(n_next_active_gpu)[1])
        
        # Copy next active nodes
        if n_active > 0
            copyto!(view(anodes_gpu, 1:n_active), view(n_next_gpu, 1:n_active))
        end

        # Update node indices for next depth
        if depth < params.max_depth && n_active > 0
            update_nodes_idx_kernel!(backend)(
                nidx, is_gpu, x_bin, tree_feat_gpu, tree_cond_bin_gpu, feattypes_gpu;
                ndrange = length(is_gpu)
            )
            KernelAbstractions.synchronize(backend)
        end
    end

    # Copy results back to CPU
    copyto!(tree.split, Array(tree_split_gpu))
    copyto!(tree.cond_bin, Array(tree_cond_bin_gpu))
    copyto!(tree.feat, Array(tree_feat_gpu))
    copyto!(tree.gain, Array(tree_gain_gpu))
    copyto!(tree.pred, Array(tree_pred_gpu .* Float32(params.eta)))
    
    # Set float conditions from bins
    for i in eachindex(tree.split)
        if tree.split[i]
            tree.cond_float[i] = edges[tree.feat[i]][tree.cond_bin[i]]
        end
    end

    return nothing
end

@kernel function apply_splits_kernel!(
    tree_split, tree_cond_bin, tree_feat, tree_gain, tree_pred,
    nodes_sum,
    n_next, n_next_active,
    best_gain, best_bin, best_feat,
    h∇L,
    active_nodes,
    depth, max_depth, lambda, gamma
)
    n_idx = @index(Global)
    
    @inbounds if n_idx <= length(active_nodes)
        node = active_nodes[n_idx]
        
        if node > 0 && depth < max_depth && best_gain[n_idx] > gamma
            # This node will split
            tree_split[node] = true
            tree_cond_bin[node] = best_bin[n_idx]
            tree_feat[node] = best_feat[n_idx]
            tree_gain[node] = best_gain[n_idx]

            child_l = node << 1
            child_r = child_l + 1
            feat = Int(tree_feat[node])
            bin = Int(tree_cond_bin[node])

            # Set children sums from histogram
            nodes_sum[1, child_l] = h∇L[1, bin, feat, node]
            nodes_sum[2, child_l] = h∇L[2, bin, feat, node]
            nodes_sum[3, child_l] = h∇L[3, bin, feat, node]
            
            nodes_sum[1, child_r] = nodes_sum[1, node] - nodes_sum[1, child_l]
            nodes_sum[2, child_r] = nodes_sum[2, node] - nodes_sum[2, child_l]
            nodes_sum[3, child_r] = nodes_sum[3, node] - nodes_sum[3, child_l]

            # Compute local values for children (avoid writing back gains array)
            p1_l = nodes_sum[1, child_l]
            p2_l = nodes_sum[2, child_l]
            w_l = nodes_sum[3, child_l]
            epsv = eltype(tree_pred)(1e-8)
            
            p1_r = nodes_sum[1, child_r]
            p2_r = nodes_sum[2, child_r]
            w_r = nodes_sum[3, child_r]
            
            # Add children to next active nodes
            idx_base = Atomix.@atomic n_next_active[1] += 2
            n_next[idx_base - 1] = child_l
            n_next[idx_base] = child_r

            # Compute predictions for children
            tree_pred[1, child_l] = -p1_l / (p2_l + lambda * w_l + epsv)
            tree_pred[1, child_r] = -p1_r / (p2_r + lambda * w_r + epsv)
        else
            # This is a leaf node
            g = nodes_sum[1, node]
            h = nodes_sum[2, node]
            w = nodes_sum[3, node]
            if w > 0.0 && h + lambda * w > 0.0
                tree_pred[1, node] = -g / (h + lambda * w + epsv)
            else
                tree_pred[1, node] = 0.0f0
            end
        end
    end
end

