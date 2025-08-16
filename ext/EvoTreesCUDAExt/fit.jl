function EvoTrees.grow_evotree!(evotree::EvoTree{L,K}, cache, params::EvoTrees.EvoTypes{L}, ::Type{<:EvoTrees.GPU}) where {L,K}
    EvoTrees.update_grads!(cache.∇, cache.pred, cache.y, params)
    is = EvoTrees.subsample(cache.is_in, cache.is_out, cache.mask, params.rowsample, params.rng)
    EvoTrees.sample!(params.rng, cache.js_, cache.js, replace=false, ordered=true)

    tree = EvoTrees.Tree{L,K}(params.max_depth)
    grow! = params.tree_type == "oblivious" ? grow_otree! : grow_tree!
    
    grow!(tree, params, cache, cache.edges, is, cache.js)
    
    push!(evotree.trees, tree)
    EvoTrees.predict!(cache.pred, tree, cache.x_bin, cache.feattypes_gpu)
    cache.nrounds[] += 1  # Note: using Ref
    return nothing
end

@kernel function sum_grads_kernel!(nodes_sum::AbstractArray{T}, @Const(∇), @Const(is)) where T
    tid_local = @index(Local, Linear)
    tid_global = @index(Global, Linear)
    total_threads = @ndrange()[1]

    shmem_g1 = @localmem T (256,)
    shmem_g2 = @localmem T (256,)

    g1 = zero(T)
    g2 = zero(T)

    # Grid-stride loop
    for i_obs in tid_global:total_threads:length(is)
        @inbounds if i_obs <= length(is)
            idx = is[i_obs]
            g1 += ∇[1, idx]
            g2 += ∇[2, idx]
        end
    end

    shmem_g1[tid_local] = g1
    shmem_g2[tid_local] = g2
    @synchronize()

    # Reduction
    stride = 128
    while stride >= 1
        @synchronize()
        if tid_local <= stride && (tid_local + stride) <= 256
            shmem_g1[tid_local] += shmem_g1[tid_local + stride]
            shmem_g2[tid_local] += shmem_g2[tid_local + stride]
        end
        stride = stride >> 1
    end
    @synchronize()

    if tid_local == 1
        Atomix.@atomic nodes_sum[1, 1] += shmem_g1[1]
        Atomix.@atomic nodes_sum[2, 1] += shmem_g2[1]
    end
end

@kernel function apply_splits_kernel!(
    tree_split, tree_cond_bin, tree_feat, tree_gain, tree_pred,
    nodes_sum, nodes_gain, h∇,
    n_next, n_next_active, 
    best_gain, best_bin, best_feat, 
    active_nodes, 
    depth::Int32, max_depth::Int32, lambda::T, gamma::T
) where T
    n_idx = @index(Global)
    @inbounds if n_idx <= length(active_nodes)
        node = active_nodes[n_idx]
        
        if depth < max_depth && best_gain[n_idx] > nodes_gain[node] + gamma && best_bin[n_idx] > Int32(0)
            tree_split[node] = true
            tree_cond_bin[node] = UInt32(best_bin[n_idx])
            tree_feat[node] = best_feat[n_idx]
            tree_gain[node] = best_gain[n_idx]

            child_l = node << 1
            child_r = (node << 1) + Int32(1)
            
            feat = Int(tree_feat[node])
            bin = Int(tree_cond_bin[node])
            
            # Calculate children statistics
            l_g1 = zero(T)
            l_g2 = zero(T)
            l_w = zero(T)
            
            for b in 1:bin
                l_g1 += h∇[1, b, feat, node]
                l_g2 += h∇[2, b, feat, node]
                l_w += h∇[3, b, feat, node]
            end
            
            p_g1 = nodes_sum[1, node]
            p_g2 = nodes_sum[2, node]
            p_w = nodes_sum[3, node]
            
            r_g1 = p_g1 - l_g1
            r_g2 = p_g2 - l_g2
            r_w = p_w - l_w
            
            nodes_sum[1, child_l] = l_g1
            nodes_sum[2, child_l] = l_g2
            nodes_sum[3, child_l] = l_w
            
            nodes_sum[1, child_r] = r_g1
            nodes_sum[2, child_r] = r_g2
            nodes_sum[3, child_r] = r_w
            
            nodes_gain[child_l] = l_g1^2 / (l_g2 + lambda + T(1e-8))
            nodes_gain[child_r] = r_g1^2 / (r_g2 + lambda + T(1e-8))
            
            idx_base = Atomix.@atomic n_next_active[1] += Int32(2)
            if idx_base <= length(n_next)
                n_next[idx_base - Int32(1)] = child_l
                n_next[idx_base] = child_r
            end
        else
            g = nodes_sum[1, node]
            h = nodes_sum[2, node]
            tree_pred[node] = -g / (h + lambda + T(1e-8))
        end
    end
end

@kernel function get_gain_kernel!(nodes_gain::AbstractArray{T}, @Const(nodes_sum), @Const(nodes), lambda::T) where T
    n_idx = @index(Global)
    @inbounds if n_idx <= length(nodes)
        node = nodes[n_idx]
        p1 = nodes_sum[1, node]
        p2 = nodes_sum[2, node]
        nodes_gain[node] = p1^2 / (p2 + lambda + T(1e-8))
    end
end

function grow_tree!(tree::EvoTrees.Tree{L,K}, params::EvoTrees.EvoTypes{L}, cache, edges, is, js) where {L,K}

    backend = KernelAbstractions.get_backend(cache.x_bin)
    js_gpu = CuArray(UInt32.(js))
    is_gpu = CuArray(UInt32.(is))
    
    T = eltype(cache.nodes_sum_gpu)  # Float32

    # Clear arrays for new tree
    cache.tree_split_gpu .= false
    cache.tree_cond_bin_gpu .= UInt32(0)
    cache.tree_feat_gpu .= Int32(0)
    cache.tree_gain_gpu .= T(0)
    cache.tree_pred_gpu .= T(0)
    
    # Initialize root node
    cache.nodes_sum_gpu[:, 1] .= T(0)
    
    kernel_sum_grads! = sum_grads_kernel!(backend, 256)
    kernel_sum_grads!(cache.nodes_sum_gpu, cache.∇, is_gpu; ndrange=min(2048, length(is_gpu)))
    KernelAbstractions.synchronize(backend)
    
    nsamples = T(length(is_gpu))
    CUDA.copyto!(view(cache.nodes_sum_gpu, 3, 1), CuArray([nsamples]))

    kernel_get_gain! = get_gain_kernel!(backend)
    kernel_get_gain!(
        cache.nodes_gain_gpu, cache.nodes_sum_gpu, 
        CuArray(Int32[1]), cache.gpu_params.lambda; 
        ndrange=1
    )
    KernelAbstractions.synchronize(backend)
    
    CUDA.copyto!(view(cache.anodes_gpu, 1:1), CuArray(Int32[1]))
    
    cache.nidx .= UInt32(1)
    n_active = 1

    for depth in 1:params.max_depth
        !iszero(n_active) || break
        
        active_nodes = view(cache.anodes_gpu, 1:n_active)
        view_gain = view(cache.best_gain_gpu, 1:n_active)
        view_bin = view(cache.best_bin_gpu, 1:n_active)
        view_feat = view(cache.best_feat_gpu, 1:n_active)
        
        # Update histograms and find best splits
        update_hist_gpu!(
            cache.h∇, view_gain, view_bin, view_feat,
            cache.∇, cache.x_bin, cache.nidx, js_gpu,
            depth, active_nodes, cache.nodes_sum_gpu, cache.gpu_params,
            cache.gains_feats, cache.bins_feats
        )

        cache.n_next_active_gpu .= Int32(0)
        
        kernel_apply_splits! = apply_splits_kernel!(backend)
        kernel_apply_splits!(
            cache.tree_split_gpu, cache.tree_cond_bin_gpu, cache.tree_feat_gpu, 
            cache.tree_gain_gpu, cache.tree_pred_gpu,
            cache.nodes_sum_gpu, cache.nodes_gain_gpu, cache.h∇,
            cache.n_next_gpu, cache.n_next_active_gpu,
            view_gain, view_bin, view_feat, active_nodes,
            Int32(depth), cache.gpu_params.max_depth, 
            cache.gpu_params.lambda, cache.gpu_params.gamma;
            ndrange = n_active
        )
        KernelAbstractions.synchronize(backend)
        
        n_active = Int(Array(cache.n_next_active_gpu)[1])
        if n_active > 0 && n_active <= length(cache.anodes_gpu)
            copyto!(view(cache.anodes_gpu, 1:n_active), view(cache.n_next_gpu, 1:n_active))
        end

        # Update node indices for next iteration
        if depth < params.max_depth && n_active > 0
            kernel_update_nodes! = update_nodes_idx_kernel!(backend)
            kernel_update_nodes!(
                cache.nidx, is_gpu, cache.x_bin, 
                cache.tree_feat_gpu, cache.tree_cond_bin_gpu, cache.feattypes_gpu;
                ndrange = length(is_gpu)
            )
            KernelAbstractions.synchronize(backend)
        end
    end

    # Copy results back to CPU
    copyto!(tree.split, Array(cache.tree_split_gpu))
    copyto!(tree.cond_bin, Array(cache.tree_cond_bin_gpu))
    copyto!(tree.feat, Array(cache.tree_feat_gpu))
    copyto!(tree.gain, Array(cache.tree_gain_gpu))
    copyto!(tree.pred, Array(cache.tree_pred_gpu))
    
    # Set condition floats based on edges
    for i in eachindex(tree.split)
        if tree.split[i]
            tree.cond_float[i] = edges[tree.feat[i]][tree.cond_bin[i]]
        end
    end
    return nothing
end

function grow_otree!(tree::EvoTrees.Tree{L,K}, params::EvoTrees.EvoTypes{L}, cache, edges, is, js) where {L,K}
    @warn "GPU implementation for oblivious trees is not yet optimized."
    return nothing
end

