using KernelAbstractions
using Atomix

@kernel function update_nodes_idx_kernel!(
    nidx::AbstractVector{UInt32}, 
    @Const(is), 
    @Const(x_bin), 
    @Const(cond_feats), 
    @Const(cond_bins), 
    @Const(feattypes)
)
    gidx = @index(Global, Linear)
    
    @inbounds if gidx <= length(is)
        obs = is[gidx]
        node = nidx[obs]
        
        if node > zero(UInt32)
            feat = cond_feats[node]
            bin = cond_bins[node]
            
            if bin == zero(UInt32)
                nidx[obs] = zero(UInt32)
            else
                feattype = feattypes[feat]
                is_left = feattype ? (x_bin[obs, feat] <= bin) : (x_bin[obs, feat] == bin)
                nidx[obs] = (node << 1) + UInt32(!is_left)
            end
        end
    end
end

@kernel function hist_kernel!(
    h∇::AbstractArray{T,4}, 
    @Const(∇), 
    @Const(x_bin), 
    @Const(nidx), 
    @Const(js)
) where {T}
    i, j = @index(Global, NTuple)
    
    @inbounds if i <= size(x_bin, 1) && j <= length(js)
        node_u = nidx[i]
        
        if node_u > zero(UInt32)
            node = Int(node_u)
            jdx = Int(js[j])
            bin = Int(x_bin[i, jdx])
            
            if bin > 0 && bin <= size(h∇, 2) && node <= size(h∇, 4)
                Atomix.@atomic h∇[1, bin, jdx, node] += ∇[1, i]
                Atomix.@atomic h∇[2, bin, jdx, node] += ∇[2, i]
                Atomix.@atomic h∇[3, bin, jdx, node] += ∇[3, i]
            end
        end
    end
end

@kernel function hist_kernel_selective!(
    h∇::AbstractArray{T,4}, 
    @Const(∇), 
    @Const(x_bin), 
    @Const(nidx), 
    @Const(js), 
    @Const(target_mask)
) where {T}
    i, j = @index(Global, NTuple)
    
    @inbounds if i <= size(x_bin, 1) && j <= length(js)
        node_u = nidx[i]
        
        if node_u > zero(UInt32)
            node = Int(node_u)
            
            if node <= length(target_mask) && target_mask[node]
                jdx = Int(js[j])
                bin = Int(x_bin[i, jdx])
                
                if bin > 0 && bin <= size(h∇, 2) && node <= size(h∇, 4)
                    Atomix.@atomic h∇[1, bin, jdx, node] += ∇[1, i]
                    Atomix.@atomic h∇[2, bin, jdx, node] += ∇[2, i]
                    Atomix.@atomic h∇[3, bin, jdx, node] += ∇[3, i]
                end
            end
        end
    end
end

@kernel function subtract_hist_kernel!(
    h∇::AbstractArray{T,4}, 
    @Const(parent_nodes), 
    @Const(left_nodes), 
    @Const(right_nodes)
) where {T}
    idx, feat, bin = @index(Global, NTuple)
    
    @inbounds if idx <= length(parent_nodes) && feat <= size(h∇, 3) && bin <= size(h∇, 2)
        parent = Int(parent_nodes[idx])
        left = Int(left_nodes[idx])
        right = Int(right_nodes[idx])
        
        if parent > 0 && parent <= size(h∇, 4) &&
           left > 0 && left <= size(h∇, 4) &&
           right > 0 && right <= size(h∇, 4)
            h∇[1, bin, feat, right] = h∇[1, bin, feat, parent] - h∇[1, bin, feat, left]
            h∇[2, bin, feat, right] = h∇[2, bin, feat, parent] - h∇[2, bin, feat, left]
            h∇[3, bin, feat, right] = h∇[3, bin, feat, parent] - h∇[3, bin, feat, left]
        end
    end
end

@kernel function scan_and_find_best_split_for_feature_kernel!(
    gains_feats::AbstractMatrix{T}, 
    bins_feats::AbstractMatrix{Int32},
    @Const(h∇), 
    @Const(nodes_sum), 
    @Const(active_nodes),
    lambda::T, 
    min_weight::T
) where {T}
    n_idx, feat = @index(Global, NTuple)
    tid = @index(Local, Linear)
    
    nbins = size(h∇, 2)
    group_size = @uniform @groupsize()[1]
    
    # Allocate shared memory for the workgroup
    shmem_g1 = @localmem T (256,)
    shmem_g2 = @localmem T (256,)
    shmem_g3 = @localmem T (256,)
    g_best_sh = @localmem T (256,)
    b_best_sh = @localmem Int32 (256,)
    
    # Initialize shared memory for all threads
    @inbounds if tid <= 256
        shmem_g1[tid] = zero(T)
        shmem_g2[tid] = zero(T)
        shmem_g3[tid] = zero(T)
        g_best_sh[tid] = T(-Inf)
        b_best_sh[tid] = Int32(0)
    end
    @synchronize()
    
    @inbounds if n_idx <= length(active_nodes) && feat <= size(h∇, 3)
        node = Int(active_nodes[n_idx])
        
        # Load histogram data into shared memory
        if tid <= nbins && node > 0 && node <= size(h∇, 4)
            shmem_g1[tid] = h∇[1, tid, feat, node]
            shmem_g2[tid] = h∇[2, tid, feat, node]
            shmem_g3[tid] = h∇[3, tid, feat, node]
        end
        @synchronize()
        
        # Parallel prefix sum (scan) - using doubling algorithm
        offset = 1
        while offset < nbins
            @synchronize()
            if tid > offset && tid <= nbins
                temp1 = shmem_g1[tid - offset]
                temp2 = shmem_g2[tid - offset]
                temp3 = shmem_g3[tid - offset]
                shmem_g1[tid] = shmem_g1[tid] + temp1
                shmem_g2[tid] = shmem_g2[tid] + temp2
                shmem_g3[tid] = shmem_g3[tid] + temp3
            end
            offset = offset << 1
        end
        @synchronize()
        
        # Find best split for this thread's bin
        if tid < nbins && node > 0 && node <= size(nodes_sum, 2)
            p_g1 = nodes_sum[1, node]
            p_g2 = nodes_sum[2, node]
            p_w = nbins > 0 ? shmem_g3[nbins] : zero(T)
            gain_p = p_g1^2 / (p_g2 + lambda + T(1e-8))
            
            l_w = shmem_g3[tid]
            r_w = p_w - l_w
            
            if l_w >= min_weight && r_w >= min_weight
                l_g1 = shmem_g1[tid]
                l_g2 = shmem_g2[tid]
                r_g1 = p_g1 - l_g1
                r_g2 = p_g2 - l_g2
                gain_l = l_g1^2 / (l_g2 + lambda + T(1e-8))
                gain_r = r_g1^2 / (r_g2 + lambda + T(1e-8))
                g_best_sh[tid] = gain_l + gain_r - gain_p
                b_best_sh[tid] = Int32(tid)
            end
        end
        @synchronize()
        
        # Reduction to find maximum gain
        stride = group_size >> 1
        while stride > 0
            @synchronize()
            if tid <= stride
                other_idx = tid + stride
                if other_idx <= group_size && g_best_sh[other_idx] > g_best_sh[tid]
                    g_best_sh[tid] = g_best_sh[other_idx]
                    b_best_sh[tid] = b_best_sh[other_idx]
                end
            end
            stride = stride >> 1
        end
        @synchronize()
        
        # Write out the best result
        if tid == 1
            gains_feats[n_idx, feat] = g_best_sh[1]
            bins_feats[n_idx, feat] = b_best_sh[1]
        end
    end
end

@kernel function reduce_across_features_kernel!(
    gains::AbstractVector{T}, 
    bins::AbstractVector{Int32}, 
    feats::AbstractVector{Int32},
    @Const(gains_feats), 
    @Const(bins_feats), 
    @Const(active_nodes)
) where {T}
    n_idx = @index(Group, Linear)
    tid = @index(Local, Linear)
    
    block_size = @uniform @groupsize()[1]
    nfeats = size(gains_feats, 2)
    
    # Shared memory for reduction
    g_best_sh = @localmem T (256,)
    b_best_sh = @localmem Int32 (256,)
    f_best_sh = @localmem Int32 (256,)
    
    # Initialize shared memory
    @inbounds if tid <= 256
        g_best_sh[tid] = typemin(T)
        b_best_sh[tid] = Int32(0)
        f_best_sh[tid] = Int32(0)
    end
    @synchronize()
    
    @inbounds if n_idx <= length(active_nodes)
        # Each thread checks multiple features
        best_g = typemin(T)
        best_b = Int32(0)
        best_f = Int32(0)
        
        for f in tid:block_size:nfeats
            if f <= nfeats
                g = gains_feats[n_idx, f]
                if g > best_g
                    best_g = g
                    best_b = bins_feats[n_idx, f]
                    best_f = Int32(f)
                end
            end
        end
        
        g_best_sh[tid] = best_g
        b_best_sh[tid] = best_b
        f_best_sh[tid] = best_f
        @synchronize()
        
        # Reduction
        stride = block_size >> 1
        while stride > 0
            @synchronize()
            if tid <= stride
                other_idx = tid + stride
                if other_idx <= block_size && g_best_sh[other_idx] > g_best_sh[tid]
                    g_best_sh[tid] = g_best_sh[other_idx]
                    b_best_sh[tid] = b_best_sh[other_idx]
                    f_best_sh[tid] = f_best_sh[other_idx]
                end
            end
            stride = stride >> 1
        end
        @synchronize()
        
        if tid == 1
            gains[n_idx] = g_best_sh[1]
            bins[n_idx] = b_best_sh[1]
            feats[n_idx] = f_best_sh[1]
        end
    end
end

# Cumulative sum kernel
@kernel function cumsum_kernel!(
    h∇L::AbstractArray{T,4}, 
    @Const(h∇), 
    @Const(active_nodes)
) where {T}
    node_idx, feat = @index(Global, NTuple)
    
    @inbounds if node_idx <= length(active_nodes) && feat <= size(h∇, 3)
        node = Int(active_nodes[node_idx])
        
        if node > 0 && node <= size(h∇, 4)
            sum1 = zero(T)
            sum2 = zero(T)
            sum3 = zero(T)
            
            for bin in 1:size(h∇, 2)
                sum1 += h∇[1, bin, feat, node]
                sum2 += h∇[2, bin, feat, node]
                sum3 += h∇[3, bin, feat, node]
                
                h∇L[1, bin, feat, node] = sum1
                h∇L[2, bin, feat, node] = sum2
                h∇L[3, bin, feat, node] = sum3
            end
        end
    end
end

function update_hist_gpu!(
    h∇, h∇L, gains, bins, feats, ∇, x_bin, nidx, js, 
    depth, active_nodes, nodes_sum_gpu, gpu_params, 
    gains_feats, bins_feats
)
    backend = KernelAbstractions.get_backend(h∇)
    n_nodes_level = 2^(depth - 1)
    
    if depth == 1
        # Process root node
        h∇[:, :, :, 1] .= 0
        kernel_hist! = hist_kernel!(backend)
        kernel_hist!(h∇, ∇, x_bin, nidx, js; ndrange=(size(x_bin, 1), length(js)))
        KernelAbstractions.synchronize(backend)
    else
        # Process left children using histogram subtraction
        dnodes = n_nodes_level:(2^depth - 1)
        left_nodes_range = dnodes[1:2:end]
        
        # Allocate arrays on the correct backend
        left_nodes = KernelAbstractions.allocate(backend, Int32, length(left_nodes_range))
        parent_nodes = KernelAbstractions.allocate(backend, Int32, length(left_nodes_range))
        right_nodes = KernelAbstractions.allocate(backend, Int32, length(left_nodes_range))
        
        copyto!(left_nodes, Int32.(left_nodes_range))
        copyto!(parent_nodes, Int32.(left_nodes_range .>> 1))
        copyto!(right_nodes, left_nodes .+ Int32(1))
        
        # Clear left nodes histograms
        for node in left_nodes_range
            h∇[:, :, :, node] .= 0
        end
        
        # Create target mask
        target_mask = KernelAbstractions.zeros(backend, Bool, size(h∇, 4))
        for node in left_nodes_range
            target_mask[node] = true
        end
        
        # Build histograms for left children
        kernel_hist_selective! = hist_kernel_selective!(backend)
        kernel_hist_selective!(h∇, ∇, x_bin, nidx, js, target_mask; 
                             ndrange=(size(x_bin, 1), length(js)))
        KernelAbstractions.synchronize(backend)
        
        # Subtract to get right children
        kernel_subtract! = subtract_hist_kernel!(backend)
        kernel_subtract!(h∇, parent_nodes, left_nodes, right_nodes; 
                        ndrange=(length(parent_nodes), size(h∇, 3), size(h∇, 2)))
        KernelAbstractions.synchronize(backend)
    end
    
    nbins = size(h∇, 2)
    nfeats = size(h∇, 3)
    num_active_nodes = length(active_nodes)
    
    # Find best splits - ensure workgroup size doesn't exceed 256
    threads_scan = min(256, nbins)
    kernel_scan_split! = scan_and_find_best_split_for_feature_kernel!(backend, threads_scan)
    kernel_scan_split!(gains_feats, bins_feats, h∇, nodes_sum_gpu, active_nodes, 
                       gpu_params.lambda, gpu_params.min_weight; 
                       ndrange=(num_active_nodes, nfeats))
    KernelAbstractions.synchronize(backend)
    
    # Reduce across features
    threads_reduce = 256
    kernel_reduce! = reduce_across_features_kernel!(backend, threads_reduce)
    kernel_reduce!(gains, bins, feats, gains_feats, bins_feats, active_nodes; 
                   ndrange=num_active_nodes)
    KernelAbstractions.synchronize(backend)
    
    # Compute cumulative sums for split application
    kernel_cumsum! = cumsum_kernel!(backend)
    kernel_cumsum!(h∇L, h∇, active_nodes; ndrange=(num_active_nodes, nfeats))
    KernelAbstractions.synchronize(backend)
    
    return nothing
end

