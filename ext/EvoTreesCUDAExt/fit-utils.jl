using KernelAbstractions
using CUDA
using Atomix

@kernel function update_nodes_idx_kernel!(
    nidx::AbstractVector{UInt32}, @Const(is), @Const(x_bin), 
    @Const(cond_feats), @Const(cond_bins), @Const(feattypes)
)
    gidx = @index(Global)
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
    h∇::AbstractArray{T,4}, @Const(∇), @Const(x_bin), 
    @Const(nidx), @Const(js)
) where {T}
    i, j = @index(Global, NTuple)
    @inbounds if i <= size(x_bin, 1) && j <= length(js)
        node_u = nidx[i]
        if node_u > zero(UInt32)
            node = Int(node_u)
            jdx = Int(js[j])
            bin = Int(x_bin[i, jdx])
            if bin > 0 && bin <= size(h∇, 2)
                Atomix.@atomic h∇[1, bin, jdx, node] += ∇[1, i]
                Atomix.@atomic h∇[2, bin, jdx, node] += ∇[2, i]
                Atomix.@atomic h∇[3, bin, jdx, node] += ∇[3, i]
            end
        end
    end
end

@kernel function hist_kernel_selective!(
    h∇::AbstractArray{T,4}, @Const(∇), @Const(x_bin), 
    @Const(nidx), @Const(js), @Const(target_mask)
) where {T}
    i, j = @index(Global, NTuple)
    @inbounds if i <= size(x_bin, 1) && j <= length(js)
        node_u = nidx[i]
        if node_u > zero(UInt32) && target_mask[Int(node_u)]
            node = Int(node_u)
            jdx = Int(js[j])
            bin = Int(x_bin[i, jdx])
            if bin > 0 && bin <= size(h∇, 2)
                Atomix.@atomic h∇[1, bin, jdx, node] += ∇[1, i]
                Atomix.@atomic h∇[2, bin, jdx, node] += ∇[2, i]
                Atomix.@atomic h∇[3, bin, jdx, node] += ∇[3, i]
            end
        end
    end
end

@kernel function subtract_hist_kernel!(
    h∇::AbstractArray{T,4}, @Const(parent_nodes), 
    @Const(left_nodes), @Const(right_nodes)
) where {T}
    idx, feat, bin = @index(Global, NTuple)
    @inbounds if idx <= length(parent_nodes) && feat <= size(h∇, 3) && bin <= size(h∇, 2)
        parent = Int(parent_nodes[idx])
        left = Int(left_nodes[idx])
        right = Int(right_nodes[idx])
        if parent > 0 && left > 0 && right > 0
            h∇[1, bin, feat, right] = h∇[1, bin, feat, parent] - h∇[1, bin, feat, left]
            h∇[2, bin, feat, right] = h∇[2, bin, feat, parent] - h∇[2, bin, feat, left]
            h∇[3, bin, feat, right] = h∇[3, bin, feat, parent] - h∇[3, bin, feat, left]
        end
    end
end

@kernel function scan_and_find_best_split_for_feature_kernel!(
    gains_feats::AbstractMatrix{T}, bins_feats::AbstractMatrix{Int32},
    @Const(h∇), @Const(nodes_sum), @Const(active_nodes),
    lambda::T, min_weight::T
) where {T}
    n_idx, feat = @index(Global, NTuple)
    tid = @index(Local, Linear)

    nbins = size(h∇, 2)
    @assert nbins <= 256 "Nbins > 256 not supported by this kernel"

    shmem_g1 = @localmem T (256,); shmem_g2 = @localmem T (256,); shmem_g3 = @localmem T (256,)
    g_best_sh = @localmem T (256,); b_best_sh = @localmem Int32 (256,)

    if tid <= nbins
        shmem_g1[tid] = zero(T); shmem_g2[tid] = zero(T); shmem_g3[tid] = zero(T)
    end
    g_best_sh[tid] = T(-Inf); b_best_sh[tid] = Int32(0)
    @synchronize()

    @inbounds if n_idx <= length(active_nodes) && feat <= size(h∇, 3)
        if tid <= nbins
            node = Int(active_nodes[n_idx])
            shmem_g1[tid] = h∇[1, tid, feat, node]
            shmem_g2[tid] = h∇[2, tid, feat, node]
            shmem_g3[tid] = h∇[3, tid, feat, node]
        end
        @synchronize()

        d = 1
        while d < nbins; @synchronize(); if tid > d; shmem_g1[tid] += shmem_g1[tid - d]; shmem_g2[tid] += shmem_g2[tid - d]; shmem_g3[tid] += shmem_g3[tid - d]; end; d <<= 1; end
        @synchronize()

        if tid <= nbins
            local gain_l::T = zero(T); local gain_r::T = zero(T)
            node = Int(active_nodes[n_idx])
            p_g1 = nodes_sum[1, node]; p_g2 = nodes_sum[2, node]; p_w = shmem_g3[nbins]
            gain_p = p_g1^2 / (p_g2 + lambda + T(1e-8))
            b = tid
            if b < nbins
                l_w = shmem_g3[b]; r_w = p_w - l_w
                if l_w >= min_weight && r_w >= min_weight
                    l_g1 = shmem_g1[b]; l_g2 = shmem_g2[b]
                    r_g1 = p_g1 - l_g1; r_g2 = p_g2 - l_g2
                    gain_l = l_g1^2 / (l_g2 + lambda + T(1e-8))
                    gain_r = r_g1^2 / (r_g2 + lambda + T(1e-8))
                    g_best_sh[tid] = gain_l + gain_r - gain_p
                    b_best_sh[tid] = Int32(b)
                end
            end
        end
        @synchronize()

        stride = blockDim().x ÷ 2
        while stride > 0; @synchronize(); if tid <= stride && (tid + stride) <= nbins; if g_best_sh[tid] < g_best_sh[tid + stride]; g_best_sh[tid] = g_best_sh[tid + stride]; b_best_sh[tid] = b_best_sh[tid + stride]; end; end; stride ÷= 2; end
        @synchronize()

        if tid == 1
            gains_feats[n_idx, feat] = g_best_sh[1]
            bins_feats[n_idx, feat] = b_best_sh[1]
        end
    end
end

@kernel function reduce_across_features_kernel!(
    gains::AbstractVector{T}, bins::AbstractVector{Int32}, feats::AbstractVector{Int32},
    @Const(gains_feats), @Const(bins_feats), @Const(active_nodes)
) where {T}
    n_idx = @index(Group, Linear); tid = @index(Local, Linear)
    block_size = @groups_size()[1]; nfeats = size(gains_feats, 2)

    g_best_sh = @localmem T (256,); b_best_sh = @localmem Int32 (256,); f_best_sh = @localmem Int32 (256,)
    g_best_sh[tid] = typemin(T); b_best_sh[tid] = Int32(0); f_best_sh[tid] = Int32(0)

    @inbounds if n_idx <= length(active_nodes)
        for f in tid:block_size:nfeats
            if f <= nfeats
                g = gains_feats[n_idx, f]
                if g > g_best_sh[tid]
                    g_best_sh[tid] = g
                    b_best_sh[tid] = bins_feats[n_idx, f]
                    f_best_sh[tid] = Int32(f)
                end
            end
        end
    end
    @synchronize()

    stride = block_size ÷ 2
    while stride > 0; @synchronize(); if tid <= stride && (tid + stride) <= block_size; if g_best_sh[tid] < g_best_sh[tid + stride]; g_best_sh[tid] = g_best_sh[tid + stride]; b_best_sh[tid] = b_best_sh[tid + stride]; f_best_sh[tid] = f_best_sh[tid + stride]; end; end; stride ÷= 2; end

    if tid == 1 && n_idx <= length(active_nodes)
        gains[n_idx] = g_best_sh[1]
        bins[n_idx] = b_best_sh[1]
        feats[n_idx] = f_best_sh[1]
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
        dnodes = 1:1
        h∇[:, :, :, dnodes] .= 0
        kernel_hist! = hist_kernel!(backend)
        kernel_hist!(h∇, ∇, x_bin, nidx, js; ndrange=(size(x_bin, 1), length(js)))
    else
        dnodes = n_nodes_level:(2^depth - 1)
        left_nodes = CuArray(Int32.(dnodes[1:2:end]))
        parent_nodes = CuArray(Int32.(dnodes[1:2:end] .>> 1))
        right_nodes = left_nodes .+ Int32(1)
        h∇[:, :, :, left_nodes] .= 0
        
        target_mask = CUDA.zeros(Bool, size(h∇, 4))
        CUDA.fill!(view(target_mask, left_nodes), true)

        kernel_hist_selective! = hist_kernel_selective!(backend)
        kernel_hist_selective!(h∇, ∇, x_bin, nidx, js, target_mask; ndrange=(size(x_bin, 1), length(js)))
        
        kernel_subtract! = subtract_hist_kernel!(backend)
        kernel_subtract!(h∇, parent_nodes, left_nodes, right_nodes; ndrange=(length(parent_nodes), size(h∇, 3), size(h∇, 2)))
    end
    KernelAbstractions.synchronize(backend)

    nbins = size(h∇, 2)
    nfeats = size(h∇, 3)
    num_active_nodes = length(active_nodes)
    
    @assert nbins <= 256
    threads_scan = nbins

    kernel_scan_split! = scan_and_find_best_split_for_feature_kernel!(backend, threads_scan)
    kernel_scan_split!(gains_feats, bins_feats, h∇, nodes_sum_gpu, active_nodes, gpu_params.lambda, gpu_params.min_weight; ndrange=(num_active_nodes, nfeats))
    KernelAbstractions.synchronize(backend)

    threads_reduce = 256
    kernel_reduce! = reduce_across_features_kernel!(backend, threads_reduce)
    kernel_reduce!(gains, bins, feats, gains_feats, bins_feats, active_nodes; ndrange=num_active_nodes)
    KernelAbstractions.synchronize(backend)
    
    # Generate h∇L needed by apply_splits! using an efficient library call
    active_nodes_cpu = Array(active_nodes)
    if !isempty(active_nodes_cpu)
        h∇_view = view(h∇, :, :, :, active_nodes_cpu)
        h∇L_view = view(h∇L, :, :, :, active_nodes_cpu)
        CUDA.cumsum!(h∇L_view, h∇_view, dims=2)
    end
    KernelAbstractions.synchronize(backend)

    return nothing
end

