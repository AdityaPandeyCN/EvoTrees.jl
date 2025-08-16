using KernelAbstractions
using CUDA
using Atomix

@kernel function update_nodes_idx_kernel!(nidx::AbstractVector{T}, @Const(is), @Const(x_bin), @Const(cond_feats), @Const(cond_bins), @Const(feattypes)) where {T<:Unsigned}
    gidx = @index(Global)
    @inbounds if gidx <= length(is)
        obs = is[gidx]
        node = nidx[obs]
        if node > 0
            feat = cond_feats[node]
            bin = cond_bins[node]
            if bin == 0
                nidx[obs] = zero(T)
            else
                feattype = feattypes[feat]
                is_left = feattype ? (x_bin[obs, feat] <= bin) : (x_bin[obs, feat] == bin)
                nidx[obs] = (node << 1) + !is_left
            end
        end
    end
end

@kernel function hist_kernel!(h∇::AbstractArray{T,4}, @Const(∇), @Const(x_bin), @Const(nidx), @Const(js)) where {T}
    i, j = @index(Global, NTuple)
    @inbounds if i <= size(x_bin, 1) && j <= length(js)
        node = nidx[i]
        if node > 0
            jdx = js[j]
            bin = x_bin[i, jdx]
            if bin > 0 && bin <= size(h∇, 2)
                Atomix.@atomic h∇[1, bin, jdx, node] += ∇[1, i]
                Atomix.@atomic h∇[2, bin, jdx, node] += ∇[2, i]
                Atomix.@atomic h∇[3, bin, jdx, node] += ∇[3, i]
            end
        end
    end
end

@kernel function hist_kernel_selective!(h∇::AbstractArray{T,4}, @Const(∇), @Const(x_bin), @Const(nidx), @Const(js), @Const(target_nodes)) where {T}
    i, j = @index(Global, NTuple)
    @inbounds if i <= size(x_bin, 1) && j <= length(js)
        node = nidx[i]
        if node > 0 && node in target_nodes
            jdx = js[j]
            bin = x_bin[i, jdx]
            if bin > 0 && bin <= size(h∇, 2)
                Atomix.@atomic h∇[1, bin, jdx, node] += ∇[1, i]
                Atomix.@atomic h∇[2, bin, jdx, node] += ∇[2, i]
                Atomix.@atomic h∇[3, bin, jdx, node] += ∇[3, i]
            end
        end
    end
end

@kernel function subtract_hist_kernel!(h∇::AbstractArray{T,4}, @Const(parent_nodes), @Const(left_nodes), @Const(right_nodes)) where {T}
    idx, feat, bin = @index(Global, NTuple)
    @inbounds if idx <= length(parent_nodes) && feat <= size(h∇, 3) && bin <= size(h∇, 2)
        parent = parent_nodes[idx]
        left = left_nodes[idx]
        right = right_nodes[idx]
        if parent > 0 && left > 0 && right > 0
            h∇[1, bin, feat, right] = h∇[1, bin, feat, parent] - h∇[1, bin, feat, left]
            h∇[2, bin, feat, right] = h∇[2, bin, feat, parent] - h∇[2, bin, feat, left]
            h∇[3, bin, feat, right] = h∇[3, bin, feat, parent] - h∇[3, bin, feat, left]
        end
    end
end

@kernel function scan_and_find_best_split_for_feature_kernel!(gains_feats::AbstractMatrix{T}, bins_feats::AbstractMatrix{Int32}, @Const(h∇), @Const(nodes_sum), @Const(active_nodes), lambda::T, min_weight::T) where {T}
    n_idx, feat = @index(Global, NTuple)
    tid = @index(Local, Linear)
    nbins = @groupsize()[1]

    @localmem shmem_g1::Vector{T} nbins
    @localmem shmem_g2::Vector{T} nbins
    @localmem shmem_g3::Vector{T} nbins
    @localmem g_best_sh::Vector{T} nbins
    @localmem b_best_sh::Vector{Int32} nbins

    g_best_sh[tid] = T(-Inf)
    b_best_sh[tid] = Int32(0)

    @inbounds if n_idx <= length(active_nodes) && feat <= size(h∇, 3)
        node = active_nodes[n_idx]
        shmem_g1[tid] = h∇[1, tid, feat, node]
        shmem_g2[tid] = h∇[2, tid, feat, node]
        shmem_g3[tid] = h∇[3, tid, feat, node]
    else
        shmem_g1[tid] = zero(T); shmem_g2[tid] = zero(T); shmem_g3[tid] = zero(T)
    end
    @synchronize()

    d = 1
    while d < nbins
        @synchronize()
        if tid > d
            shmem_g1[tid] += shmem_g1[tid - d]
            shmem_g2[tid] += shmem_g2[tid - d]
            shmem_g3[tid] += shmem_g3[tid - d]
        end
        d *= 2
    end
    @synchronize()

    @inbounds if n_idx <= length(active_nodes) && feat <= size(h∇, 3)
        node = active_nodes[n_idx]
        p_g1 = T(nodes_sum[1, node])
        p_g2 = T(nodes_sum[2, node])
        p_w = shmem_g3[nbins]
        gain_p = p_g1^2 / (p_g2 + lambda + T(1e-8))

        b = tid
        if b < nbins
            l_w = shmem_g3[b]
            r_w = p_w - l_w
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

    stride = nbins ÷ 2
    while stride > 0
        @synchronize()
        if tid <= stride && g_best_sh[tid] < g_best_sh[tid + stride]
            g_best_sh[tid] = g_best_sh[tid + stride]
            b_best_sh[tid] = b_best_sh[tid + stride]
        end
        stride ÷= 2
    end

    @inbounds if tid == 1 && n_idx <= length(active_nodes) && feat <= size(h∇, 3)
        gains_feats[n_idx, feat] = g_best_sh[1]
        bins_feats[n_idx, feat] = b_best_sh[1]
    end
end

@kernel function reduce_across_features_kernel!(gains::AbstractVector{T}, bins::AbstractVector{Int32}, feats::AbstractVector{Int32}, @Const(gains_feats), @Const(bins_feats), @Const(active_nodes)) where {T}
    n_idx, tid = @index(Global, NTuple)
    tid = @index(Local, Linear)
    block_size = @groupsize()[1]
    nfeats = size(gains_feats, 2)

    @localmem g_best_sh::Vector{T} block_size
    @localmem b_best_sh::Vector{Int32} block_size
    @localmem f_best_sh::Vector{Int32} block_size

    g_best_sh[tid] = T(-Inf)
    b_best_sh[tid] = Int32(0)
    f_best_sh[tid] = Int32(0)

    @inbounds if n_idx <= length(active_nodes)
        for f in tid:block_size:nfeats
            g = gains_feats[n_idx, f]
            if g > g_best_sh[tid]
                g_best_sh[tid] = g
                b_best_sh[tid] = bins_feats[n_idx, f]
                f_best_sh[tid] = Int32(f)
            end
        end
    end
    @synchronize()

    stride = block_size ÷ 2
    while stride > 0
        @synchronize()
        if tid <= stride && g_best_sh[tid] < g_best_sh[tid + stride]
            g_best_sh[tid] = g_best_sh[tid + stride]
            b_best_sh[tid] = b_best_sh[tid + stride]
            f_best_sh[tid] = f_best_sh[tid + stride]
        end
        stride ÷= 2
    end

    @inbounds if tid == 1 && n_idx <= length(active_nodes)
        gains[n_idx] = g_best_sh[1]
        bins[n_idx] = b_best_sh[1]
        feats[n_idx] = f_best_sh[1]
    end
end

function update_hist_gpu!(h∇, gains, bins, feats, ∇, x_bin, nidx, js, depth, active_nodes, nodes_sum_gpu, params, gains_feats, bins_feats)
    backend = KernelAbstractions.get_backend(h∇)
    n_nodes_level = 2^(depth - 1)
    
    if depth == 1
        dnodes = 1:1
        h∇[:, :, :, dnodes] .= 0
        kernel_hist! = hist_kernel!(backend)
        kernel_hist!(h∇, ∇, x_bin, nidx, js; ndrange=(size(x_bin, 1), length(js)))
    else
        dnodes = n_nodes_level:(2^depth - 1)
        left_nodes = CuArray(dnodes[1:2:end])
        right_nodes = CuArray(dnodes[2:2:end])
        parent_nodes = CuArray(div.(dnodes, 2))
        h∇[:, :, :, left_nodes] .= 0

        kernel_hist_selective! = hist_kernel_selective!(backend)
        kernel_hist_selective!(h∇, ∇, x_bin, nidx, js, left_nodes; ndrange=(size(x_bin, 1), length(js)))

        kernel_subtract! = subtract_hist_kernel!(backend)
        kernel_subtract!(h∇, parent_nodes, left_nodes, right_nodes; ndrange=(length(parent_nodes), size(h∇, 3), size(h∇, 2)))
    end

    nbins = size(h∇, 2)

    nfeats = length(js)
    num_active_nodes = length(active_nodes)

    threads_per_block_scan = nbins
    grid_size_scan = (num_active_nodes, nfeats)
    kernel_scan_split! = scan_and_find_best_split_for_feature_kernel!(backend, threads_per_block_scan)
    local Tloc = eltype(gains_feats)
    kernel_scan_split!(gains_feats, bins_feats, h∇, nodes_sum_gpu, active_nodes, Tloc(params.lambda), Tloc(params.min_weight); ndrange=grid_size_scan)

    threads_per_block_reduce = min(256, nfeats)
    grid_size_reduce = num_active_nodes
    kernel_reduce! = reduce_across_features_kernel!(backend, threads_per_block_reduce)
    kernel_reduce!(gains, bins, feats, gains_feats, bins_feats, active_nodes; ndrange=grid_size_reduce)

    KernelAbstractions.synchronize(backend)
    return nothing
end

