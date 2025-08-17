using KernelAbstractions
using Atomix

@kernel function update_nodes_idx_kernel!(
    nidx::AbstractVector{T},
    @Const(is),
    @Const(x_bin),
    @Const(cond_feats),
    @Const(cond_bins),
    @Const(feattypes),
) where {T<:Unsigned}
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

@kernel function hist_kernel!(
    h∇::AbstractArray{T,4},
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
) where {T}
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

@kernel function hist_kernel_selective!(
    h∇::AbstractArray{T,4},
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
    @Const(target_nodes),
) where {T}
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

@kernel function hist_kernel_selective_mask!(
    h∇::AbstractArray{T,4},
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
    @Const(target_mask),
) where {T}
    i, j = @index(Global, NTuple)
    @inbounds if i <= size(x_bin, 1) && j <= length(js)
        node = nidx[i]
        if node > 0 && target_mask[node] != 0
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

@kernel function fill_mask_kernel!(mask::AbstractVector{UInt8}, @Const(nodes))
    i = @index(Global)
    @inbounds if i <= length(nodes)
        node = nodes[i]
        if node > 0 && node <= length(mask)
            mask[node] = UInt8(1)
        end
    end
end

@kernel function fill_children_nodes_kernel!(left_nodes::AbstractVector{Int32}, right_nodes::AbstractVector{Int32}, @Const(parent_nodes))
    i = @index(Global)
    @inbounds if i <= length(parent_nodes)
        p = parent_nodes[i]
        left_nodes[i] = Int32(p << 1)
        right_nodes[i] = Int32((p << 1) + 1)
    end
end

@kernel function zero_node_hist_kernel!(h∇::AbstractArray{T,4}, @Const(nodes)) where {T}
    idx, feat, bin = @index(Global, NTuple)
    @inbounds if idx <= length(nodes) && feat <= size(h∇, 3) && bin <= size(h∇, 2)
        node = nodes[idx]
        if node > 0
            h∇[1, bin, feat, node] = zero(T)
            h∇[2, bin, feat, node] = zero(T)
            h∇[3, bin, feat, node] = zero(T)
        end
    end
end

@kernel function subtract_hist_kernel!(
    h∇::AbstractArray{T,4},
    @Const(parent_nodes),
    @Const(left_nodes),
    @Const(right_nodes),
) where {T}
    idx, feat, bin = @index(Global, NTuple)
    
    @inbounds if idx <= length(parent_nodes) && feat <= size(h∇, 3) && bin <= size(h∇, 2)
        parent = parent_nodes[idx]
        left = left_nodes[idx]
        right = right_nodes[idx]
        
        h∇[1, bin, feat, right] = max(T(0), h∇[1, bin, feat, parent] - h∇[1, bin, feat, left])
        h∇[2, bin, feat, right] = max(T(0), h∇[2, bin, feat, parent] - h∇[2, bin, feat, left])
        h∇[3, bin, feat, right] = max(T(0), h∇[3, bin, feat, parent] - h∇[3, bin, feat, left])
    end
end

@kernel function scan_hist_kernel_serial!(
    hL::AbstractArray{T,4},
    hR::AbstractArray{T,4},
    @Const(h∇),
    @Const(active_nodes),
) where {T}
    n_idx, feat = @index(Global, NTuple)
    
    nbins = size(h∇, 2)
    
    @inbounds if n_idx <= length(active_nodes) && feat <= size(h∇, 3)
        node = active_nodes[n_idx]
        if node > 0
            s1 = zero(T); s2 = zero(T); s3 = zero(T)
            @inbounds for bin in 1:nbins
                s1 += h∇[1, bin, feat, node]
                s2 += h∇[2, bin, feat, node]
                s3 += h∇[3, bin, feat, node]
                hL[1, bin, feat, node] = s1
                hL[2, bin, feat, node] = s2
                hL[3, bin, feat, node] = s3
            end
            hR[1, nbins, feat, node] = s1
            hR[2, nbins, feat, node] = s2
            hR[3, nbins, feat, node] = s3
        end
    end
end

@kernel function find_best_split_kernel_parallel!(
    gains::AbstractVector{T},
    bins::AbstractVector{Int32},
    feats::AbstractVector{Int32},
    @Const(hL),
    @Const(hR),
    @Const(nodes_sum),
    @Const(active_nodes),
    lambda::T,
    min_weight::T,
) where {T}
    n_idx = @index(Global)
    
    @inbounds if n_idx <= length(active_nodes)
        node = active_nodes[n_idx]
        
        if node == 0
            gains[n_idx] = T(-Inf)
            bins[n_idx] = Int32(0)
            feats[n_idx] = Int32(0)
        else
        nbins = size(hL, 2)
        nfeats = size(hL, 3)
        
        g_best = T(-Inf)
        b_best = Int32(0)
        f_best = Int32(0)
        
        p_g1 = nodes_sum[1, node]
        p_g2 = nodes_sum[2, node]
            gain_p = p_g1^2 / (p_g2 + lambda + 1e-8)
        
        for f in 1:nfeats
            p_w = hR[3, nbins, f, node]
            for b in 1:(nbins - 1)
                l_w = hL[3, b, f, node]
                r_w = p_w - l_w
                if l_w >= min_weight && r_w >= min_weight
                    l_g1 = hL[1, b, f, node]
                    l_g2 = hL[2, b, f, node]
                    r_g1 = p_g1 - l_g1
                    r_g2 = p_g2 - l_g2
                        gain_l = l_g1^2 / (l_g2 + lambda + 1e-8)
                        gain_r = r_g1^2 / (r_g2 + lambda + 1e-8)
                    g = gain_l + gain_r - gain_p
                    if g > g_best
                        g_best = g
                        b_best = Int32(b)
                        f_best = Int32(f)
                    end
                end
            end
        end
        
        gains[n_idx] = g_best
        bins[n_idx] = b_best
        feats[n_idx] = f_best
        end
    end
end

@kernel function fill_active_nodes_kernel!(active_nodes::AbstractVector{Int32}, offset::Int32)
    idx = @index(Global)
    @inbounds active_nodes[idx] = idx + offset
end

@kernel function hist_kernel_is!(
    h∇::AbstractArray{T,4},
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
    @Const(is),
) where {T}
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(is) && j <= length(js)
        obs = is[i]
        node = nidx[obs]
        if node > 0
            jdx = js[j]
            bin = x_bin[obs, jdx]
            if bin > 0 && bin <= size(h∇, 2)
                Atomix.@atomic h∇[1, bin, jdx, node] += ∇[1, obs]
                Atomix.@atomic h∇[2, bin, jdx, node] += ∇[2, obs]
                Atomix.@atomic h∇[3, bin, jdx, node] += ∇[3, obs]
            end
        end
    end
end

@kernel function hist_kernel_selective_mask_is!(
    h∇::AbstractArray{T,4},
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
    @Const(target_mask),
    @Const(is),
) where {T}
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(is) && j <= length(js)
        obs = is[i]
        node = nidx[obs]
        if node > 0 && target_mask[node] != 0
            jdx = js[j]
            bin = x_bin[obs, jdx]
            if bin > 0 && bin <= size(h∇, 2)
                Atomix.@atomic h∇[1, bin, jdx, node] += ∇[1, obs]
                Atomix.@atomic h∇[2, bin, jdx, node] += ∇[2, obs]
                Atomix.@atomic h∇[3, bin, jdx, node] += ∇[3, obs]
            end
        end
    end
end

function update_hist_gpu!(
    h∇, hL, hR, gains, bins, feats, ∇, x_bin, nidx, js, is, depth, active_nodes_gpu, nodes_sum_gpu, params
)
    backend = KernelAbstractions.get_backend(h∇)
    
    # Use the provided active nodes (1:n_active), with zeros meaning inactive
    active_nodes = active_nodes_gpu
    n_active = length(active_nodes)
    
    if depth == 1
        # Root level: build parent histogram from scratch
        h∇ .= 0
        kernel_hist_is! = hist_kernel_is!(backend)
        kernel_hist_is!(h∇, ∇, x_bin, nidx, js, is; ndrange = (length(is), length(js)))
    else
        # Build children histograms at current depth using selective accumulation for left children,
        # then derive right children by subtraction from their parent hist
        parent_nodes = active_nodes
        left_nodes = KernelAbstractions.zeros(backend, Int32, length(parent_nodes))
        right_nodes = KernelAbstractions.zeros(backend, Int32, length(parent_nodes))
        kernel_fill_children! = fill_children_nodes_kernel!(backend)
        kernel_fill_children!(left_nodes, right_nodes, parent_nodes; ndrange = length(parent_nodes))

        kernel_zero_nodes! = zero_node_hist_kernel!(backend)
        kernel_zero_nodes!(h∇, left_nodes; ndrange = (length(left_nodes), size(h∇, 3), size(h∇, 2)))
        kernel_zero_nodes!(h∇, right_nodes; ndrange = (length(right_nodes), size(h∇, 3), size(h∇, 2)))

        target_mask = KernelAbstractions.zeros(backend, UInt8, size(h∇, 4) + 1)
        kernel_fill_mask! = fill_mask_kernel!(backend)
        kernel_fill_mask!(target_mask, left_nodes; ndrange = length(left_nodes))

        kernel_hist_selective_mask_is! = hist_kernel_selective_mask_is!(backend)
        kernel_hist_selective_mask_is!(h∇, ∇, x_bin, nidx, js, target_mask, is;
                                       ndrange = (length(is), length(js)))
        
        kernel_subtract! = subtract_hist_kernel!(backend)
        kernel_subtract!(h∇, parent_nodes, left_nodes, right_nodes; 
                        ndrange = (length(parent_nodes), size(h∇, 3), size(h∇, 2)))
    end
    
    # Scan over the provided active nodes
    hL .= 0
    hR .= 0
    kernel_scan_serial! = scan_hist_kernel_serial!(backend)
    kernel_scan_serial!(hL, hR, h∇, active_nodes; ndrange = (length(active_nodes), size(h∇, 3)))
        KernelAbstractions.synchronize(backend)
        
    # Find best split per active node at this depth
    kernel_find_split! = find_best_split_kernel_parallel!(backend)
    kernel_find_split!(
        gains, bins, feats, hL, hR, nodes_sum_gpu, active_nodes,
        params.lambda + 1e-8, params.min_weight;
        ndrange = length(active_nodes)
    )
    
    KernelAbstractions.synchronize(backend)
    return nothing
end

