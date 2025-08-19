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
                nidx[obs] = (node << 1) + T(Int(!is_left))
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

@kernel function zero_node_hist_kernel!(h∇::AbstractArray{T,4}, @Const(nodes), @Const(js)) where {T}
    idx, j_idx, bin = @index(Global, NTuple)
    @inbounds if idx <= length(nodes) && j_idx <= length(js) && bin <= size(h∇, 2)
        node = nodes[idx]
        feat = js[j_idx]
        if node > 0
            h∇[1, bin, feat, node] = zero(T)
            h∇[2, bin, feat, node] = zero(T)
            h∇[3, bin, feat, node] = zero(T)
        end
    end
end

@kernel function find_best_split_kernel_parallel!(
    gains::AbstractVector{T},
    bins::AbstractVector{Int32},
    feats::AbstractVector{Int32},
    @Const(hL),
    @Const(hR),
    nodes_sum,
    @Const(active_nodes),
    @Const(js),
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
        
        g_best = T(-Inf)
        b_best = Int32(0)
        f_best = Int32(0)
        
        # Get node sum from the first feature in js (they should all be the same)
        f_first = js[1]
        p_g1 = hR[1, nbins, f_first, node]
        p_g2 = hR[2, nbins, f_first, node]
        p_w  = hR[3, nbins, f_first, node]
        # Write to nodes_sum for use in apply_splits
        nodes_sum[1, node] = p_g1
        nodes_sum[2, node] = p_g2
        nodes_sum[3, node] = p_w
        
        gain_p = p_g1^2 / (p_g2 + lambda * p_w + T(1e-8))
        
        @inbounds for j_idx in 1:length(js)
            f = js[j_idx]
            f_w = hR[3, nbins, f, node]
            for b in 1:(nbins - 1)
                l_w = hL[3, b, f, node]
                r_w = f_w - l_w
                if l_w >= min_weight && r_w >= min_weight
                    l_g1 = hL[1, b, f, node]
                    l_g2 = hL[2, b, f, node]
                    r_g1 = p_g1 - l_g1
                    r_g2 = p_g2 - l_g2
                        gain_l = l_g1^2 / (l_g2 + lambda * l_w + T(1e-8))
                        gain_r = r_g1^2 / (r_g2 + lambda * r_w + T(1e-8))
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

@kernel function find_best_split_from_hist_kernel!(
    gains::AbstractVector{T},
    bins::AbstractVector{Int32},
    feats::AbstractVector{Int32},
    @Const(h∇),
    nodes_sum,
    @Const(active_nodes),
    @Const(js),
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
            nbins = size(h∇, 2)
            # compute parent sums using first feature (identical across features)
            f_first = js[1]
            p_g1 = zero(T); p_g2 = zero(T); p_w = zero(T)
            @inbounds for b in 1:nbins
                p_g1 += h∇[1, b, f_first, node]
                p_g2 += h∇[2, b, f_first, node]
                p_w  += h∇[3, b, f_first, node]
            end
            nodes_sum[1, node] = p_g1
            nodes_sum[2, node] = p_g2
            nodes_sum[3, node] = p_w
            
            gain_p = p_g1^2 / (p_g2 + lambda * p_w + T(1e-8))
            
            g_best = T(-Inf)
            b_best = Int32(0)
            f_best = Int32(0)
            
            @inbounds for j_idx in 1:length(js)
                f = js[j_idx]
                s1 = zero(T); s2 = zero(T); s3 = zero(T)
                @inbounds for b in 1:(nbins - 1)
                    s1 += h∇[1, b, f, node]
                    s2 += h∇[2, b, f, node]
                    s3 += h∇[3, b, f, node]
                    l_w = s3
                    r_w = p_w - l_w
                    if l_w >= min_weight && r_w >= min_weight
                        l_g1 = s1
                        l_g2 = s2
                        r_g1 = p_g1 - l_g1
                        r_g2 = p_g2 - l_g2
                        gain_l = l_g1^2 / (l_g2 + lambda * l_w + T(1e-8))
                        gain_r = r_g1^2 / (r_g2 + lambda * r_w + T(1e-8))
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

@kernel function hist_kernel_is_tiled!(
    h∇::AbstractArray{T,4},
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
    @Const(is),
    obs_per_thread::Int32,
) where {T}
    tile_i, j = @index(Global, NTuple)
    
    @inbounds if j <= length(js)
        start_idx = (tile_i - 1) * obs_per_thread + 1
        if start_idx <= length(is)
            end_idx = min(tile_i * obs_per_thread, length(is))
            jdx = js[j]
            @inbounds for i in start_idx:end_idx
                obs = is[i]
                node = nidx[obs]
                if node > 0
                    bin = x_bin[obs, jdx]
                    if bin > 0 && bin <= size(h∇, 2)
                        Atomix.@atomic h∇[1, bin, jdx, node] += ∇[1, obs]
                        Atomix.@atomic h∇[2, bin, jdx, node] += ∇[2, obs]
                        Atomix.@atomic h∇[3, bin, jdx, node] += ∇[3, obs]
                    end
                end
            end
        end
    end
end

@kernel function hist_kernel_selective_mask_is_tiled!(
    h∇::AbstractArray{T,4},
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
    @Const(target_mask),
    @Const(is),
    obs_per_thread::Int32,
) where {T}
    tile_i, j = @index(Global, NTuple)
    
    @inbounds if j <= length(js)
        start_idx = (tile_i - 1) * obs_per_thread + 1
        if start_idx <= length(is)
            end_idx = min(tile_i * obs_per_thread, length(is))
            jdx = js[j]
            @inbounds for i in start_idx:end_idx
                obs = is[i]
                node = nidx[obs]
                if node > 0 && target_mask[node] != 0
                    bin = x_bin[obs, jdx]
                    if bin > 0 && bin <= size(h∇, 2)
                        Atomix.@atomic h∇[1, bin, jdx, node] += ∇[1, obs]
                        Atomix.@atomic h∇[2, bin, jdx, node] += ∇[2, obs]
                        Atomix.@atomic h∇[3, bin, jdx, node] += ∇[3, obs]
                    end
                end
            end
        end
    end
end

@kernel function hist_kernel_shared64!(
    h∇::AbstractArray{T,4},
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
    @Const(is),
    @Const(active_nodes),
) where {T}
    # Group over (node_idx, feature_idx), threads iterate observations with stride
    g_node, g_feat = @index(Group, NTuple)
    t = @index(Local)
    nt = @groupsize()

    # Bounds check
    @inbounds if g_node <= length(active_nodes) && g_feat <= length(js)
        node = active_nodes[g_node]
        feat = js[g_feat]
        # shared-memory histograms for this (node, feat)
        sh1 = @localmem(T, 64)
        sh2 = @localmem(T, 64)
        sh3 = @localmem(T, 64)

        # zero shared hist (each thread zeros a strided subset)
        @inbounds for b = t:nt:64
            sh1[b] = zero(T)
            sh2[b] = zero(T)
            sh3[b] = zero(T)
        end
        @barrier()

        # accumulate observations into shared hist (strided by thread id)
        @inbounds for i = t:nt:length(is)
            obs = is[i]
            if nidx[obs] == node
                bin = Int(x_bin[obs, feat])
                if bin > 0 && bin <= 64
                    Atomix.@atomic sh1[bin] += ∇[1, obs]
                    Atomix.@atomic sh2[bin] += ∇[2, obs]
                    Atomix.@atomic sh3[bin] += ∇[3, obs]
                end
            end
        end
        @barrier()

        # flush shared hist to global once per bin (strided across threads)
        @inbounds for b = t:nt:64
            Atomix.@atomic h∇[1, b, feat, node] += sh1[b]
            Atomix.@atomic h∇[2, b, feat, node] += sh2[b]
            Atomix.@atomic h∇[3, b, feat, node] += sh3[b]
        end
    end
end

@kernel function hist_kernel_shared64_depth1!(
    h∇::AbstractArray{T,4},
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
    @Const(is),
) where {T}
    # Special case for depth 1: node is always 1
    g_feat = @index(Group)
    t = @index(Local)
    nt = @groupsize()

    @inbounds if g_feat <= length(js)
        node = Int32(1)
        feat = js[g_feat]

        sh1 = @localmem(T, 64)
        sh2 = @localmem(T, 64)
        sh3 = @localmem(T, 64)

        @inbounds for b = t:nt:64
            sh1[b] = zero(T)
            sh2[b] = zero(T)
            sh3[b] = zero(T)
        end
        @barrier()

        @inbounds for i = t:nt:length(is)
            obs = is[i]
            # nidx may be uninitialized for depth 1, but node is 1 for root
            bin = Int(x_bin[obs, feat])
            if bin > 0 && bin <= 64
                Atomix.@atomic sh1[bin] += ∇[1, obs]
                Atomix.@atomic sh2[bin] += ∇[2, obs]
                Atomix.@atomic sh3[bin] += ∇[3, obs]
            end
        end
        @barrier()

        @inbounds for b = t:nt:64
            # write to global hist for root node
            Atomix.@atomic h∇[1, b, feat, node] += sh1[b]
            Atomix.@atomic h∇[2, b, feat, node] += sh2[b]
            Atomix.@atomic h∇[3, b, feat, node] += sh3[b]
        end
    end
end

function update_hist_gpu!(
    h∇, hL, hR, gains, bins, feats, ∇, x_bin, nidx, js, is, depth, active_nodes, nodes_sum_gpu, params,
    left_nodes_buf, right_nodes_buf, target_mask_buf
)
    backend = KernelAbstractions.get_backend(h∇)
    
    n_active = length(active_nodes)
    
    if depth == 1
        h∇ .= 0
        # Shared-memory histogram for depth 1 (root node)
        hist_shared64_depth1! = hist_kernel_shared64_depth1!(backend)
        hist_shared64_depth1!(h∇, ∇, x_bin, nidx, js, is; ndrange = length(js), workgroupsize = 128)
        hist_shared64_depth1! = hist_kernel_shared64_depth1!(backend)
        hist_shared64_depth1!(h∇, ∇, x_bin, nidx, js, is; ndrange = length(js))
    else
        # Build histograms for the current active nodes (parents to split now)
        zero_nodes! = zero_node_hist_kernel!(backend)
        zero_nodes!(h∇, active_nodes, js; ndrange = (n_active, length(js), size(h∇, 2)))

        target_mask_buf .= 0
        fill_mask! = fill_mask_kernel!(backend)
        fill_mask!(target_mask_buf, active_nodes; ndrange = n_active)

        # Shared-memory histogram for active nodes
        hist_shared64! = hist_kernel_shared64!(backend)
        hist_shared64!(h∇, ∇, x_bin, nidx, js, is, active_nodes; ndrange = (n_active, length(js)), workgroupsize = 128)
    end

    # Compute best splits directly from histograms, writing nodes_sum
    find_split! = find_best_split_from_hist_kernel!(backend)
    find_split!(
        gains, bins, feats, h∇, nodes_sum_gpu, active_nodes, js,
        eltype(gains)(params.lambda), eltype(gains)(params.min_weight);
        ndrange = n_active
    )
    
    KernelAbstractions.synchronize(backend)
    return nothing
end

