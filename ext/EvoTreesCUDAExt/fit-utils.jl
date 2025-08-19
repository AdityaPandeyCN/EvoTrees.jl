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

# Bin-threaded tiled histogram: one group per (node, feature),
# local threads correspond to bin indices; each thread accumulates its bin over obs.
@kernel function hist_kernel_bins_tiled!(
    h∇::AbstractArray{T,4},
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
    @Const(is),
    @Const(active_nodes),
) where {T}
    g_node, g_feat = @index(Group, NTuple)
    l_bin = @index(Local)

    @inbounds if g_node <= length(active_nodes) && g_feat <= length(js) && l_bin <= 64
        node_i = Int(active_nodes[g_node])
        node_u = UInt32(node_i)
        feat_i = Int(js[g_feat])
        s1 = T(0); s2 = T(0); s3 = T(0)
        i = l_bin
        @inbounds while i <= length(is)
            obs = is[i]
            if nidx[obs] == node_u
                b = Int(x_bin[obs, feat_i])
                if b == l_bin
                    s1 += ∇[1, obs]
                    s2 += ∇[2, obs]
                    s3 += ∇[3, obs]
                end
            end
            i += 64
        end
        Atomix.@atomic h∇[1, l_bin, feat_i, node_i] += s1
        Atomix.@atomic h∇[2, l_bin, feat_i, node_i] += s2
        Atomix.@atomic h∇[3, l_bin, feat_i, node_i] += s3
    end
end

@kernel function hist_kernel_bins_tiled_depth1!(
    h∇::AbstractArray{T,4},
    @Const(∇),
    @Const(x_bin),
    @Const(js),
    @Const(is),
) where {T}
    g_feat = @index(Group)
    l_bin = @index(Local)

    @inbounds if g_feat <= length(js) && l_bin <= 64
        node_i = Int32(1)
        feat_i = Int(js[g_feat])
        s1 = T(0); s2 = T(0); s3 = T(0)
        i = l_bin
        @inbounds while i <= length(is)
            obs = is[i]
            b = Int(x_bin[obs, feat_i])
            if b == l_bin
                s1 += ∇[1, obs]
                s2 += ∇[2, obs]
                s3 += ∇[3, obs]
            end
            i += 64
        end
        Atomix.@atomic h∇[1, l_bin, feat_i, node_i] += s1
        Atomix.@atomic h∇[2, l_bin, feat_i, node_i] += s2
        Atomix.@atomic h∇[3, l_bin, feat_i, node_i] += s3
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
        # One group per feature, 64 local threads (one per bin)
        hist_bins_d1! = hist_kernel_bins_tiled_depth1!(backend)
        hist_bins_d1!(h∇, ∇, x_bin, js, is; ndrange = length(js), workgroupsize = 64)
    else
        # Build histograms for the current active nodes
        zero_nodes! = zero_node_hist_kernel!(backend)
        zero_nodes!(h∇, active_nodes, js; ndrange = (n_active, length(js), size(h∇, 2)))

        # One group per (node, feature), 64 local threads (one per bin)
        hist_bins! = hist_kernel_bins_tiled!(backend)
        hist_bins!(h∇, ∇, x_bin, nidx, js, is, active_nodes; ndrange = (n_active, length(js)), workgroupsize = 64)
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

