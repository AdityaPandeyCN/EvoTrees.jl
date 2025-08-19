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

# NEW: Segmented histogram kernel - eliminates atomics
@kernel function hist_kernel_segmented!(
    h∇_segments::AbstractArray{T,5}, # [grad_dim, bin, feat, node, segment]
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
    @Const(is),
) where {T}
    i, j, seg_id = @index(Global, NTuple)
    @inbounds if i <= length(is) && j <= length(js) && seg_id <= size(h∇_segments, 5)
        obs = is[i]
        node = nidx[obs]
        if node > 0
            feat = js[j]
            bin = x_bin[obs, feat]
            if bin > 0 && bin <= size(h∇_segments, 2)
                # No atomics - each thread writes to its own segment
                h∇_segments[1, bin, feat, node, seg_id] = ∇[1, obs]
                h∇_segments[2, bin, feat, node, seg_id] = ∇[2, obs] 
                h∇_segments[3, bin, feat, node, seg_id] = ∇[3, obs]
            end
        end
    end
end

# NEW: Selective segmented histogram kernel 
@kernel function hist_kernel_selective_segmented!(
    h∇_segments::AbstractArray{T,5},
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
    @Const(is),
    @Const(target_nodes),
) where {T}
    i, j, seg_id = @index(Global, NTuple)
    @inbounds if i <= length(is) && j <= length(js) && seg_id <= size(h∇_segments, 5)
        obs = is[i]
        node = nidx[obs]
        if node > 0 && target_nodes[node] != 0
            feat = js[j]
            bin = x_bin[obs, feat]
            if bin > 0 && bin <= size(h∇_segments, 2)
                h∇_segments[1, bin, feat, node, seg_id] = ∇[1, obs]
                h∇_segments[2, bin, feat, node, seg_id] = ∇[2, obs]
                h∇_segments[3, bin, feat, node, seg_id] = ∇[3, obs]
            end
        end
    end
end

# NEW: Reduction kernel to sum segments
@kernel function reduce_hist_segments_kernel!(
    h∇::AbstractArray{T,4},
    @Const(h∇_segments),
) where {T}
    grad_dim, bin, feat, node = @index(Global, NTuple)
    @inbounds if grad_dim <= size(h∇, 1) && bin <= size(h∇, 2) && feat <= size(h∇, 3) && node <= size(h∇, 4)
        total = zero(T)
        for seg in 1:size(h∇_segments, 5)
            total += h∇_segments[grad_dim, bin, feat, node, seg]
        end
        h∇[grad_dim, bin, feat, node] = total
    end
end

# NEW: Count observations per node for smarter subtraction
@kernel function count_node_obs_kernel!(
    node_counts::AbstractVector{Int32},
    @Const(nidx),
    @Const(is),
)
    i = @index(Global)
    @inbounds if i <= length(is)
        obs = is[i]
        node = nidx[obs]
        if node > 0
            Atomix.@atomic node_counts[node] += Int32(1)
        end
    end
end

# NEW: Smart node separation based on observation counts
@kernel function separate_nodes_smart_kernel!(
    smaller_buf::AbstractVector{Int32},
    larger_buf::AbstractVector{Int32},
    smaller_count::AbstractVector{Int32},
    larger_count::AbstractVector{Int32},
    @Const(active_nodes),
    @Const(node_counts),
)
    idx = @index(Global)
    @inbounds if idx <= length(active_nodes)
        node = active_nodes[idx]
        if node > 0
            left_child = node << 1
            right_child = left_child + 1
            
            if left_child <= length(node_counts) && right_child <= length(node_counts)
                left_obs = node_counts[left_child]
                right_obs = node_counts[right_child]
                
                # Put smaller child in smaller_buf for explicit computation
                if left_obs <= right_obs
                    pos = Atomix.@atomic smaller_count[1] += Int32(1)
                    if pos <= length(smaller_buf)
                        smaller_buf[pos] = left_child
                    end
                    pos = Atomix.@atomic larger_count[1] += Int32(1)
                    if pos <= length(larger_buf)
                        larger_buf[pos] = right_child
                    end
                else
                    pos = Atomix.@atomic smaller_count[1] += Int32(1)
                    if pos <= length(smaller_buf)
                        smaller_buf[pos] = right_child
                    end
                    pos = Atomix.@atomic larger_count[1] += Int32(1)
                    if pos <= length(larger_buf)
                        larger_buf[pos] = left_child
                    end
                end
            end
        end
    end
end

@kernel function histogram_subtraction_kernel!(
    h∇::AbstractArray{T,4},
    @Const(h∇_parent),
    @Const(smaller_nodes),
    @Const(larger_nodes),
    @Const(js),
) where {T}
    idx, j_idx, bin = @index(Global, NTuple)
    
    @inbounds if idx <= length(larger_nodes) && j_idx <= length(js) && bin <= size(h∇, 2)
        larger_node = larger_nodes[idx]
        smaller_node = smaller_nodes[idx]
        parent_node = larger_node >> 1  # Parent of larger node
        feat = js[j_idx]
        
        # larger = parent - smaller
        h∇[1, bin, feat, larger_node] = h∇_parent[1, bin, feat, parent_node] - h∇[1, bin, feat, smaller_node]
        h∇[2, bin, feat, larger_node] = h∇_parent[2, bin, feat, parent_node] - h∇[2, bin, feat, smaller_node]
        h∇[3, bin, feat, larger_node] = h∇_parent[3, bin, feat, parent_node] - h∇[3, bin, feat, smaller_node]
    end
end

# OPTIMIZED: Main function with better subtraction trick
function update_hist_gpu!(
    h∇, h∇_parent, gains, bins, feats, ∇, x_bin, nidx, js, is, depth, active_nodes, nodes_sum_gpu, params,
    left_nodes_buf, right_nodes_buf, target_mask_buf
)
    backend = KernelAbstractions.get_backend(h∇)
    n_active = length(active_nodes)
    
    # Adaptive segmentation: more segments for problematic cases (high depth, small datasets)
    n_obs = length(is)
    if depth >= 8 && n_obs <= 1_000_000  # Target the bottleneck cases
        n_segments = min(64, max(16, n_obs ÷ 2000 + 1))
    elseif n_obs <= 100_000
        n_segments = min(32, max(8, n_obs ÷ 5000 + 1))
    else
        n_segments = min(16, max(4, n_obs ÷ 25000 + 1))
    end
    
    if depth == 1
        # Root node: build histogram using segmented approach
        h∇ .= 0
        h∇_segments = similar(h∇, size(h∇)..., n_segments)
        h∇_segments .= 0
        
        # Build histogram segments (no atomics)
        hist_seg! = hist_kernel_segmented!(backend)
        hist_seg!(h∇_segments, ∇, x_bin, nidx, js, is; 
                 ndrange = (length(is), length(js), n_segments))
        
        # Reduce segments
        reduce! = reduce_hist_segments_kernel!(backend)
        reduce!(h∇, h∇_segments; 
               ndrange = (size(h∇, 1), size(h∇, 2), size(h∇, 3), size(h∇, 4)))
        
        copyto!(h∇_parent, h∇)
        
    elseif n_active > 0
        copyto!(h∇_parent, h∇)
        
        # Count observations per node for smart subtraction
        node_counts = KernelAbstractions.zeros(backend, Int32, size(h∇, 4))
        count_obs! = count_node_obs_kernel!(backend)
        count_obs!(node_counts, nidx, is; ndrange = length(is))
        KernelAbstractions.synchronize(backend)
        
        # Smart separation: smaller child gets explicit computation, larger gets subtraction
        smaller_count = KernelAbstractions.zeros(backend, Int32, 1)
        larger_count = KernelAbstractions.zeros(backend, Int32, 1)
        
        separate_smart! = separate_nodes_smart_kernel!(backend)
        separate_smart!(left_nodes_buf, right_nodes_buf, smaller_count, larger_count, 
                       active_nodes, node_counts; ndrange = n_active)
        KernelAbstractions.synchronize(backend)
        
        n_smaller = Int(Array(smaller_count)[1])
        n_larger = Int(Array(larger_count)[1])
        
        if n_smaller > 0
            # Compute histogram for smaller children only
            smaller_nodes = view(left_nodes_buf, 1:n_smaller)
            
            # Zero smaller children
            zero_nodes! = zero_node_hist_kernel!(backend)
            zero_nodes!(h∇, smaller_nodes, js; ndrange = (n_smaller, length(js), size(h∇, 2)))
            
            # Set target mask for smaller children
            target_mask_buf .= 0
            fill_mask! = fill_mask_kernel!(backend)
            fill_mask!(target_mask_buf, smaller_nodes; ndrange = n_smaller)
            
            # Build histogram using segmented approach for smaller children only
            h∇_segments = similar(h∇, size(h∇)..., n_segments)
            h∇_segments .= 0
            
            hist_seg_selective! = hist_kernel_selective_segmented!(backend)
            hist_seg_selective!(h∇_segments, ∇, x_bin, nidx, js, is, target_mask_buf; 
                               ndrange = (length(is), length(js), n_segments))
            
            # Reduce segments for smaller children
            reduce_selective! = reduce_hist_segments_kernel!(backend)
            reduce_selective!(h∇, h∇_segments; 
                            ndrange = (size(h∇, 1), size(h∇, 2), size(h∇, 3), size(h∇, 4)))
        end
        
        if n_larger > 0
            # Compute larger children by subtraction (much faster)
            larger_nodes = view(right_nodes_buf, 1:n_larger)
            smaller_nodes = view(left_nodes_buf, 1:n_smaller)
            
            subtract_hist! = histogram_subtraction_kernel!(backend)
            subtract_hist!(h∇, h∇_parent, smaller_nodes, larger_nodes, js; 
                          ndrange = (n_larger, length(js), size(h∇, 2)))
        end
    end
    
    # Find best splits
    find_split! = find_best_split_from_hist_kernel!(backend)
    find_split!(gains, bins, feats, h∇, nodes_sum_gpu, active_nodes, js,
                eltype(gains)(params.lambda), eltype(gains)(params.min_weight);
                ndrange = n_active)
    
    KernelAbstractions.synchronize(backend)
end

