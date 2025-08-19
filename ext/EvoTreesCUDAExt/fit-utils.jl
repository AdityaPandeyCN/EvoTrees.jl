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
    idx, j_idx, bin_chunk = @index(Global, NTuple)
    @inbounds if idx <= length(nodes) && j_idx <= length(js)
        node = nodes[idx]
        feat = js[j_idx]
        if node > 0
            nbins = size(h∇, 2)
            # Each thread handles 4 bins
            @inbounds for b_offset in 0:3
                bin = (bin_chunk - 1) * 4 + b_offset + 1
                if bin <= nbins
                    h∇[1, bin, feat, node] = zero(T)
                    h∇[2, bin, feat, node] = zero(T)
                    h∇[3, bin, feat, node] = zero(T)
                end
            end
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

@kernel function separate_nodes_kernel!(
    left_buf::AbstractVector{Int32},
    right_buf::AbstractVector{Int32},
    parent_buf::AbstractVector{Int32},
    left_count::AbstractVector{Int32},
    right_count::AbstractVector{Int32},
    @Const(active_nodes)
)
    idx = @index(Global)
    @inbounds if idx <= length(active_nodes)
        node = active_nodes[idx]
        if node > 0
            if node % 2 == 0  # Left child
                pos = Atomix.@atomic left_count[1] += Int32(1)
                if pos <= length(left_buf)
                    left_buf[pos] = node
                end
            else  # Right child  
                pos = Atomix.@atomic right_count[1] += Int32(1)
                if pos <= length(right_buf) && pos <= length(parent_buf)
                    right_buf[pos] = node
                    parent_buf[pos] = node >> 1
                end
            end
        end
    end
end

@kernel function histogram_subtraction_kernel!(
    h∇::AbstractArray{T,4},
    @Const(h∇_parent),
    @Const(left_nodes),
    @Const(right_nodes), 
    @Const(parent_nodes),
    @Const(js),
    nodes_sum
) where {T}
    r_idx, j_idx = @index(Global, NTuple)
    
    @inbounds if r_idx <= length(right_nodes) && j_idx <= length(js)
        right_node = right_nodes[r_idx]
        parent_node = parent_nodes[r_idx]
        left_node = right_node - 1  # Left sibling is always right_node - 1 (since right=2k+1, left=2k)
        
        feat = js[j_idx]
        nbins = size(h∇, 2)
        
        # Bounds checking for array access
        if right_node <= size(h∇, 4) && parent_node <= size(h∇_parent, 4) && 
           left_node <= size(h∇, 4) && feat <= size(h∇, 3) && parent_node <= size(nodes_sum, 2)
            
            # If first feature, also compute parent sum (only once)
            if j_idx == 1
                p_sum1 = zero(T)
                p_sum2 = zero(T)
                p_sum3 = zero(T)
                @inbounds for bin in 1:nbins
                    p_sum1 += h∇_parent[1, bin, feat, parent_node]
                    p_sum2 += h∇_parent[2, bin, feat, parent_node]
                    p_sum3 += h∇_parent[3, bin, feat, parent_node]
                end
                nodes_sum[1, parent_node] = p_sum1
                nodes_sum[2, parent_node] = p_sum2
                nodes_sum[3, parent_node] = p_sum3
            end
            
            # Subtract: right = parent - left
            @inbounds for bin in 1:nbins
                h∇[1, bin, feat, right_node] = h∇_parent[1, bin, feat, parent_node] - h∇[1, bin, feat, left_node]
                h∇[2, bin, feat, right_node] = h∇_parent[2, bin, feat, parent_node] - h∇[2, bin, feat, left_node]
                h∇[3, bin, feat, right_node] = h∇_parent[3, bin, feat, parent_node] - h∇[3, bin, feat, left_node]
            end
        end
    end
end

function update_hist_gpu!(
    h∇, h∇_parent, gains, bins, feats, ∇, x_bin, nidx, js, is, depth, active_nodes, nodes_sum_gpu, params,
    left_nodes_buf, right_nodes_buf, target_mask_buf
)
    backend = KernelAbstractions.get_backend(h∇)
    n_active = length(active_nodes)
    
    if depth == 1
        # Root node - build histogram normally
        h∇ .= 0
        obs_per_thread = Int32(max(64, min(256, div(length(is), 1024))))
        n_tiles = Int(ceil(length(is) / obs_per_thread))
        hist_is_tiled! = hist_kernel_is_tiled!(backend)
        hist_is_tiled!(h∇, ∇, x_bin, nidx, js, is, obs_per_thread; ndrange = (n_tiles, length(js)))
        copyto!(h∇_parent, h∇)
    else
        # Use GPU kernel to separate nodes - NO CPU ALLOCATION!
        left_count = KernelAbstractions.zeros(backend, Int32, 1)
        right_count = KernelAbstractions.zeros(backend, Int32, 1)
        
        # Use second half of left_nodes_buf for parent storage to avoid conflicts
        if length(left_nodes_buf) < 2
            error("left_nodes_buf too small for parent storage")
        end
        max_parent_nodes = min(n_active, length(left_nodes_buf) ÷ 2)
        parent_buf_start = length(left_nodes_buf) ÷ 2 + 1
        parent_buf_end = min(parent_buf_start + max_parent_nodes - 1, length(left_nodes_buf))
        parent_buf = view(left_nodes_buf, parent_buf_start:parent_buf_end)
        
        # Separate nodes entirely on GPU
        separate_nodes! = separate_nodes_kernel!(backend)
        separate_nodes!(view(left_nodes_buf, 1:length(left_nodes_buf)÷2), right_nodes_buf, parent_buf, 
                       left_count, right_count, active_nodes; ndrange = n_active)
        KernelAbstractions.synchronize(backend)
        
        # Get counts (single scalar reads)
        n_left = Int(Array(left_count)[1])
        n_right = Int(Array(right_count)[1])
        
        # Ensure counts don't exceed buffer sizes
        n_left = min(n_left, length(left_nodes_buf) ÷ 2)
        n_right = min(n_right, length(right_nodes_buf), length(parent_buf))
        
        left_nodes_gpu = view(left_nodes_buf, 1:n_left)
        right_nodes_gpu = view(right_nodes_buf, 1:n_right)
        parent_nodes_gpu = view(parent_buf, 1:n_right)
        
        # Build histograms ONLY for left children
        if n_left > 0
            # Zero with fewer threads - 16 instead of 64
            zero_nodes! = zero_node_hist_kernel!(backend)
            zero_nodes!(h∇, left_nodes_gpu, js; ndrange = (n_left, length(js), 16))
            
            # Clear and set mask
            target_mask_buf .= 0
            fill_mask! = fill_mask_kernel!(backend)
            if n_left > 0
                fill_mask!(target_mask_buf, left_nodes_gpu; ndrange = n_left)
            end
            
            # Better tile size calculation
            obs_per_thread = Int32(max(64, min(256, div(length(is), max(512, 1024 >> (depth-2))))))
            n_tiles = Int(ceil(length(is) / obs_per_thread))
            
            hist_selective_mask_is_tiled! = hist_kernel_selective_mask_is_tiled!(backend)
            hist_selective_mask_is_tiled!(h∇, ∇, x_bin, nidx, js, target_mask_buf, is, obs_per_thread; 
                                          ndrange = (n_tiles, length(js)))
        end
        
        # Compute right children by subtraction
        if n_right > 0
            subtract_hist! = histogram_subtraction_kernel!(backend)
            subtract_hist!(h∇, h∇_parent, left_nodes_gpu, right_nodes_gpu, parent_nodes_gpu, js, nodes_sum_gpu; 
                          ndrange = (n_right, length(js)))
        end
        
        copyto!(h∇_parent, h∇)
    end
    
    # Find best splits
    find_split! = find_best_split_from_hist_kernel!(backend)
    find_split!(gains, bins, feats, h∇, nodes_sum_gpu, active_nodes, js,
                eltype(gains)(params.lambda), eltype(gains)(params.min_weight);
                ndrange = n_active)
    
    KernelAbstractions.synchronize(backend)
    return nothing
end

