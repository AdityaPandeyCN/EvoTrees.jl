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

@kernel function zero_node_hist_kernel_optimized!(h∇::AbstractArray{T,4}, @Const(nodes), @Const(js)) where {T}
    idx, j_idx, bin_chunk = @index(Global, NTuple)
    @inbounds if idx <= length(nodes) && j_idx <= length(js)
        node = nodes[idx]
        feat = js[j_idx]
        if node > 0
            nbins = size(h∇, 2)
            base_bin = (bin_chunk - 1) << 3  # Process 8 bins per thread instead of 4
            # Each thread handles 8 bins for better throughput
            @inbounds for b_offset in 1:8
                bin = base_bin + b_offset
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

@kernel function hist_kernel_is_tiled_optimized!(
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
            
            # Process multiple observations per thread for better memory efficiency
            @inbounds for i in start_idx:end_idx
                obs = is[i]
                node = nidx[obs]
                if node > 0
                    bin = x_bin[obs, jdx]
                    if bin > 0 && bin <= size(h∇, 2)
                        # Reduced atomic contention by grouping operations
                        g1, g2, g3 = ∇[1, obs], ∇[2, obs], ∇[3, obs]
                        Atomix.@atomic h∇[1, bin, jdx, node] += g1
                        Atomix.@atomic h∇[2, bin, jdx, node] += g2
                        Atomix.@atomic h∇[3, bin, jdx, node] += g3
                    end
                end
            end
        end
    end
end

@kernel function hist_kernel_selective_mask_is_tiled_optimized!(
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
            
            # Optimized loop with better memory access
            @inbounds for i in start_idx:end_idx
                obs = is[i]
                node = nidx[obs]
                # Early exit if node not in target mask
                if node > 0 && target_mask[node] != 0
                    bin = x_bin[obs, jdx]
                    if bin > 0 && bin <= size(h∇, 2)
                        # Cache gradient values to reduce memory reads
                        g1, g2, g3 = ∇[1, obs], ∇[2, obs], ∇[3, obs]
                        Atomix.@atomic h∇[1, bin, jdx, node] += g1
                        Atomix.@atomic h∇[2, bin, jdx, node] += g2
                        Atomix.@atomic h∇[3, bin, jdx, node] += g3
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
            if node & 1 == 0  # Left child (faster than % 2)
                pos = Atomix.@atomic left_count[1] += Int32(1)
                if pos <= length(left_buf)
                    left_buf[pos] = node
                end
            else  # Right child  
                pos = Atomix.@atomic right_count[1] += Int32(1)
                if pos <= length(right_buf)
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
        left_node = right_node - 1  # Left sibling is always right_node - 1
        
        feat = js[j_idx]
        nbins = size(h∇, 2)
        
        # If first feature, compute parent sum (only once)
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

@kernel function histogram_subtraction_kernel_optimized!(
    h∇::AbstractArray{T,4},
    @Const(h∇_parent),
    @Const(left_nodes),
    @Const(right_nodes), 
    @Const(parent_nodes),
    @Const(js),
    nodes_sum
) where {T}
    r_idx, j_idx, bin_chunk = @index(Global, NTuple)
    
    @inbounds if r_idx <= length(right_nodes) && j_idx <= length(js)
        right_node = right_nodes[r_idx]
        parent_node = parent_nodes[r_idx]
        left_node = right_node - 1  # Left sibling is always right_node - 1
        
        feat = js[j_idx]
        nbins = size(h∇, 2)
        
        # If first feature and first bin chunk, compute parent sum (only once)
        if j_idx == 1 && bin_chunk == 1
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
        
        # Process 4 bins per thread for better throughput
        base_bin = (bin_chunk - 1) << 2  # * 4
        @inbounds for b_offset in 1:4
            bin = base_bin + b_offset
            if bin <= nbins
                # Vectorized subtraction: right = parent - left
                h∇[1, bin, feat, right_node] = h∇_parent[1, bin, feat, parent_node] - h∇[1, bin, feat, left_node]
                h∇[2, bin, feat, right_node] = h∇_parent[2, bin, feat, parent_node] - h∇[2, bin, feat, left_node]
                h∇[3, bin, feat, right_node] = h∇_parent[3, bin, feat, parent_node] - h∇[3, bin, feat, left_node]
            end
        end
    end
end

function ensure_parent_hist!(h∇_parent_ref, h∇, backend)
    if h∇_parent_ref[] === nothing
        h∇_parent_ref[] = similar(h∇)
    end
    return h∇_parent_ref[]
end

function update_hist_gpu!(
    h∇, h∇_parent_ref, gains, bins, feats, ∇, x_bin, nidx, js, is, depth, active_nodes, nodes_sum_gpu, params,
    left_nodes_buf, right_nodes_buf, target_mask_buf
)
    backend = KernelAbstractions.get_backend(h∇)
    n_active = length(active_nodes)
    
    if depth == 1
        # Root node - build histogram normally
        h∇ .= 0
        obs_per_thread = Int32(max(128, min(512, div(length(is), 512))))  # Increased tile size
        n_tiles = Int(ceil(length(is) / obs_per_thread))
        hist_is_tiled! = hist_kernel_is_tiled_optimized!(backend)
        hist_is_tiled!(h∇, ∇, x_bin, nidx, js, is, obs_per_thread; ndrange = (n_tiles, length(js)))
        # REMOVED: copyto!(h∇_parent, h∇) - expensive and unnecessary for depth 1
    else
        # Use GPU kernel to separate nodes - NO CPU ALLOCATION!
        left_count = KernelAbstractions.zeros(backend, Int32, 1)
        right_count = KernelAbstractions.zeros(backend, Int32, 1)
        
        # Use second half of left_nodes_buf for parent storage
        half_buf_size = length(left_nodes_buf) >> 1
        parent_buf = view(left_nodes_buf, half_buf_size+1:length(left_nodes_buf))
        
        # Separate nodes entirely on GPU
        separate_nodes! = separate_nodes_kernel!(backend)
        separate_nodes!(view(left_nodes_buf, 1:half_buf_size), right_nodes_buf, parent_buf, 
                       left_count, right_count, active_nodes; ndrange = n_active)
        KernelAbstractions.synchronize(backend)
        
        # Get counts (single scalar reads)
        n_left = Int(Array(left_count)[1])
        n_right = Int(Array(right_count)[1])
        
        # Ensure counts don't exceed buffer sizes
        n_left = min(n_left, half_buf_size)
        n_right = min(n_right, length(right_nodes_buf))
        
        left_nodes_gpu = view(left_nodes_buf, 1:n_left)
        right_nodes_gpu = view(right_nodes_buf, 1:n_right)
        parent_nodes_gpu = view(parent_buf, 1:n_right)
        
        # Build histograms ONLY for left children - OPTIMIZED
        if n_left > 0
            # Zero with optimal threads - adjusted for 8 bins per thread
            zero_nodes! = zero_node_hist_kernel_optimized!(backend)
            zero_nodes!(h∇, left_nodes_gpu, js; ndrange = (n_left, length(js), 8))  # 8 threads per bin group
            
            # Clear and set mask
            target_mask_buf .= 0
            fill_mask! = fill_mask_kernel!(backend)
            fill_mask!(target_mask_buf, left_nodes_gpu; ndrange = n_left)
            
            # OPTIMIZED tile size - adaptive based on depth
            base_obs_per_thread = max(128, min(512, div(length(is), max(256, 512 >> (depth-2)))))
            obs_per_thread = Int32(base_obs_per_thread)
            n_tiles = Int(ceil(length(is) / obs_per_thread))
            
            hist_selective_mask_is_tiled! = hist_kernel_selective_mask_is_tiled_optimized!(backend)
            hist_selective_mask_is_tiled!(h∇, ∇, x_bin, nidx, js, target_mask_buf, is, obs_per_thread; 
                                          ndrange = (n_tiles, length(js)))
        end
        
        # Smart parent histogram update - only allocate and copy when subtraction is needed
        if n_right > 0
            h∇_parent = ensure_parent_hist!(h∇_parent_ref, h∇, backend)
            copyto!(h∇_parent, h∇)
        end
        
        # Compute right children by subtraction - OPTIMIZED: only when needed
        if n_right > 0
            h∇_parent = h∇_parent_ref[]  # We know it exists now
            subtract_hist! = histogram_subtraction_kernel_optimized!(backend)
            # Calculate bin chunks for 4 bins per thread
            nbins = size(h∇, 2)
            bin_chunks = Int(ceil(nbins / 4))
            subtract_hist!(h∇, h∇_parent, left_nodes_gpu, right_nodes_gpu, parent_nodes_gpu, js, nodes_sum_gpu; 
                          ndrange = (n_right, length(js), bin_chunks))
        end
        
        # REMOVED: copyto!(h∇_parent, h∇) - move this to only when needed
    end
    
    # Find best splits
    find_split! = find_best_split_from_hist_kernel!(backend)
    find_split!(gains, bins, feats, h∇, nodes_sum_gpu, active_nodes, js,
                eltype(gains)(params.lambda), eltype(gains)(params.min_weight);
                ndrange = n_active)
    
    KernelAbstractions.synchronize(backend)
end

