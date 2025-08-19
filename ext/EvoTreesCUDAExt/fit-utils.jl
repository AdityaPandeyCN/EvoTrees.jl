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

function update_hist_gpu!(
    h∇, h∇_parent, gains, bins, feats, ∇, x_bin, nidx, js, is, depth, active_nodes, nodes_sum_gpu, params,
    left_nodes_buf, right_nodes_buf, target_mask_buf
)
    backend = KernelAbstractions.get_backend(h∇)
    n_active = length(active_nodes)
    
    if depth == 1
        # Root node - build histogram normally
        h∇ .= 0
        obs_per_thread = Int32(max(32, min(256, div(length(is), 512))))
        n_tiles = Int(ceil(length(is) / obs_per_thread))
        hist_is_tiled! = hist_kernel_is_tiled!(backend)
        hist_is_tiled!(h∇, ∇, x_bin, nidx, js, is, obs_per_thread; ndrange = (n_tiles, length(js)))
        
        # Save root histogram as parent for next level
        copyto!(h∇_parent, h∇)
    else
        # For deeper levels, separate left and right children
        # Create vectors to separate nodes
        left_nodes_vec = Vector{Int32}()
        right_nodes_vec = Vector{Int32}()
        parent_nodes_vec = Vector{Int32}()
        
        # Copy active nodes to CPU to avoid scalar indexing
        active_nodes_cpu = Array(active_nodes)
        
        # Separate nodes into left (even) and right (odd) children
        for i in 1:n_active
            node = active_nodes_cpu[i]
            if node > 0
                if node % 2 == 0  # Left child (even node number: 2*parent)
                    push!(left_nodes_vec, node)
                else  # Right child (odd node number: 2*parent + 1)
                    push!(right_nodes_vec, node)
                    push!(parent_nodes_vec, node >> 1)  # Parent of this node
                end
            end
        end
        
        # Convert to GPU arrays using the preallocated buffers
        left_nodes_gpu = if length(left_nodes_vec) > 0
            copyto!(view(left_nodes_buf, 1:length(left_nodes_vec)), left_nodes_vec)
            view(left_nodes_buf, 1:length(left_nodes_vec))
        else
            view(left_nodes_buf, 1:0)  # Empty view
        end
        
        right_nodes_gpu = if length(right_nodes_vec) > 0
            copyto!(view(right_nodes_buf, 1:length(right_nodes_vec)), right_nodes_vec)
            view(right_nodes_buf, 1:length(right_nodes_vec))
        else
            view(right_nodes_buf, 1:0)  # Empty view
        end
        
        # For parent nodes, we'll use part of the target_mask_buf temporarily as storage
        parent_nodes_gpu = if length(parent_nodes_vec) > 0
            # Reinterpret part of target_mask_buf as Int32 for parent storage
            parent_buf_reinterpreted = reinterpret(Int32, view(target_mask_buf, 1:(length(parent_nodes_vec)*4)))
            copyto!(view(parent_buf_reinterpreted, 1:length(parent_nodes_vec)), parent_nodes_vec)
            view(parent_buf_reinterpreted, 1:length(parent_nodes_vec))
        else
            reinterpret(Int32, view(target_mask_buf, 1:0))  # Empty view
        end
        
        # Build histograms ONLY for left children
        if length(left_nodes_vec) > 0
            # Zero only left node histograms
            zero_nodes! = zero_node_hist_kernel!(backend)
            zero_nodes!(h∇, left_nodes_gpu, js; ndrange = (length(left_nodes_vec), length(js), size(h∇, 2)))
            
            # Build mask for left nodes - use second half of target_mask_buf
            mask_offset = length(parent_nodes_vec) * 4  # Account for parent nodes storage
            mask_view = view(target_mask_buf, mask_offset+1:length(target_mask_buf))
            mask_view .= 0
            fill_mask! = fill_mask_kernel!(backend)
            fill_mask!(mask_view, left_nodes_gpu; ndrange = length(left_nodes_vec))
            
            # Build histograms for left children only
            obs_per_thread = Int32(max(32, min(256, div(length(is), 1024))))
            n_tiles = Int(ceil(length(is) / obs_per_thread))
            hist_selective_mask_is_tiled! = hist_kernel_selective_mask_is_tiled!(backend)
            hist_selective_mask_is_tiled!(h∇, ∇, x_bin, nidx, js, mask_view, is, obs_per_thread; 
                                          ndrange = (n_tiles, length(js)))
        end
        
        # Compute right children by subtraction
        if length(right_nodes_vec) > 0
            subtract_hist! = histogram_subtraction_kernel!(backend)
            subtract_hist!(h∇, h∇_parent, left_nodes_gpu, right_nodes_gpu, parent_nodes_gpu, js, nodes_sum_gpu; 
                          ndrange = (length(right_nodes_vec), length(js)))
        end
        
        # Save current histograms as parent for next level
        copyto!(h∇_parent, h∇)
    end
    
    # Find best splits (unchanged)
    find_split! = find_best_split_from_hist_kernel!(backend)
    find_split!(
        gains, bins, feats, h∇, nodes_sum_gpu, active_nodes, js,
        eltype(gains)(params.lambda), eltype(gains)(params.min_weight);
        ndrange = n_active
    )
    
    KernelAbstractions.synchronize(backend)
    return nothing
end

