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

# Single histogram kernel for all nodes at current depth
@kernel function hist_kernel_depth!(
    h∇::AbstractArray{T,4},
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
    @Const(is),
    @Const(nodes_at_depth),  # Which nodes we're building histograms for
) where {T}
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(is) && j <= length(js)
        obs = is[i]
        node = nidx[obs]
        
        # Check if this observation's node is one we're computing
        # (This is more efficient than a mask lookup)
        valid = false
        for k in 1:length(nodes_at_depth)
            if node == nodes_at_depth[k]
                valid = true
                break
            end
        end
        
        if valid
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

# Optimized histogram kernel using node range (for contiguous nodes)
@kernel function hist_kernel_range!(
    h∇::AbstractArray{T,4},
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
    @Const(is),
    node_start::Int32,
    node_end::Int32,
    even_only::Int32,
) where {T}
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(is) && j <= length(js)
        obs = is[i]
        node = nidx[obs]
        
        if node >= node_start && node <= node_end
            process = (even_only == 0) || ((node & 1) == 0)
            if process
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
end

@kernel function subtract_hist_kernel!(
    h∇::AbstractArray{T,4},
    parent_start::Int32,
    parent_end::Int32,
    @Const(js),
) where {T}
    parent_idx, f_idx, bin = @index(Global, NTuple)
    
    @inbounds if parent_idx <= (parent_end - parent_start + 1) && f_idx <= length(js) && bin <= size(h∇, 2)
        parent = parent_start + parent_idx - 1
        left = parent << 1
        right = left + 1
        feat = js[f_idx]
        
        # Right = Parent - Left
        h∇[1, bin, feat, right] = h∇[1, bin, feat, parent] - h∇[1, bin, feat, left]
        h∇[2, bin, feat, right] = h∇[2, bin, feat, parent] - h∇[2, bin, feat, left]
        h∇[3, bin, feat, right] = h∇[3, bin, feat, parent] - h∇[3, bin, feat, left]
    end
end

@kernel function scan_hist_kernel!(
    hL::AbstractArray{T,4},
    hR::AbstractArray{T,4},
    @Const(h∇),
    node_start::Int32,
    node_end::Int32,
    @Const(js),
) where {T}
    node_idx, f_idx = @index(Global, NTuple)
    nbins = size(h∇, 2)
    
    @inbounds if node_idx <= (node_end - node_start + 1) && f_idx <= length(js)
        node = node_start + node_idx - 1
        f = js[f_idx]
        
        s1 = zero(T); s2 = zero(T); s3 = zero(T)
        for bin in 1:nbins
            s1 += h∇[1, bin, f, node]
            s2 += h∇[2, bin, f, node]
            s3 += h∇[3, bin, f, node]
            hL[1, bin, f, node] = s1
            hL[2, bin, f, node] = s2
            hL[3, bin, f, node] = s3
        end
        
        hR[1, nbins, f, node] = s1
        hR[2, nbins, f, node] = s2
        hR[3, nbins, f, node] = s3
    end
end

@kernel function find_best_split_kernel!(
    gains::AbstractVector{T},
    bins::AbstractVector{Int32},
    feats::AbstractVector{Int32},
    @Const(hL),
    @Const(hR),
    @Const(nodes_sum),
    node_start::Int32,
    node_end::Int32,
    @Const(js),
    lambda::T,
    min_weight::T,
) where {T}
    node_idx = @index(Global)
    
    @inbounds if node_idx <= (node_end - node_start + 1)
        node = node_start + node_idx - 1
        idx = node_idx  # Output index
        
        nbins = size(hL, 2)
        g_best = T(-Inf)
        b_best = Int32(0)
        f_best = Int32(0)
        
        p_g1 = nodes_sum[1, node]
        p_g2 = nodes_sum[2, node]
        p_w  = nodes_sum[3, node]
        gain_p = p_g1^2 / (p_g2 + lambda * p_w + T(1e-8))
        
        for f_idx in 1:length(js)
            f = js[f_idx]
            f_w = hR[3, nbins, f, node]
            
            if f_w < 2 * min_weight
                continue
            end
            
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
        
        gains[idx] = g_best
        bins[idx] = b_best
        feats[idx] = f_best
    end
end

@kernel function write_nodes_sum_range!(nodes_sum, @Const(hR), node_start::Int32, node_end::Int32, @Const(js))
    idx = @index(Global)
    @inbounds if idx <= (node_end - node_start + 1)
        node = node_start + idx - 1
        nbins = size(hR, 2)
        f = js[1]
        nodes_sum[1, node] = hR[1, nbins, f, node]
        nodes_sum[2, node] = hR[2, nbins, f, node]
        nodes_sum[3, node] = hR[3, nbins, f, node]
    end
end

function update_hist_gpu!(
    h∇, hL, hR, gains, bins, feats, ∇, x_bin, nidx, js, is, depth, active_nodes, nodes_sum_gpu, params,
    left_nodes_buf, right_nodes_buf, target_mask_buf
)
    backend = KernelAbstractions.get_backend(h∇)
    
    profile = get(ENV, "EVO_PROF", "0") == "1"
    t_hist = 0.0
    t_scan = 0.0
    t_find = 0.0
    
    # Nodes at this depth level are contiguous
    # At depth d, nodes range from 2^(d-1) to 2^d - 1
    nodes_at_depth_start = Int32(2^(depth - 1))
    nodes_at_depth_end = Int32(2^depth - 1)
    n_nodes = nodes_at_depth_end - nodes_at_depth_start + 1
    
    if depth == 1
        # Root node - just compute its histogram
        h∇ .= 0
        hist_range! = hist_kernel_range!(backend)
        t_hist += @elapsed begin
                         hist_range!(h∇, ∇, x_bin, nidx, js, is, Int32(1), Int32(1), Int32(0); 
                        ndrange = (length(is), length(js)))
            KernelAbstractions.synchronize(backend)
        end
    else
        # For depth > 1, use histogram subtraction trick
        # Parent nodes are at previous level
        parent_start = Int32(2^(depth - 2))
        parent_end = Int32(2^(depth - 1) - 1)
        
        # Zero out children histograms
        h∇[:, :, :, nodes_at_depth_start:nodes_at_depth_end] .= 0
        
        # Build histograms only for left children (even nodes at current depth)
        hist_range! = hist_kernel_range!(backend)
        t_hist += @elapsed begin
            # Left children are even nodes: 2^(d-1), 2^(d-1)+2, ...
            # We can optimize by only computing for observations that belong to left children
            # For now, compute for all and filter by node
                         hist_range!(h∇, ∇, x_bin, nidx, js, is, 
                        nodes_at_depth_start, nodes_at_depth_end, Int32(1);
                        ndrange = (length(is), length(js))
            KernelAbstractions.synchronize(backend)
            
            # Subtract to get right children
            subtract! = subtract_hist_kernel!(backend)
            n_parents = parent_end - parent_start + 1
            subtract!(h∇, parent_start, parent_end, js; 
                     ndrange = (n_parents, length(js), size(h∇, 2)))
            KernelAbstractions.synchronize(backend)
        end
    end
    
    # Scan for cumulative histograms
    scan! = scan_hist_kernel!(backend)
    t_scan += @elapsed begin
        scan!(hL, hR, h∇, nodes_at_depth_start, nodes_at_depth_end, js; 
              ndrange = (n_nodes, length(js)))
        KernelAbstractions.synchronize(backend)
    end
    
        # Update nodes_sum from first feature on GPU
    write_nodes_sum_rng! = write_nodes_sum_range!(backend)
    write_nodes_sum_rng!(nodes_sum_gpu, hR, nodes_at_depth_start, nodes_at_depth_end, js; ndrange = n_nodes)

    # Find best splits
    find_split! = find_best_split_kernel!(backend)
    t_find += @elapsed begin
        find_split!(gains, bins, feats, hL, hR, nodes_sum_gpu,
                   nodes_at_depth_start, nodes_at_depth_end, js,
                   eltype(gains)(params.lambda), eltype(gains)(params.min_weight);
                   ndrange = n_nodes)
        KernelAbstractions.synchronize(backend)
    end
    
    if profile
        @info "gpu_prof:update_hist" depth=depth n_nodes=n_nodes t_hist=t_hist t_scan=t_scan t_find=t_find
    end
    
    return nothing
end

