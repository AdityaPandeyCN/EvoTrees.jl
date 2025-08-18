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

# removed range-based and subtraction kernels; active-node kernels are used
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

@kernel function zero_node_hist_kernel_js_list!(h∇::AbstractArray{T,4}, @Const(active_nodes), @Const(js)) where {T}
    idx, j_idx, bin = @index(Global, NTuple)
    @inbounds if idx <= length(active_nodes) && j_idx <= length(js) && bin <= size(h∇, 2)
        node = active_nodes[idx]
        if node > 0
            feat = js[j_idx]
            h∇[1, bin, feat, node] = zero(T)
            h∇[2, bin, feat, node] = zero(T)
            h∇[3, bin, feat, node] = zero(T)
        end
    end
end

@kernel function fill_mask_kernel_list!(mask::AbstractVector{UInt8}, @Const(active_nodes))
    i = @index(Global)
    @inbounds if i <= length(active_nodes)
        node = active_nodes[i]
        if node > 0 && node <= length(mask)
            mask[node] = UInt8(1)
        end
    end
end

@kernel function hist_kernel_selective_mask_is!(
    h∇::AbstractArray{T,4},
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
    @Const(is),
    @Const(target_mask),
) where {T}
    i, j = @index(Global, NTuple)
    @inbounds if i <= length(is) && j <= length(js)
        obs = is[i]
        node = nidx[obs]
        if node > 0 && node <= length(target_mask) && target_mask[node] != 0
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

@kernel function scan_hist_kernel_active!(
    hL::AbstractArray{T,4},
    hR::AbstractArray{T,4},
    @Const(h∇),
    @Const(active_nodes),
    @Const(js),
) where {T}
    idx, f_idx = @index(Global, NTuple)
    nbins = size(h∇, 2)
    @inbounds if idx <= length(active_nodes) && f_idx <= length(js)
        node = active_nodes[idx]
        if node > 0
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
end

@kernel function write_nodes_sum_from_scan!(nodes_sum, @Const(hR), @Const(active_nodes), @Const(js))
    i = @index(Global)
    @inbounds if i <= length(active_nodes)
        node = active_nodes[i]
        if node > 0
            nbins = size(hR, 2)
            f = js[1]
            nodes_sum[1, node] = hR[1, nbins, f, node]
            nodes_sum[2, node] = hR[2, nbins, f, node]
            nodes_sum[3, node] = hR[3, nbins, f, node]
        end
    end
end

@kernel function find_best_split_kernel_active!(
    gains::AbstractVector{T},
    bins::AbstractVector{Int32},
    feats::AbstractVector{Int32},
    @Const(hL),
    @Const(hR),
    @Const(nodes_sum),
    @Const(active_nodes),
    @Const(js),
    lambda::T,
    min_weight::T,
) where {T}
    idx = @index(Global)
    @inbounds if idx <= length(active_nodes)
        node = active_nodes[idx]
        if node == 0
            gains[idx] = T(-Inf)
            bins[idx] = Int32(0)
            feats[idx] = Int32(0)
        else
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
 
    n_active = length(active_nodes)
 
    if depth == 1
        # Root node - just compute its histogram
        h∇ .= 0
        zero_list! = zero_node_hist_kernel_js_list!(backend)
        zero_list!(h∇, active_nodes, js; ndrange = (n_active, length(js), size(h∇, 2)))
        fill_mask_list! = fill_mask_kernel_list!(backend)
        target_mask_buf .= 0
        fill_mask_list!(target_mask_buf, active_nodes; ndrange = n_active)
        hist_sel_mask! = hist_kernel_selective_mask_is!(backend)
        t_hist += @elapsed begin
            hist_sel_mask!(h∇, ∇, x_bin, nidx, js, is, target_mask_buf; 
                           ndrange = (length(is), length(js)))
            KernelAbstractions.synchronize(backend)
        end
    else
        # For depth > 1, build parent hist for active nodes
        zero_list! = zero_node_hist_kernel_js_list!(backend)
        zero_list!(h∇, active_nodes, js; ndrange = (n_active, length(js), size(h∇, 2)))
        fill_mask_list! = fill_mask_kernel_list!(backend)
        target_mask_buf .= 0
        fill_mask_list!(target_mask_buf, active_nodes; ndrange = n_active)
        hist_sel_mask! = hist_kernel_selective_mask_is!(backend)
        t_hist += @elapsed begin
            hist_sel_mask!(h∇, ∇, x_bin, nidx, js, is, target_mask_buf; 
                           ndrange = (length(is), length(js)))
            KernelAbstractions.synchronize(backend)
        end
    end
    
    # Scan for cumulative histograms
    scan_act! = scan_hist_kernel_active!(backend)
    t_scan += @elapsed begin
        scan_act!(hL, hR, h∇, active_nodes, js; 
                  ndrange = (n_active, length(js)))
        KernelAbstractions.synchronize(backend)
    end
    
    # Update nodes_sum from first feature on GPU
    write_nodes_sum_act! = write_nodes_sum_from_scan!(backend)
    write_nodes_sum_act!(nodes_sum_gpu, hR, active_nodes, js; ndrange = n_active)

    # Find best splits
    find_split_act! = find_best_split_kernel_active!(backend)
    t_find += @elapsed begin
        find_split_act!(gains, bins, feats, hL, hR, nodes_sum_gpu,
                        active_nodes, js, eltype(gains)(params.lambda), eltype(gains)(params.min_weight);
                        ndrange = n_active)
        KernelAbstractions.synchronize(backend)
    end
    
    if profile
        @info "gpu_prof:update_hist" depth=depth n_active=n_active t_hist=t_hist t_scan=t_scan t_find=t_find
    end
    
    return nothing
end

