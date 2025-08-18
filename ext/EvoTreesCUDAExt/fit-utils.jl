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

@kernel function zero_node_hist_kernel_js!(h∇::AbstractArray{T,4}, @Const(nodes), @Const(js)) where {T}
    idx, j_idx, bin = @index(Global, NTuple)
    @inbounds if idx <= length(nodes) && j_idx <= length(js) && bin <= size(h∇, 2)
        node = nodes[idx]
        if node > 0
            feat = js[j_idx]
            h∇[1, bin, feat, node] = zero(T)
            h∇[2, bin, feat, node] = zero(T)
            h∇[3, bin, feat, node] = zero(T)
        end
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

@kernel function scan_hist_kernel_serial_js!(
    hL::AbstractArray{T,4},
    hR::AbstractArray{T,4},
    @Const(h∇),
    @Const(active_nodes),
    @Const(js),
) where {T}
    n_idx, f_idx = @index(Global, NTuple)
    
    nbins = size(h∇, 2)
    
    @inbounds if n_idx <= length(active_nodes) && f_idx <= length(js)
        node = active_nodes[n_idx]
        if node > 0
            f = js[f_idx]
            s1 = zero(T); s2 = zero(T); s3 = zero(T)
            @inbounds for bin in 1:nbins
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
        p_w  = nodes_sum[3, node]
            gain_p = p_g1^2 / (p_g2 + lambda * p_w + T(1e-8))
        
        for f in 1:nfeats
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

@kernel function find_best_split_kernel_parallel_js!(
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
        
        p_g1 = nodes_sum[1, node]
        p_g2 = nodes_sum[2, node]
        p_w  = nodes_sum[3, node]
            gain_p = p_g1^2 / (p_g2 + lambda * p_w + T(1e-8))
        
        for f_idx in 1:length(js)
            f = js[f_idx]
            f_w = hR[3, nbins, f, node]
            # Early skip if the total weight cannot produce a valid split
            if f_w < min_weight + min_weight
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
        
        gains[n_idx] = g_best
        bins[n_idx] = b_best
        feats[n_idx] = f_best
        end
    end
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

@kernel function filter_is_active_kernel!(is_out, out_len, @Const(is), @Const(nidx), @Const(target_mask))
    i = @index(Global)
    @inbounds if i <= length(is)
        obs = is[i]
        node = nidx[obs]
        if node > 0 && target_mask[node] != 0
            idx = Atomix.@atomic out_len[1] += 1
            is_out[idx] = obs
        end
    end
end

@kernel function write_nodes_sum_from_scan!(nodes_sum, @Const(hR), @Const(active_nodes), @Const(js))
    n_idx = @index(Global)
    @inbounds if n_idx <= length(active_nodes)
        node = active_nodes[n_idx]
        if node > 0
            nbins = size(hR, 2)
            f = js[1]
            nodes_sum[1, node] = hR[1, nbins, f, node]
            nodes_sum[2, node] = hR[2, nbins, f, node]
            nodes_sum[3, node] = hR[3, nbins, f, node]
        end
    end
end

function update_hist_gpu!(
    h∇, hL, hR, gains, bins, feats, ∇, x_bin, nidx, js, is, depth, active_nodes, nodes_sum_gpu, params,
    left_nodes_buf, right_nodes_buf, target_mask_buf
)
    backend = KernelAbstractions.get_backend(h∇)
    
    n_active = length(active_nodes)
    
    profile = get(ENV, "EVO_PROF", "0") == "1"
    t_hist = 0.0
    t_scan = 0.0
    t_write = 0.0
    t_find = 0.0
    
    if depth == 1
        h∇ .= 0
        hist_is! = hist_kernel_is!(backend)
        t_hist += @elapsed begin
            hist_is!(h∇, ∇, x_bin, nidx, js, is; ndrange = (length(is), length(js)), workgroupsize=(256,1))
            KernelAbstractions.synchronize(backend)
        end
    else
        # Build histograms for the current active nodes (parents to split now)
        zero_nodes_js! = zero_node_hist_kernel_js!(backend)
        t_hist += @elapsed begin
            zero_nodes_js!(h∇, active_nodes, js; ndrange = (n_active, length(js), size(h∇, 2)), workgroupsize=(32,4,4))
            target_mask_buf .= 0
            fill_mask! = fill_mask_kernel!(backend)
            fill_mask!(target_mask_buf, active_nodes; ndrange = n_active, workgroupsize=256)

            # compact observations belonging to current active nodes
            is_active = KernelAbstractions.zeros(backend, eltype(is), length(is))
            is_active_len = KernelAbstractions.zeros(backend, Int32, 1)
            filter_is_act! = filter_is_active_kernel!(backend)
            filter_is_act!(is_active, is_active_len, is, nidx, target_mask_buf; ndrange = length(is), workgroupsize=256)
            KernelAbstractions.synchronize(backend)
            n_is_act = Int(Array(is_active_len)[1])

            hist_selective_mask_is! = hist_kernel_selective_mask_is!(backend)
            hist_selective_mask_is!(h∇, ∇, x_bin, nidx, js, target_mask_buf, is_active;
                                            ndrange = (n_is_act, length(js)), workgroupsize=(256,1))
            KernelAbstractions.synchronize(backend)
        end
    end

    scan_serial_js! = scan_hist_kernel_serial_js!(backend)
    t_scan += @elapsed begin
        scan_serial_js!(hL, hR, h∇, active_nodes, js; ndrange = (n_active, length(js)), workgroupsize=(64,4))
        KernelAbstractions.synchronize(backend)
    end

    write_nodes_sum! = write_nodes_sum_from_scan!(backend)
    t_write += @elapsed begin
        write_nodes_sum!(nodes_sum_gpu, hR, active_nodes, js; ndrange = n_active, workgroupsize=256)
        KernelAbstractions.synchronize(backend)
    end

    find_split_js! = find_best_split_kernel_parallel_js!(backend)
    t_find += @elapsed begin
        find_split_js!(
            gains, bins, feats, hL, hR, nodes_sum_gpu, active_nodes, js,
            eltype(gains)(params.lambda), eltype(gains)(params.min_weight);
            ndrange = n_active, workgroupsize=256
        )
        KernelAbstractions.synchronize(backend)
    end
    
    if profile
        @info "gpu_prof:update_hist" depth=depth n_active=n_active t_hist=t_hist t_scan=t_scan t_write=t_write t_find=t_find
    end
    
    return nothing
end

