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

@kernel function hist_kernel!(
    h∇::AbstractArray{T,4},
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
    @Const(is),
) where {T}
    gidx = @index(Global, Linear)
    
    n_feats = length(js)
    n_obs = length(is)
    n_grads = size(∇, 1)  # Support variable gradient dimensions
    total_work_items = n_feats * cld(n_obs, 8)
    
    if gidx <= total_work_items
        feat_idx = (gidx - 1) % n_feats + 1
        obs_chunk = (gidx - 1) ÷ n_feats
        
        feat = js[feat_idx]
        
        start_idx = obs_chunk * 8 + 1
        end_idx = min(start_idx + 7, n_obs)
        
        @inbounds for obs_idx in start_idx:end_idx
            obs = is[obs_idx]
            node = nidx[obs]
            if node > 0 && node <= size(h∇, 4)
                bin = x_bin[obs, feat]
                if bin > 0 && bin <= size(h∇, 2)
                    # Support variable gradient dimensions instead of hard-coded 3
                    for k in 1:n_grads
                        grad_val = ∇[k, obs]
                        Atomix.@atomic h∇[k, bin, feat, node] += grad_val
                    end
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
    @Const(feattypes),
    @Const(monotone_constraints),
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
            return
        end

        nbins = size(h∇, 2)
        n_grads = size(h∇, 1)

        # Accumulate parent stats directly into the nodes_sum buffer to avoid allocation
        for k in 1:n_grads
            nodes_sum[k, node] = zero(T)
        end
        for j_idx in 1:length(js)
            f = js[j_idx]
            for b in 1:nbins
                for k in 1:n_grads
                    nodes_sum[k, node] += h∇[k, b, f, node]
                end
            end
        end

        p_g1 = nodes_sum[1, node]
        p_g2 = nodes_sum[2, node]
        p_w = nodes_sum[n_grads, node] # Weight is always the last element
        gain_p = p_g1^2 / (p_g2 + lambda * p_w + T(1e-8))

        g_best, b_best, f_best = T(-Inf), Int32(0), Int32(0)

        for j_idx in 1:length(js)
            f = js[j_idx]
            constraint = monotone_constraints[f]
            feattype = feattypes[f]
            
            split_end = feattype ? (nbins - 1) : nbins
            
            # تخصيص (Specialize) for common cases to ensure static, allocation-free code
            if n_grads == 3 # For MSE, LogLoss
                l_g1, l_g2, l_w = zero(T), zero(T), zero(T)
                for b in 1:split_end
                    l_g1 += h∇[1, b, f, node]
                    l_g2 += h∇[2, b, f, node]
                    l_w  += h∇[3, b, f, node]
                    
                    r_w = p_w - l_w
                    if l_w >= min_weight && r_w >= min_weight
                        r_g1, r_g2 = p_g1 - l_g1, p_g2 - l_g2
                        
                        valid_split = true
                        if constraint != 0
                            pred_l = -l_g1 / (l_g2 + lambda * l_w + T(1e-8))
                            pred_r = -r_g1 / (r_g2 + lambda * r_w + T(1e-8))
                            valid_split = (constraint > 0 && pred_l < pred_r) || (constraint < 0 && pred_l > pred_r)
                        end

                        if valid_split
                            gain_l = l_g1^2 / (l_w * lambda + l_g2 + T(1e-8))
                            gain_r = r_g1^2 / (r_w * lambda + r_g2 + T(1e-8))
                            g = gain_l + gain_r - gain_p
                            if g > g_best
                                g_best, b_best, f_best = g, Int32(b), Int32(f)
                            end
                        end
                    end
                end
            elseif n_grads == 5 # For GaussianMLE
                l_g1, l_g2, l_g3, l_g4, l_w = zero(T), zero(T), zero(T), zero(T), zero(T)
                # Note: Gain logic below assumes grad1/hess1. Update if Gaussian logic is different.
                for b in 1:split_end
                    l_g1 += h∇[1, b, f, node]
                    l_g2 += h∇[2, b, f, node]
                    l_g3 += h∇[3, b, f, node]
                    l_g4 += h∇[4, b, f, node]
                    l_w  += h∇[5, b, f, node]
                    
                    r_w = p_w - l_w
                    if l_w >= min_weight && r_w >= min_weight
                        p_g1_gauss = nodes_sum[1, node]
                        p_g2_gauss = nodes_sum[3, node] # hess_mu
                        
                        r_g1, r_g2 = p_g1_gauss - l_g1, p_g2_gauss - l_g3

                        # Simplified gain for demonstration; update with correct multi-dimensional gain logic
                        gain_l = l_g1^2 / (l_g3 * lambda + l_g3 + T(1e-8))
                        gain_r = r_g1^2 / (r_w * lambda + r_g2 + T(1e-8))
                        g = gain_l + gain_r - gain_p
                        if g > g_best
                            g_best, b_best, f_best = g, Int32(b), Int32(f)
                        end
                    end
                end
            end
        end
        gains[n_idx] = g_best
        bins[n_idx] = b_best
        feats[n_idx] = f_best
    end
end

@kernel function separate_nodes_kernel!(
    build_nodes, build_count,
    subtract_nodes, subtract_count,
    @Const(active_nodes)
)
    idx = @index(Global)
    @inbounds if idx <= length(active_nodes)
        node = active_nodes[idx]
        
        if node > 0
            if idx % 2 == 1
                pos = Atomix.@atomic build_count[1] += 1
                build_nodes[pos] = node
            else
                pos = Atomix.@atomic subtract_count[1] += 1
                subtract_nodes[pos] = node
            end
        end
    end
end

@kernel function subtract_hist_kernel!(h∇, @Const(subtract_nodes))
    gidx = @index(Global)

    n_k = size(h∇, 1)
    n_b = size(h∇, 2)
    n_j = size(h∇, 3)
    n_elements_per_node = n_k * n_b * n_j

    node_idx = (gidx - 1) ÷ n_elements_per_node + 1
    
    if node_idx <= length(subtract_nodes)
        remainder = (gidx - 1) % n_elements_per_node
        j = remainder ÷ (n_k * n_b) + 1
        
        remainder = remainder % (n_k * n_b)
        b = remainder ÷ n_k + 1
        
        k = remainder % n_k + 1
        
        @inbounds node = subtract_nodes[node_idx]
        
        if node > 0
            parent = node >> 1
            sibling = node ⊻ 1
            
            @inbounds h∇[k, b, j, node] = h∇[k, b, j, parent] - h∇[k, b, j, sibling]
        end
    end
end

function update_hist_gpu!(
    h∇, gains, bins, feats, ∇, x_bin, nidx, js, is, depth, active_nodes, nodes_sum_gpu, params,
    left_nodes_buf, right_nodes_buf, target_mask_buf, feattypes_gpu, monotone_constraints_gpu
)
    backend = KernelAbstractions.get_backend(h∇)
    n_active = length(active_nodes)
    
    h∇ .= 0
    
    n_feats = length(js)
    n_obs_chunks = cld(length(is), 8)
    num_threads = n_feats * n_obs_chunks
    
    workgroup_size = 256
    hist_kernel_f! = hist_kernel!(backend, workgroup_size)
    hist_kernel_f!(h∇, ∇, x_bin, nidx, js, is; ndrange = num_threads)
    
    KernelAbstractions.synchronize(backend)
    
    find_split! = find_best_split_from_hist_kernel!(backend, workgroup_size)
    find_split!(gains, bins, feats, h∇, nodes_sum_gpu, active_nodes, js,
                feattypes_gpu, monotone_constraints_gpu,
                eltype(gains)(params.lambda), eltype(gains)(params.min_weight);
                ndrange = max(n_active, 1))
                
    KernelAbstractions.synchronize(backend)
end

