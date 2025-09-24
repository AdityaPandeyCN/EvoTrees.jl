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

@kernel function hist_kernel!(
    h∇::AbstractArray{T,4},
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
    @Const(is),
    K::Int,
    chunk_size::Int
) where {T}
    gidx = @index(Global, Linear)
    
    n_feats = length(js)
    n_obs = length(is)
    total_work_items = n_feats * cld(n_obs, chunk_size)
    
    if gidx <= total_work_items
        feat_idx = (gidx - 1) % n_feats + 1
        obs_chunk = (gidx - 1) ÷ n_feats
        
        feat = js[feat_idx]
        
        start_idx = obs_chunk * chunk_size + 1
        end_idx = min(start_idx + (chunk_size - 1), n_obs)
        
        @inbounds for obs_idx in start_idx:end_idx
            obs = is[obs_idx]
            node = nidx[obs]
            if node > 0 && node <= size(h∇, 4)
                bin = x_bin[obs, feat]
                if bin > 0 && bin <= size(h∇, 2)
                    for k in 1:(2*K+1)
                        grad = ∇[k, obs]
                        Atomix.@atomic h∇[k, bin, feat, node] += grad
                    end
                end
            end
        end
    end
end

@kernel function reduce_root_sums_kernel!(nodes_sum, @Const(∇), @Const(is))
    idx = @index(Global)
    if idx <= length(is)
        obs = is[idx]
        n_k = size(∇, 1)
        @inbounds for k in 1:n_k
            Atomix.@atomic nodes_sum[k, 1] += ∇[k, obs]
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
    L2::T,
    min_weight::T,
    K::Int,
    loss_id::Int32,
    sums_temp::AbstractArray{T,2}
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
            eps = T(1e-8)
            
            # Initialize node sums from first feature - manual sum instead of sum()
            if !isempty(js)
                first_feat = js[1]
                for k in 1:(2*K+1)
                    sum_val = zero(T)
                    for b in 1:nbins
                        sum_val += h∇[k, b, first_feat, node]
                    end
                    nodes_sum[k, node] = sum_val
                end
            end
            
            # Pre-calculate parent components
            w_p = nodes_sum[2*K+1, node]
            λw_p = lambda * w_p
            
            # Parent gain depends on loss
            gain_p = zero(T)
            if loss_id == 1 # GradientRegression (K may be 1)
                if K == 1
                    g_p = nodes_sum[1, node]
                    h_p = nodes_sum[2, node]
                    denom_p = h_p + λw_p + L2
                    denom_p = denom_p < eps ? eps : denom_p
                    gain_p = g_p^2 / denom_p / 2
                else
                    # fall back to per-class gradient-like (not expected)
                    for k in 1:K
                        g_p = nodes_sum[k, node]
                        h_p = nodes_sum[K+k, node]
                        denom_p = h_p + λw_p + L2
                        denom_p = denom_p < eps ? eps : denom_p
                        gain_p += g_p^2 / denom_p / 2
                    end
                end
            elseif loss_id == 2 # MLE2P (Gaussian/Logistic MLE)
                # sums: [g1, g2, h1, h2, w]
                g1 = nodes_sum[1, node]
                g2 = nodes_sum[2, node]
                h1 = nodes_sum[3, node]
                h2 = nodes_sum[4, node]
                denom1 = h1 + λw_p + L2
                denom2 = h2 + λw_p + L2
                denom1 = denom1 < eps ? eps : denom1
                denom2 = denom2 < eps ? eps : denom2
                gain_p = (g1^2 / denom1 + g2^2 / denom2) / 2
            elseif loss_id == 3 # MLogLoss
                for k in 1:K
                    gk = nodes_sum[k, node]
                    hk = nodes_sum[K+k, node]
                    denom = hk + λw_p + L2
                    denom = denom < eps ? eps : denom
                    gain_p += gk^2 / denom / 2
                end
            elseif loss_id == 4 || loss_id == 5 # MAE or Quantile
                # parent term is not subtracted for MAE/Quantile in CPU impl
                gain_p = zero(T)
            elseif loss_id == 6 # Credibility
                # Z = VHM / (VHM + EVPV)
                μp = nodes_sum[1, node] / w_p
                VHM = μp^2
                EVPV = nodes_sum[2, node] / w_p - VHM
                EVPV = EVPV < eps ? eps : EVPV
                Zp = VHM / (VHM + EVPV)
                gain_p = Zp * abs(nodes_sum[1, node]) / (1 + L2 / w_p)
            end
            
            g_best = T(-Inf)
            b_best = Int32(0)
            f_best = Int32(0)
            
            for j_idx in 1:length(js)
                f = js[j_idx]
                is_numeric = feattypes[f]
                constraint = monotone_constraints[f]
                
                # Initialize accumulators directly in sums_temp for K > 1
                if K == 1
                    acc1 = zero(T) # g or first-order term
                    acc2 = zero(T) # h or second-order term or y^2 term depending on loss
                    accw = zero(T)
                else
                    for kk in 1:(2*K+1)
                        sums_temp[kk, n_idx] = zero(T)
                    end
                end
                
                for b in 1:(nbins - 1)
                    # Update accumulator based on feature type
                    if K == 1
                        if is_numeric
                            # For K==1, hist layout: [g1, h1_or_aux, w]
                            acc1 += h∇[1, b, f, node]
                            acc2 += h∇[2, b, f, node]
                            accw += h∇[3, b, f, node]
                        else
                            acc1 = h∇[1, b, f, node]
                            acc2 = h∇[2, b, f, node]
                            accw = h∇[3, b, f, node]
                        end
                        w_l = accw
                        w_r = w_p - w_l
                        (w_l < min_weight || w_r < min_weight) && continue
                        
                        g_val = zero(T)
                        if loss_id == 1
                            # GradientRegression
                            g_l = acc1
                            h_l = acc2
                            g_r = nodes_sum[1, node] - g_l
                            h_r = nodes_sum[2, node] - h_l
                            d_l = h_l + lambda * w_l + L2; d_l = d_l < eps ? eps : d_l
                            d_r = h_r + lambda * w_r + L2; d_r = d_r < eps ? eps : d_r
                            g_val = (g_l^2 / d_l + g_r^2 / d_r) / 2 - gain_p
                            # constraint check via pred scalars
                            if constraint != 0
                                pred_l = -acc1 / d_l
                                pred_r = -(nodes_sum[1, node] - acc1) / d_r
                                if (constraint == -1 && pred_l <= pred_r) || (constraint == 1 && pred_l >= pred_r)
                                    continue
                                end
                            end
                        elseif loss_id == 2
                            # MLE2P (use first parameter for constraint)
                            g1_l = acc1
                            h1_l = acc2
                            g1_r = nodes_sum[1, node] - g1_l
                            h1_r = nodes_sum[3, node] - h1_l
                            # second parameter components from hist: indices 2 and 4
                            g2_l = is_numeric ? (sums_temp[2, n_idx] += h∇[2, b, f, node]; sums_temp[2, n_idx]) : h∇[2, b, f, node]
                            h2_l = is_numeric ? (sums_temp[4, n_idx] += h∇[4, b, f, node]; sums_temp[4, n_idx]) : h∇[4, b, f, node]
                            g2_r = nodes_sum[2, node] - g2_l
                            h2_r = nodes_sum[4, node] - h2_l
                            d1_l = h1_l + lambda * w_l + L2; d1_l = d1_l < eps ? eps : d1_l
                            d1_r = h1_r + lambda * w_r + L2; d1_r = d1_r < eps ? eps : d1_r
                            d2_l = h2_l + lambda * w_l + L2; d2_l = d2_l < eps ? eps : d2_l
                            d2_r = h2_r + lambda * w_r + L2; d2_r = d2_r < eps ? eps : d2_r
                            g_val = ((g1_l^2 / d1_l + g2_l^2 / d2_l) + (g1_r^2 / d1_r + g2_r^2 / d2_r)) / 2 - gain_p
                            if constraint != 0
                                pred_l = -g1_l / d1_l
                                pred_r = -g1_r / d1_r
                                if (constraint == -1 && pred_l <= pred_r) || (constraint == 1 && pred_l >= pred_r)
                                    continue
                                end
                            end
                        elseif loss_id == 3
                            # Not supporting constraint for softmax
                            gain_l = zero(T); gain_r = zero(T)
                            # Use class 1..K from nodes_sum (K should be >1 in this branch)
                            for k in 1:K
                                # For K==1 branch not taken
                            end
                            # This branch not used when K==1; keep g_val untouched
                        elseif loss_id == 4 || loss_id == 5
                            # MAE / Quantile gains
                            μp = nodes_sum[1, node] / w_p
                            μl = acc1 / w_l
                            μr = (nodes_sum[1, node] - acc1) / w_r
                            d_l = 1 + lambda + L2 / w_l
                            d_r = 1 + lambda + L2 / w_r
                            d_l = d_l < eps ? eps : d_l
                            d_r = d_r < eps ? eps : d_r
                            g_val = abs(μl - μp) * w_l / d_l + abs(μr - μp) * w_r / d_r
                            if constraint != 0
                                pred_l = acc1 / (w_l + lambda * w_l + L2)
                                pred_r = (nodes_sum[1, node] - acc1) / (w_r + lambda * w_r + L2)
                                if (constraint == -1 && pred_l <= pred_r) || (constraint == 1 && pred_l >= pred_r)
                                    continue
                                end
                            end
                        elseif loss_id == 6
                            # Credibility gains
                            μp = nodes_sum[1, node] / w_p
                            VHM_p = μp^2
                            EVPV_p = nodes_sum[2, node] / w_p - VHM_p
                            EVPV_p = EVPV_p < eps ? eps : EVPV_p
                            Zp = VHM_p / (VHM_p + EVPV_p)
                            μl = acc1 / w_l
                            VHM_l = μl^2
                            EVPV_l = acc2 / w_l - VHM_l
                            EVPV_l = EVPV_l < eps ? eps : EVPV_l
                            Zl = VHM_l / (VHM_l + EVPV_l)
                            g_l = Zl * abs(acc1) / (1 + L2 / w_l)
                            μr = (nodes_sum[1, node] - acc1) / w_r
                            VHM_r = μr^2
                            EVPV_r = (nodes_sum[2, node] - acc2) / w_r - VHM_r
                            EVPV_r = EVPV_r < eps ? eps : EVPV_r
                            Zr = VHM_r / (VHM_r + EVPV_r)
                            g_r = Zr * abs(nodes_sum[1, node] - acc1) / (1 + L2 / w_r)
                            g_val = g_l + g_r - Zp * abs(nodes_sum[1, node]) / (1 + L2 / w_p)
                            if constraint != 0
                                pred_l = acc1 / (w_l + L2)
                                pred_r = (nodes_sum[1, node] - acc1) / (w_r + L2)
                                if (constraint == -1 && pred_l <= pred_r) || (constraint == 1 && pred_l >= pred_r)
                                    continue
                                end
                            end
                        end
                        
                    else  # K > 1
                        if is_numeric
                            for kk in 1:(2*K+1)
                                sums_temp[kk, n_idx] += h∇[kk, b, f, node]
                            end
                        else
                            for kk in 1:(2*K+1)
                                sums_temp[kk, n_idx] = h∇[kk, b, f, node]
                            end
                        end
                        
                        w_l = sums_temp[2*K+1, n_idx]
                        w_r = w_p - w_l
                        (w_l < min_weight || w_r < min_weight) && continue
                        
                        # For multi-output non-softmax (e.g., MLE2P), enforce monotone constraint on first parameter
                        if constraint != 0 && loss_id != 3
                            g_l1 = sums_temp[1, n_idx]
                            h_l1 = sums_temp[K+1, n_idx]
                            g_r1 = nodes_sum[1, node] - g_l1
                            h_r1 = nodes_sum[K+1, node] - h_l1
                            d1_l = h_l1 + lambda * w_l + L2; d1_l = d1_l < eps ? eps : d1_l
                            d1_r = h_r1 + lambda * w_r + L2; d1_r = d1_r < eps ? eps : d1_r
                            pred_l = -g_l1 / d1_l
                            pred_r = -g_r1 / d1_r
                            if (constraint == -1 && pred_l <= pred_r) || (constraint == 1 && pred_l >= pred_r)
                                continue
                            end
                        end
                        
                        # Only two multi-output branches are supported: MLogLoss and Gradient-like fallback
                        if loss_id == 3
                            gain_l = zero(T)
                            gain_r = zero(T)
                            for k in 1:K
                                g_l = sums_temp[k, n_idx]
                                h_l = sums_temp[K+k, n_idx]
                                g_r = nodes_sum[k, node] - g_l
                                h_r = nodes_sum[K+k, node] - h_l
                                d_l = h_l + lambda * w_l + L2; d_l = d_l < eps ? eps : d_l
                                d_r = h_r + lambda * w_r + L2; d_r = d_r < eps ? eps : d_r
                                gain_l += g_l^2 / d_l / 2
                                gain_r += g_r^2 / d_r / 2
                            end
                            g_val = gain_l + gain_r - gain_p
                        else
                            # Fallback gradient-like formula (covers MLE2P)
                            gain_l = zero(T)
                            gain_r = zero(T)
                            for k in 1:K
                                g_l = sums_temp[k, n_idx]
                                h_l = sums_temp[K+k, n_idx]
                                g_r = nodes_sum[k, node] - g_l
                                h_r = nodes_sum[K+k, node] - h_l
                                d_l = h_l + lambda * w_l + L2; d_l = d_l < eps ? eps : d_l
                                d_r = h_r + lambda * w_r + L2; d_r = d_r < eps ? eps : d_r
                                gain_l += g_l^2 / d_l / 2
                                gain_r += g_r^2 / d_r / 2
                            end
                            g_val = gain_l + gain_r - gain_p
                        end
                    end
                    
                    # Update best split if better
                    if g_val > g_best
                        g_best = g_val
                        b_best = Int32(b)
                        f_best = Int32(f)
                    end
                end
            end
            
            gains[n_idx] = g_best
            bins[n_idx] = b_best
            feats[n_idx] = f_best
        end
    end
end

# Brief: Splits active nodes into two groups: odd-indexed nodes whose histograms must be built from data (build_nodes)
# and even-indexed nodes whose histograms can be derived by subtraction from their sibling (subtract_nodes).
@kernel function separate_nodes_kernel!(
    build_nodes, build_count,
    subtract_nodes, subtract_count,
    @Const(active_nodes)
)
    idx = @index(Global)
    @inbounds node = active_nodes[idx]
    
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

# Brief: For nodes marked as subtract_nodes, rebuild their histograms by subtracting the sibling histogram from the parent histogram.
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
    h∇, gains::AbstractVector{T}, bins::AbstractVector{Int32}, feats::AbstractVector{Int32}, ∇, x_bin, nidx, js, is, depth, active_nodes, nodes_sum_gpu, params,
    feattypes, monotone_constraints, K, loss_id::Int32, L2::T, sums_temp=nothing
) where {T}
    backend = KernelAbstractions.get_backend(h∇)
    n_active = length(active_nodes)
    
    if sums_temp === nothing && K > 1
        sums_temp = similar(nodes_sum_gpu, 2*K+1, max(n_active, 1))
    elseif K == 1
        sums_temp = similar(nodes_sum_gpu, 1, 1)
    end
    
    # Clear hist only for the nodes being built; copy node ids to CPU to avoid scalar device indexing
    if n_active > 0
        anodes_cpu = Array(view(active_nodes, 1:n_active))
        for node in anodes_cpu
            if node > 0
                @views h∇[:, :, :, node] .= zero(eltype(h∇))
            end
        end
        KernelAbstractions.synchronize(backend)
    end
    
    n_feats = length(js)
    chunk_size = 64
    n_obs_chunks = cld(length(is), chunk_size)
    num_threads = n_feats * n_obs_chunks
    
    hist_kernel_f! = hist_kernel!(backend)
    workgroup_size = min(256, max(64, num_threads))
    hist_kernel_f!(h∇, ∇, x_bin, nidx, js, is, K, chunk_size; ndrange = num_threads, workgroupsize = workgroup_size)
    KernelAbstractions.synchronize(backend)
    
    find_split! = find_best_split_from_hist_kernel!(backend)
    find_split!(gains, bins, feats, h∇, nodes_sum_gpu, active_nodes, js, feattypes, monotone_constraints,
                eltype(gains)(params.lambda), L2, eltype(gains)(params.min_weight), K, loss_id, sums_temp;
                ndrange = max(n_active, 1), workgroupsize = min(256, max(64, n_active)))
    KernelAbstractions.synchronize(backend)
end

