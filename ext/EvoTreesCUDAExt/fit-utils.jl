using KernelAbstractions
using Atomix
using StaticArrays

const MAX_K = 8

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
                is_left = feattypes[feat] ? (x_bin[obs, feat] <= bin) : (x_bin[obs, feat] == bin)
                nidx[obs] = (node << 1) + T(!is_left)
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
    K::Int
) where {T}
    gidx = @index(Global, Linear)
    n_feats = length(js)
    n_obs = length(is)
    obs_per_thread = 64
    
    total_work = cld(n_obs, obs_per_thread) * n_feats
    if gidx <= total_work
        feat_idx = (gidx - 1) % n_feats + 1
        obs_chunk = (gidx - 1) ÷ n_feats
        feat = js[feat_idx]
        
        start_idx = obs_chunk * obs_per_thread + 1
        end_idx = min(start_idx + obs_per_thread - 1, n_obs)
        
        @inbounds for obs_idx in start_idx:end_idx
            obs = is[obs_idx]
            node = nidx[obs]
            if node > 0 && node <= size(h∇, 4)
                bin = x_bin[obs, feat]
                if bin > 0 && bin <= size(h∇, 2)
                    for k in 1:(2*K+1)
                        Atomix.@atomic h∇[k, bin, feat, node] += ∇[k, obs]
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
    L2::T,
    K::Int,
    is_mae::Bool,
    is_quantile::Bool,
    is_mle2p::Bool
) where {T}
    n_idx = @index(Global)
    @inbounds if n_idx <= length(active_nodes)
        node = active_nodes[n_idx]
        if node == 0
            gains[n_idx], bins[n_idx], feats[n_idx] = T(-Inf), Int32(0), Int32(0)
        else
            nbins = size(h∇, 2)
            eps = T(1e-8)
            
            @inbounds begin
                local_f = js[1]
                n_grad_hess = is_mle2p ? 5 : (2 * K + 1)
                for k in 1:n_grad_hess
                    total = zero(T)
                    for b in 1:nbins
                        total += T(h∇[k, b, local_f, node])
                    end
                    nodes_sum[k, node] = total
                end
            end
            
            w_p = is_mle2p ? nodes_sum[5, node] : nodes_sum[2*K+1, node]
            g_best, b_best, f_best = T(-Inf), Int32(0), Int32(0)
            
            if is_mae || is_quantile
                parent_g = nodes_sum[1, node]
                parent_h = nodes_sum[2, node]
                
                gain_p = parent_g^2 / (parent_h + lambda * w_p + L2 + eps) / 2
                
                for j_idx in 1:length(js)
                    f = js[j_idx]
                    is_numeric = feattypes[f]
                    
                    bin_range = is_numeric ? (1:(nbins-1)) : (1:(nbins-1))
                    s_w = zero(T)
                    cum_g = zero(T)
                    cum_h = zero(T)
                    
                    for b in bin_range
                        if is_numeric
                            s_w += h∇[2*K+1, b, f, node]
                            cum_g += h∇[1, b, f, node]
                            cum_h += h∇[2, b, f, node]
                        else
                            s_w = h∇[2*K+1, b, f, node]
                            cum_g = h∇[1, b, f, node]
                            cum_h = h∇[2, b, f, node]
                        end
                        
                        l_w, r_w = s_w, w_p - s_w
                        if l_w >= min_weight && r_w >= min_weight
                            l_g, r_g = cum_g, parent_g - cum_g
                            l_h, r_h = cum_h, parent_h - cum_h
                            
                            gain_l = l_g^2 / (l_h + lambda * l_w + L2 + eps) / 2
                            gain_r = r_g^2 / (r_h + lambda * r_w + L2 + eps) / 2
                            g = gain_l + gain_r - gain_p
                            
                            if g > g_best
                                g_best, b_best, f_best = g, Int32(b), Int32(f)
                            end
                        end
                    end
                end
            else
                gain_p = zero(T)
                if is_mle2p && K == 2
                    g1, g2 = nodes_sum[1, node], nodes_sum[2, node]
                    h1, h2 = nodes_sum[3, node], nodes_sum[4, node]
                    gain_p = (g1^2 / (h1 + lambda * w_p + L2 + eps) + g2^2 / (h2 + lambda * w_p + L2 + eps)) / 2
                else
                    w = nodes_sum[2*K+1, node]
                    if w > eps
                        for kk in 1:K
                            g = nodes_sum[kk, node]
                            h = nodes_sum[K+kk, node]
                            gain_p += g^2 / (h + lambda * w + L2 + eps)
                        end
                        gain_p /= 2
                    end
                end
                
                for j_idx in 1:length(js)
                    f = js[j_idx]
                    is_numeric = feattypes[f]
                    constraint = monotone_constraints[f]
                    
                    bin_range = is_numeric ? (1:(nbins-1)) : (1:(nbins-1))
                    s_w = zero(T)
                    cum_g = MVector{MAX_K,T}(ntuple(_->zero(T), MAX_K))
                    cum_h = MVector{MAX_K,T}(ntuple(_->zero(T), MAX_K))
                    
                    for b in bin_range
                        if is_numeric
                            s_w += is_mle2p ? T(h∇[5, b, f, node]) : T(h∇[2*K+1, b, f, node])
                            @inbounds for kk in 1:K
                                cum_g[kk] += T(h∇[kk, b, f, node])
                                if is_mle2p
                                    cum_h[kk] += T(h∇[kk+2, b, f, node])
                                else
                                    cum_h[kk] += T(h∇[K+kk, b, f, node])
                                end
                            end
                        else
                            s_w = is_mle2p ? T(h∇[5, b, f, node]) : T(h∇[2*K+1, b, f, node])
                            @inbounds for kk in 1:K
                                cum_g[kk] = T(h∇[kk, b, f, node])
                                if is_mle2p
                                    cum_h[kk] = T(h∇[kk+2, b, f, node])
                                else
                                    cum_h[kk] = T(h∇[K+kk, b, f, node])
                                end
                            end
                        end
                        
                        if s_w >= min_weight && (w_p - s_w) >= min_weight
                            gain_l_unscaled, gain_r_unscaled = zero(T), zero(T)
                            predL, predR = zero(T), zero(T)
                            
                            @inbounds for kk in 1:K
                                l_g, l_h = cum_g[kk], cum_h[kk]
                                r_w = w_p - s_w

                                if is_mle2p
                                    r_g, r_h = nodes_sum[kk, node] - l_g, nodes_sum[kk+2, node] - l_h
                                else
                                    r_g, r_h = nodes_sum[kk, node] - l_g, nodes_sum[K+kk, node] - l_h
                                end

                                denomL = l_h + lambda * s_w + L2 + eps
                                denomR = r_h + lambda * r_w + L2 + eps
                                
                                gain_l_unscaled += l_g^2 / denomL
                                gain_r_unscaled += r_g^2 / denomR

                                if constraint != 0 && (!is_mle2p || kk == 1)
                                    predL += -l_g / denomL
                                    predR += -r_g / denomR
                                end
                            end
                            
                            constraint_ok = (constraint == 0) || 
                                           (constraint == -1 && predL > predR) || 
                                           (constraint == 1 && predL < predR)

                            if constraint_ok
                                gain_l = gain_l_unscaled / 2
                                gain_r = gain_r_unscaled / 2
                                g = gain_l + gain_r - gain_p
                                if g > g_best
                                    g_best, b_best, f_best = g, Int32(b), Int32(f)
                                end
                            end
                        end
                    end
                end
            end
            gains[n_idx], bins[n_idx], feats[n_idx] = g_best, b_best, f_best
        end
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
            if (node % 2) == 0
                pos = Atomix.@atomic build_count[1] += 1
                build_nodes[pos] = node
            else
                pos = Atomix.@atomic subtract_count[1] += 1
                subtract_nodes[pos] = node
            end
        end
    end
end

@kernel function subtract_hist_kernel!(h∇::AbstractArray{T,4}, @Const(subtract_nodes), n_k, n_b, n_j) where {T}
    gidx = @index(Global)
    n_elements = n_k * n_b * n_j
    
    node_idx = (gidx - 1) ÷ n_elements + 1
    if node_idx <= length(subtract_nodes)
        @inbounds node = subtract_nodes[node_idx]
        if node > 0
            parent = node >> 1
            sibling = node ⊻ 1
            
            elem_idx = (gidx - 1) % n_elements
            j = elem_idx ÷ (n_k * n_b) + 1
            remainder = elem_idx % (n_k * n_b)
            b = remainder ÷ n_k + 1
            k = remainder % n_k + 1
            
            h∇[k, b, j, node] = h∇[k, b, j, parent] - h∇[k, b, j, sibling]
        end
    end
end

function update_hist_gpu!(
    h∇, gains, bins, feats, ∇, x_bin, nidx, js, is, depth, active_nodes,
    nodes_sum_gpu, params, left_nodes_buf, right_nodes_buf, target_mask_buf,
    feattypes, monotone_constraints, K;
    is_mae::Bool=false, is_quantile::Bool=false, is_cred::Bool=false, is_mle2p::Bool=false
)
    backend = KernelAbstractions.get_backend(h∇)
    n_active = length(active_nodes)
    
    h∇ .= 0
    
    n_work = cld(length(is), 64) * length(js)
    workgroup_size = min(256, n_work)
    hist_kernel!(backend)(
        h∇, ∇, x_bin, nidx, js, is, K;
        ndrange = n_work,
        workgroupsize = workgroup_size
    )
    
    if n_active > 16 && depth > 2
        build_count = KernelAbstractions.zeros(backend, Int32, 1)
        subtract_count = KernelAbstractions.zeros(backend, Int32, 1)
        
        separate_nodes_kernel!(backend)(
            left_nodes_buf, build_count, right_nodes_buf, subtract_count, active_nodes;
            ndrange = n_active, workgroupsize = min(256, n_active)
        )
        
        KernelAbstractions.synchronize(backend)
        
        n_subtract = Array(subtract_count)[1]
        if n_subtract > 0
            n_k, n_b, n_j = size(h∇, 1), size(h∇, 2), size(h∇, 3)
            subtract_hist_kernel!(backend)(
                h∇, view(right_nodes_buf, 1:n_subtract), n_k, n_b, n_j;
                ndrange = n_subtract * n_k * n_b * n_j,
                workgroupsize = 256
            )
        end
    end
    
    find_best_split_from_hist_kernel!(backend)(
        gains, bins, feats, h∇, nodes_sum_gpu, active_nodes, js,
        feattypes, monotone_constraints,
        eltype(gains)(params.lambda),
        eltype(gains)(params.min_weight),
        eltype(gains)(params.L2),
        K,
        is_mae,
        is_quantile,
        is_mle2p;
        ndrange = n_active,
        workgroupsize = min(256, n_active)
    )
    
    KernelAbstractions.synchronize(backend)
end

