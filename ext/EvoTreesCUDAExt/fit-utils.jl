using KernelAbstractions
using Atomix

# ============================
# Core Kernels - Consolidated
# ============================

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

@kernel function fill_mask_kernel!(mask::AbstractVector{UInt8}, @Const(nodes))
    i = @index(Global)
    @inbounds if i <= length(nodes)
        node = nodes[i]
        node > 0 && node <= length(mask) && (mask[node] = UInt8(1))
    end
end

# ============================
# Optimized Histogram Building
# ============================

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
    obs_per_thread = 8  # Keep original - the issue is not this
    
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
                        Atomix.@atomic h∇[k, bin, feat_idx, node] += ∇[k, obs]
                    end
                end
            end
        end
    end
end

# ============================
# Smart Split Finding - Handles both numeric and categorical
# ============================

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
    K::Int
) where {T}
    n_idx = @index(Global)
    @inbounds if n_idx <= length(active_nodes)
        node = active_nodes[n_idx]
        if node == 0
            gains[n_idx], bins[n_idx], feats[n_idx] = T(-Inf), Int32(0), Int32(0)
        else
            nbins = size(h∇, 2)
            eps = T(1e-8)
            
            # Compute node statistics
            for k in 1:(2*K+1)
                sum_val = zero(T)
                for j_idx in 1:length(js), b in 1:nbins
                    sum_val += h∇[k, b, j_idx, node]
                end
                nodes_sum[k, node] = sum_val
            end
            
            # Parent gain
            w_p = nodes_sum[2*K+1, node]
            gain_p = zero(T)
            # For simplicity, compute gain for first gradient only
            g, h = nodes_sum[1, node], nodes_sum[K+1, node]
            gain_p = g^2 / (h + lambda * w_p / K + eps)
            
            g_best, b_best, f_best = T(-Inf), Int32(0), Int32(0)
            
            # Find best split across features
            for j_idx in 1:length(js)
                f = js[j_idx]
                is_numeric = feattypes[f]
                constraint = monotone_constraints[f]
                
                cum_g = cum_h = cum_w = zero(T)
                
                for b in 1:(nbins - 1)
                    # Update cumulative stats
                    if is_numeric
                        cum_g += h∇[1, b, j_idx, node]
                        cum_h += h∇[K+1, b, j_idx, node]
                        cum_w += h∇[2*K+1, b, j_idx, node]
                    else
                        cum_g = h∇[1, b, j_idx, node]
                        cum_h = h∇[K+1, b, j_idx, node]
                        cum_w = h∇[2*K+1, b, j_idx, node]
                    end
                    
                    l_w, r_w = cum_w, w_p - cum_w
                    if l_w >= min_weight && r_w >= min_weight
                        l_g, l_h = cum_g, cum_h
                        r_g, r_h = nodes_sum[1, node] - l_g, nodes_sum[K+1, node] - l_h
                        
                        # Quick monotonic check
                        if constraint == 0 || 
                           (constraint == -1 && -l_g/(l_h + lambda * l_w / K + eps) > -r_g/(r_h + lambda * r_w / K + eps)) ||
                           (constraint == 1 && -l_g/(l_h + lambda * l_w / K + eps) < -r_g/(r_h + lambda * r_w / K + eps))
                            g = l_g^2 / (l_h + lambda * l_w / K + eps) + r_g^2 / (r_h + lambda * r_w / K + eps) - gain_p
                            if g > g_best
                                g_best, b_best, f_best = g, Int32(b), Int32(f)
                            end
                        end
                    end
                end
            end
            
            gains[n_idx], bins[n_idx], feats[n_idx] = g_best, b_best, f_best
        end
    end
end

# ============================
# Split-Build Pattern Helpers
# ============================

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

@kernel function subtract_hist_kernel!(h∇L::AbstractArray{T,4}, @Const(subtract_nodes)) where {T}
    gidx = @index(Global)
    n_elements = size(h∇, 1) * size(h∇, 2) * size(h∇, 3)
    
    node_idx = (gidx - 1) ÷ n_elements + 1
    if node_idx <= length(subtract_nodes)
        @inbounds node = subtract_nodes[node_idx]
        if node > 0
            parent = node >> 1
            sibling = node ⊻ 1
            
            elem_idx = (gidx - 1) % n_elements
            j = elem_idx ÷ (size(h∇, 1) * size(h∇, 2)) + 1
            remainder = elem_idx % (size(h∇, 1) * size(h∇, 2))
            b = remainder ÷ size(h∇, 1) + 1
            k = remainder % size(h∇, 1) + 1
            
            @inbounds h∇L[k, b, j, node] = h∇[k, b, j, parent] - h∇[k, b, j, sibling]
        end
    end
end

# ============================
# Main Update Function - Streamlined
# ============================

function update_hist_gpu!(
    h∇, gains, bins, feats, ∇, x_bin, nidx, js, is, depth, active_nodes,
    nodes_sum_gpu, params, left_nodes_buf, right_nodes_buf, target_mask_buf,
    feattypes, monotone_constraints, K
)
    backend = KernelAbstractions.get_backend(h∇)
    n_active = length(active_nodes)
    
    h∇ .= 0
    
    # Build histogram with optimized workgroup size
    n_work = cld(length(is), 8) * length(js)  # Back to original
    workgroup_size = min(256, n_work)  # Simple approach
    hist_kernel!(backend)(
        h∇, ∇, x_bin, nidx, js, is, K;
        ndrange = n_work,
        workgroupsize = workgroup_size
    )
    
    # Apply split-build optimization for deep trees
    if n_active > 16 && depth > 2
        build_count = KernelAbstractions.zeros(backend, Int32, 1)
        subtract_count = KernelAbstractions.zeros(backend, Int32, 1)
        
        separate_nodes_kernel!(backend)(
            left_nodes_buf, build_count, right_nodes_buf, subtract_count, active_nodes;
            ndrange = n_active, workgroupsize = min(256, n_active)
        )
        
        # Only sync when we need the counts
        KernelAbstractions.synchronize(backend)
        
        n_subtract = Array(subtract_count)[1]
        if n_subtract > 0
            subtract_hist_kernel!(backend)(
                h∇, view(right_nodes_buf, 1:n_subtract);
                ndrange = n_subtract * size(h∇, 1) * size(h∇, 2) * size(h∇, 3),
                workgroupsize = 256
            )
        end
    end
    
    # Find best splits
    find_best_split_from_hist_kernel!(backend)(
        gains, bins, feats, h∇, nodes_sum_gpu, active_nodes, js,
        feattypes, monotone_constraints,
        eltype(gains)(params.lambda),
        eltype(gains)(params.min_weight), K;
        ndrange = n_active,
        workgroupsize = min(256, n_active)
    )
    
    # Only sync at the end when results are needed
    KernelAbstractions.synchronize(backend)
end

