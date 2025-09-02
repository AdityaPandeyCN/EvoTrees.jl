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
    tix, tiy, k = @index(Local, NTuple{3})
    bdx, bdy = @groupsize()
    bix, biy = @index(Group, NTuple{2})
    gdx = @gridsize(1)
    
    j = tiy + bdy * (biy - 1)
    @inbounds if j <= length(js)
        jdx = js[j]
        i_max = length(is)
        niter = cld(i_max, bdx * gdx)
        
        for iter = 1:niter
            i = tix + bdx * (bix - 1) + bdx * gdx * (iter - 1)
            @inbounds if i <= i_max
                idx = is[i]
                node = nidx[idx]
                @inbounds if node > 0 && node <= size(h∇, 4)
                    bin = x_bin[idx, jdx]
                    @inbounds if bin > 0 && bin <= size(h∇, 2)
                        Atomix.@atomic h∇[k, bin, j, node] += ∇[k, idx]
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
            
            # Parent gain - SUM OVER ALL K
            w_p = nodes_sum[2*K+1, node]
            gain_p = zero(T)
            for k in 1:K  # FIX: Loop over all K
                g = nodes_sum[k, node]
                h = nodes_sum[K+k, node]
                gain_p += g^2 / (h + lambda * w_p / K + eps)
            end
            
            g_best, b_best, f_best = T(-Inf), Int32(0), Int32(0)
            
            # Find best split across features
            for j_idx in 1:length(js)
                f = js[j_idx]
                is_numeric = feattypes[f]
                constraint = monotone_constraints[f]
                
                for b in 1:(nbins - 1)
                    # Calculate weights
                    cum_w = zero(T)
                    if is_numeric
                        for bb in 1:b
                            cum_w += h∇[2*K+1, bb, j_idx, node]
                        end
                    else
                        cum_w = h∇[2*K+1, b, j_idx, node]
                    end
                    
                    l_w, r_w = cum_w, w_p - cum_w
                    if l_w >= min_weight && r_w >= min_weight
                        gain_valid = true
                        gain_l = zero(T)
                        gain_r = zero(T)
                        
                        # FIX: Compute gain for ALL K outputs
                        for k in 1:K
                            cum_g = zero(T)
                            cum_h = zero(T)
                            
                            if is_numeric
                                for bb in 1:b
                                    cum_g += h∇[k, bb, j_idx, node]
                                    cum_h += h∇[K+k, bb, j_idx, node]
                                end
                            else
                                cum_g = h∇[k, b, j_idx, node]
                                cum_h = h∇[K+k, b, j_idx, node]
                            end
                            
                            l_g, l_h = cum_g, cum_h
                            r_g = nodes_sum[k, node] - l_g
                            r_h = nodes_sum[K+k, node] - l_h
                            
                            # Monotonic constraint check (only for first output)
                            if k == 1 && constraint != 0
                                pred_l = -l_g / (l_h + lambda * l_w / K + eps)
                                pred_r = -r_g / (r_h + lambda * r_w / K + eps)
                                if (constraint == -1 && pred_l <= pred_r) || 
                                   (constraint == 1 && pred_l >= pred_r)
                                    gain_valid = false
                                    break  # Exit k loop
                                end
                            end
                            
                            if gain_valid
                                gain_l += l_g^2 / (l_h + lambda * l_w / K + eps)
                                gain_r += r_g^2 / (r_h + lambda * r_w / K + eps)
                            end
                        end
                        
                        if gain_valid
                            g = gain_l + gain_r - gain_p
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

@kernel function subtract_hist_kernel!(h∇L::AbstractArray{T,4}, @Const(h∇), @Const(subtract_nodes)) where {T}
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
    
    # Build histogram with optimized configuration
    k = size(h∇, 1)
    max_threads = 1024
    ty = max(1, min(length(js), fld(max_threads, k)))
    tx = min(64, max(1, min(length(is), fld(max_threads, k * ty))))
    threads = (k, ty, tx)
    by = cld(length(js), ty)
    bx = min(65535 ÷ by, cld(length(is), tx))
    blocks = (1, by, bx)
    
    hist_kernel!(backend)(
        h∇, ∇, x_bin, nidx, js, is, K;
        ndrange = (length(is), length(js), k),
        workgroupsize = threads
    )
    
    # Apply split-build optimization for deep trees
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
            subtract_hist_kernel!(backend)(
                h∇, h∇, view(right_nodes_buf, 1:n_subtract);
                ndrange = n_subtract * size(h∇, 1) * size(h∇, 2) * size(h∇, 3),
                workgroupsize = 256
            )
        end
    end
    
    KernelAbstractions.synchronize(backend)
    
    # Find best splits with optimized workgroup size
    find_best_split_from_hist_kernel!(backend)(
        gains, bins, feats, h∇, nodes_sum_gpu, active_nodes, js,
        feattypes, monotone_constraints,
        eltype(gains)(params.lambda),
        eltype(gains)(params.min_weight), K;
        ndrange = n_active,
        workgroupsize = min(512, n_active)
    )
    
    KernelAbstractions.synchronize(backend)
end

