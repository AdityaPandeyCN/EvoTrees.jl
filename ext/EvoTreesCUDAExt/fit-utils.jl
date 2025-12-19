using KernelAbstractions
using Atomix

"""
	update_nodes_idx_kernel!(nidx, is, x_bin, cond_feats, cond_bins, feattypes)

Update observation-to-node assignments by traversing splits (left child = node*2, right child = node*2+1).
"""
@kernel function update_nodes_idx_kernel!(
    nidx::AbstractVector{T},        # Node index for each observation (in/out)
    @Const(is),                     # Observation indices to process
    @Const(x_bin),                  # Binned feature values [n_obs, n_feats]
    @Const(cond_feats),             # Split feature for each node
    @Const(cond_bins),              # Split threshold for each node
    @Const(feattypes),              # Feature types (true=numeric, false=categorical)
) where {T<:Unsigned}
    gidx = @index(Global)
    @inbounds if gidx <= length(is)
        obs = is[gidx]              # Get observation index
        node = nidx[obs]            # Get current node for this observation
        if node > 0                 # If observation is in an active node
            feat = cond_feats[node] # Get split feature for this node
            bin = cond_bins[node]   # Get split threshold
            # If bin == 0, node is a leaf - keep the current node ID
            if bin != 0
                feattype = feattypes[feat]
                is_left = feattype ? (x_bin[obs, feat] <= bin) : (x_bin[obs, feat] == bin)
                nidx[obs] = (node << 1) + T(Int(!is_left))
            end
            # If bin == 0, do nothing - nidx[obs] already has the leaf node ID
        end
    end
end

"""
	hist_kernel!(h∇, ∇, x_bin, nidx, js, is, K, chunk_size, target_mask)

Build gradient histograms for active nodes using atomic operations to accumulate gradients by bin.
"""
@kernel function hist_kernel!(
    h∇::AbstractArray{T,4},         # Histogram [2K+1, n_bins, n_feats, n_nodes]
    @Const(∇),                      # Gradients [2K+1, n_obs]
    @Const(x_bin),                  # Binned features [n_obs, n_feats]
    @Const(nidx),                   # Node index for each observation
    @Const(js),                     # Feature indices to process
    @Const(is),                     # Observation indices to process
    K::Int,                         # Number of output dimensions
    chunk_size::Int,                # Observations per thread (reduces contention)
    @Const(target_mask)             # Mask indicating which nodes to build histograms for
) where {T}
    gidx = @index(Global, Linear)

    n_feats = length(js)
    n_obs = length(is)
    total_chunks = cld(n_obs, chunk_size)
    total_threads = n_feats * total_chunks

    if gidx <= total_threads
        feat_idx = (gidx - 1) % n_feats + 1
        chunk_idx = (gidx - 1) ÷ n_feats
        feat = js[feat_idx]

        start_obs = chunk_idx * chunk_size + 1
        end_obs = min(start_obs + chunk_size - 1, n_obs)

        @inbounds for obs_idx in start_obs:end_obs
            obs = is[obs_idx]
            node = nidx[obs]

            if node > 0 && node <= size(h∇, 4) && target_mask[node] != 0
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

# Split active siblings into BUILD (smaller) vs SUBTRACT (larger) lists; sibling via node ⊻ 1
@kernel function separate_nodes_kernel!(
    build_nodes, build_count,       # Output: nodes to build via observation scan
    subtract_nodes, subtract_count, # Output: nodes to compute via subtraction
    @Const(active_nodes),           # Input: all active child nodes at current depth
    @Const(node_counts)             # Input: raw counts per node (number of observations)
)
    idx = @index(Global)
    @inbounds if idx <= length(active_nodes)
        node = active_nodes[idx]

        if node > 0
            sibling = node ⊻ 1

            # Compare raw observation counts (not weights)
            w_node = node_counts[node]
            w_sibling = node_counts[sibling]

            # Tiebreak by node id on equality
            if w_node < w_sibling || (w_node == w_sibling && node < sibling)
                pos = Atomix.@atomic build_count[1] += 1
                build_nodes[pos] = node
            else
                pos = Atomix.@atomic subtract_count[1] += 1
                subtract_nodes[pos] = node
            end
        end
    end
end

# Compute hist via subtraction: h∇[child] = h∇[parent] - h∇[sibling]
@kernel function subtract_hist_kernel!(
    h∇,                    # Histogram [2K+1, n_bins, n_feats, n_nodes] - modified in-place
    @Const(subtract_nodes) # List of larger children to compute via subtraction
)
    gidx = @index(Global)

    # Decode histogram dimensions to parallelize across all elements
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

"""
	reduce_root_sums_kernel!(nodes_sum, ∇, is)

Accumulate gradient sums for the root node using atomic operations.
"""
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

"""
    compute_nodes_sum_kernel!(nodes_sum, h∇, active_nodes, K)

Precompute gradient sums for each active node by summing histogram across all bins.
"""
@kernel function compute_nodes_sum_kernel!(
    nodes_sum,
    @Const(h∇),
    @Const(active_nodes),
    K::Int
)
    gidx = @index(Global)
    n_active = length(active_nodes)
    n_k = 2 * K + 1

    # Parallelizes over n_active * (2K+1) threads
    # Each thread computes one gradient component for one node
    @inbounds if gidx <= n_active * n_k
        n_idx = (gidx - 1) ÷ n_k + 1
        k = (gidx - 1) % n_k + 1
        node = active_nodes[n_idx]

        if node > 0
            nbins = size(h∇, 2)
            # Sum histogram values across all bins for gradient component k
            sum_val = zero(eltype(nodes_sum))
            for b in 1:nbins
                sum_val += h∇[k, b, 1, node]
            end
            nodes_sum[k, node] = sum_val
        end
    end
end

"""
    find_best_split_parallel_kernel!(L, gains, bins, h∇, nodes_sum, active_nodes, 
        js, feattypes, monotone_constraints, lambda, L2, min_weight, K, n_feats, sums_temp)

Find best split for each (node, feature) pair. Julia specializes on type L,
compiling away loss-type branches at kernel generation time.
"""
@kernel function find_best_split_parallel_kernel!(
    ::Type{L},
    gains::AbstractMatrix{T},
    bins::AbstractMatrix{Int32},
    @Const(h∇), @Const(nodes_sum), @Const(active_nodes), @Const(js),
    @Const(feattypes), @Const(monotone_constraints),
    lambda::T, L2::T, min_weight::T,
    K::Int, n_feats::Int, sums_temp::AbstractArray{T,2},
) where {T,L}
    gidx = @index(Global)
    n_active = length(active_nodes)
    ε = T(1e-8)

    @inbounds if gidx <= n_active * n_feats
        n_idx = (gidx - 1) ÷ n_feats + 1
        f_idx = (gidx - 1) % n_feats + 1
        node = active_nodes[n_idx]

        if node == 0
            gains[f_idx, n_idx] = T(-Inf)
            bins[f_idx, n_idx] = Int32(0)
        else
            f, nbins = js[f_idx], size(h∇, 2)
            is_numeric = feattypes[f]
            constraint = monotone_constraints[f]
            w_p = nodes_sum[2*K+1, node]

            # === Parent gain (compile-time specialized) ===
            gain_p = zero(T)
            if L <: EvoTrees.GradientRegression || L <: EvoTrees.MLE2P || L == EvoTrees.MLogLoss
                λw = lambda * w_p
                for k in 1:K
                    d = nodes_sum[K+k, node] + λw + L2
                    gain_p += nodes_sum[k, node]^2 / (d < ε ? ε : d)
                end
                gain_p /= 2
            elseif L <: EvoTrees.Cred
                μ = nodes_sum[1, node] / w_p
                VHM = μ^2
                EVPV = nodes_sum[2, node] / w_p - VHM
                EVPV = EVPV < ε ? ε : EVPV
                gain_p = VHM / (VHM + EVPV) * abs(nodes_sum[1, node]) / (1 + L2 / w_p)
            end

            g_best, b_best = T(-Inf), Int32(0)
            temp_idx = (n_idx - 1) * n_feats + f_idx
            acc1, acc2, accw = zero(T), zero(T), zero(T)

            if K > 1
                for kk in 1:(2*K+1); sums_temp[kk, temp_idx] = zero(T); end
            end

            b_max = is_numeric ? (nbins - 1) : nbins
            for b in 1:b_max
                # Accumulate histogram
                if K == 1
                    if is_numeric
                        acc1 += h∇[1,b,f,node]; acc2 += h∇[2,b,f,node]; accw += h∇[3,b,f,node]
                    else
                        acc1, acc2, accw = h∇[1,b,f,node], h∇[2,b,f,node], h∇[3,b,f,node]
                    end
                    w_l, w_r = accw, w_p - accw
                else
                    for kk in 1:(2*K+1)
                        if is_numeric
                            sums_temp[kk, temp_idx] += h∇[kk,b,f,node]
                        else
                            sums_temp[kk, temp_idx] = h∇[kk,b,f,node]
                        end
                    end
                    w_l, w_r = sums_temp[2*K+1, temp_idx], w_p - sums_temp[2*K+1, temp_idx]
                end

                (w_l < min_weight || w_r < min_weight) && continue

                skip, g_val = false, zero(T)

                # === Split gain (compile-time specialized) ===
                if L <: EvoTrees.GradientRegression || L <: EvoTrees.MLE2P || L == EvoTrees.MLogLoss
                    if K == 1
                        g_l, h_l = acc1, acc2
                        g_r, h_r = nodes_sum[1,node] - g_l, nodes_sum[2,node] - h_l
                        d_l = h_l + lambda*w_l + L2; d_l = d_l < ε ? ε : d_l
                        d_r = h_r + lambda*w_r + L2; d_r = d_r < ε ? ε : d_r
                        g_val = (g_l^2/d_l + g_r^2/d_r)/2 - gain_p
                        # Monotone constraints only for GradientRegression/MLE2P
                        if (L <: EvoTrees.GradientRegression || L <: EvoTrees.MLE2P) && constraint != 0
                            skip = (constraint == -1 && -g_l/d_l <= -g_r/d_r) ||
                                   (constraint == 1 && -g_l/d_l >= -g_r/d_r)
                        end
                    else
                        # Monotone constraints only for GradientRegression/MLE2P
                        if (L <: EvoTrees.GradientRegression || L <: EvoTrees.MLE2P) && constraint != 0
                            g1, h1 = sums_temp[1,temp_idx], sums_temp[K+1,temp_idx]
                            d1l = h1 + lambda*w_l + L2; d1l = d1l < ε ? ε : d1l
                            d1r = nodes_sum[K+1,node] - h1 + lambda*w_r + L2; d1r = d1r < ε ? ε : d1r
                            pl, pr = -g1/d1l, -(nodes_sum[1,node]-g1)/d1r
                            skip = (constraint == -1 && pl <= pr) || (constraint == 1 && pl >= pr)
                        end
                        if !skip
                            for k in 1:K
                                g_l, h_l = sums_temp[k,temp_idx], sums_temp[K+k,temp_idx]
                                g_r, h_r = nodes_sum[k,node] - g_l, nodes_sum[K+k,node] - h_l
                                d_l = h_l + lambda*w_l + L2; d_l = d_l < ε ? ε : d_l
                                d_r = h_r + lambda*w_r + L2; d_r = d_r < ε ? ε : d_r
                                g_val += (g_l^2/d_l + g_r^2/d_r)/2
                            end
                            g_val -= gain_p
                        end
                    end
                elseif L == EvoTrees.MAE || L == EvoTrees.Quantile
                    μp = nodes_sum[1,node] / w_p
                    μl, μr = acc1/w_l, (nodes_sum[1,node]-acc1)/w_r
                    d_l = 1 + lambda + L2/w_l; d_l = d_l < ε ? ε : d_l
                    d_r = 1 + lambda + L2/w_r; d_r = d_r < ε ? ε : d_r
                    g_val = abs(μl-μp)*w_l/d_l + abs(μr-μp)*w_r/d_r
                elseif L <: EvoTrees.Cred
                    μl = acc1/w_l
                    Vl, El = μl^2, acc2/w_l - μl^2; El = El < ε ? ε : El
                    gl = Vl/(Vl+El) * abs(acc1) / (1 + L2/w_l)
                    s1r, s2r = nodes_sum[1,node]-acc1, nodes_sum[2,node]-acc2
                    μr = s1r/w_r
                    Vr, Er = μr^2, s2r/w_r - μr^2; Er = Er < ε ? ε : Er
                    gr = Vr/(Vr+Er) * abs(s1r) / (1 + L2/w_r)
                    g_val = gl + gr - gain_p
                end

                if !skip && g_val > g_best
                    g_best, b_best = g_val, Int32(b)
                end
            end

            gains[f_idx, n_idx] = g_best
            bins[f_idx, n_idx] = b_best
        end
    end
end

"""
	clear_hist_kernel!(h∇, active_nodes, n_active)

Clear (zero) histogram entries for specified active nodes.
"""
@kernel function clear_hist_kernel!(h∇, @Const(active_nodes), n_active)
    idx = @index(Global, Linear)
    n_elements = size(h∇, 1) * size(h∇, 2) * size(h∇, 3)
    total = n_elements * n_active

    if idx <= total
        node_idx = (idx - 1) ÷ n_elements + 1
        element_idx = (idx - 1) % n_elements

        @inbounds node = active_nodes[node_idx]
        if node > 0
            k = element_idx % size(h∇, 1) + 1
            b = (element_idx ÷ size(h∇, 1)) % size(h∇, 2) + 1
            j = element_idx ÷ (size(h∇, 1) * size(h∇, 2)) + 1
            h∇[k, b, j, node] = zero(eltype(h∇))
        end
    end
end

"""
	clear_mask_kernel!(mask)

Clear (zero) all entries in a mask array.
"""
@kernel function clear_mask_kernel!(mask)
    idx = @index(Global)
    if idx <= length(mask)
        mask[idx] = 0
    end
end

"""
	mark_active_nodes_kernel!(mask, active_nodes)

Mark specified active nodes in a mask array by setting their entries to 1.
"""
@kernel function mark_active_nodes_kernel!(mask, @Const(active_nodes))
    idx = @index(Global)
    if idx <= length(active_nodes)
        node = active_nodes[idx]
        if node > 0 && node <= length(mask)
            mask[node] = 1
        end
    end
end

# Count raw number of observations per node for the current is/nidx mapping
@kernel function count_nodes_kernel!(node_counts, @Const(nidx), @Const(is))
    idx = @index(Global)
    if idx <= length(is)
        obs = is[idx]
        node = nidx[obs]
        if node > 0 && node <= length(node_counts)
            Atomix.@atomic node_counts[node] += 1
        end
    end
end

"""
	update_hist_gpu!(h∇, ∇, x_bin, nidx, js, is, depth, active_nodes, nodes_sum_gpu, params, feattypes, monotone_constraints, K, sums_temp, target_mask, backend)

Build histograms for active nodes by clearing previous entries and invoking the histogram kernel.
"""
function update_hist_gpu!(
    h∇, ∇, x_bin, nidx, js, is, depth, active_nodes, nodes_sum_gpu, params,
    feattypes, monotone_constraints, K, target_mask, backend,
)
    n_active = length(active_nodes)

    clear_mask_kernel!(backend)(target_mask; ndrange=length(target_mask))
    KernelAbstractions.synchronize(backend)

    mark_active_nodes_kernel!(backend)(target_mask, active_nodes; ndrange=n_active)
    KernelAbstractions.synchronize(backend)

    if n_active > 0
        clear_hist_kernel!(backend)(
            h∇, active_nodes, n_active;
            ndrange=n_active * size(h∇, 1) * size(h∇, 2) * size(h∇, 3),
        )
        KernelAbstractions.synchronize(backend)
    end

    chunk_size = 16
    n_obs_chunks = cld(length(is), chunk_size)
    num_threads = length(js) * n_obs_chunks

    hist_kernel_f! = hist_kernel!(backend)
    hist_kernel_f!(
        h∇, ∇, x_bin, nidx, js, is, K, chunk_size, target_mask;
        ndrange=num_threads,
    )
    KernelAbstractions.synchronize(backend)
end

"""
    reduce_best_split_kernel!(best_gain, best_bin, best_feat, gains, bins, js, n_feats)

Reduce per-feature gains to find the best split for each active node.
"""
@kernel function reduce_best_split_kernel!(
    best_gain,
    best_bin,
    best_feat,
    @Const(gains),
    @Const(bins),
    @Const(js),
    n_feats::Int
)
    n_idx = @index(Global)

    @inbounds if n_idx <= size(gains, 2)
        best_f_idx = 1
        best_g = gains[1, n_idx]

        for f_idx in 2:n_feats
            g = gains[f_idx, n_idx]
            if g > best_g
                best_g = g
                best_f_idx = f_idx
            end
        end

        best_gain[n_idx] = best_g
        best_bin[n_idx] = bins[best_f_idx, n_idx]
        best_feat[n_idx] = js[best_f_idx]
    end
end

