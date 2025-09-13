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

@kernel function create_mask_kernel!(mask::AbstractVector{UInt8}, @Const(nodes))
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
    @Const(mask),
    K::Int,
    is_mle2p::Bool
) where {T}
    gidx = @index(Global, Linear)
    n_feats = length(js)
    n_obs = length(is)
    obs_per_thread = 64

    total_work = cld(n_obs, obs_per_thread) * n_feats
    if gidx <= total_work
        feat_idx = (gidx - 1) % n_feats + 1
        obs_chunk_idx = (gidx - 1) ÷ n_feats
        feat = js[feat_idx]

        start_idx = obs_chunk_idx * obs_per_thread + 1
        end_idx = min(start_idx + obs_per_thread - 1, n_obs)

        @inbounds for i_obs in start_idx:end_idx
            obs = is[i_obs]
            node = nidx[obs]
            if node > 0 && node <= length(mask) && mask[node] == 1
                bin = x_bin[obs, feat]
                if bin > 0 && bin <= size(h∇, 2)
                    n_grad_hess = is_mle2p ? 5 : (2 * K + 1)
                    for k in 1:n_grad_hess
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

            if w_p >= 2 * min_weight
                parent_gain = zero(T)
                if is_mle2p && K == 2
                    g1, g2 = nodes_sum[1, node], nodes_sum[2, node]
                    h1, h2 = nodes_sum[3, node], nodes_sum[4, node]
                    parent_gain = g1^2 / (h1 + lambda * w_p + L2 + eps) + g2^2 / (h2 + lambda * w_p + L2 + eps)
                else
                    for kk in 1:K
                        g = nodes_sum[kk, node]
                        h = nodes_sum[K+kk, node]
                        parent_gain += g^2 / (h + lambda * w_p + L2 + eps)
                    end
                end

                for j_idx in 1:length(js)
                    f = js[j_idx]
                    constraint = monotone_constraints[f]
                    s_w = zero(T)
                    cum_g = MVector{MAX_K,T}(ntuple(_ -> zero(T), MAX_K))
                    cum_h = MVector{MAX_K,T}(ntuple(_ -> zero(T), MAX_K))

                    for b in 1:(nbins-1)
                        s_w += is_mle2p ? T(h∇[5, b, f, node]) : T(h∇[2*K+1, b, f, node])
                        for kk in 1:K
                            cum_g[kk] += T(h∇[kk, b, f, node])
                            cum_h[kk] += is_mle2p ? T(h∇[kk+2, b, f, node]) : T(h∇[K+kk, b, f, node])
                        end

                        if s_w >= min_weight && (w_p - s_w) >= min_weight
                            left_gain, right_gain = zero(T), zero(T)
                            predL, predR = zero(T), zero(T)
                            
                            for kk in 1:K
                                l_g, l_h = cum_g[kk], cum_h[kk]
                                r_w = w_p - s_w
                                r_g = nodes_sum[kk, node] - l_g
                                r_h = (is_mle2p ? nodes_sum[kk+2, node] : nodes_sum[K+kk, node]) - l_h
                                
                                denomL = l_h + lambda * s_w + L2 + eps
                                denomR = r_h + lambda * r_w + L2 + eps
                                
                                left_gain += l_g^2 / denomL
                                right_gain += r_g^2 / denomR

                                if constraint != 0 && (!is_mle2p || kk == 1)
                                    predL += -l_g / denomL
                                    predR += -r_g / denomR
                                end
                            end
                            
                            constraint_ok = (constraint == 0) || (constraint == -1 && predL > predR) || (constraint == 1 && predL < predR)
                            
                            if constraint_ok
                                g = left_gain + right_gain - parent_gain
                                if g > g_best
                                    g_best, b_best, f_best = g, Int32(b), Int32(f)
                                end
                            end
                        end
                    end
                end
                g_best /= 2
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
            if (node % 2) == 0 # Even node ID is a left child to be built
                pos = Atomix.@atomic build_count[1] += 1
                build_nodes[pos] = node
            else # Odd node ID is a right child to be subtracted
                pos = Atomix.@atomic subtract_count[1] += 1
                subtract_nodes[pos] = node
            end
        end
    end
end

@kernel function subtract_hist_kernel!(
    h∇::AbstractArray{T,4},
    @Const(h∇_parent),
    @Const(subtract_nodes)
) where {T}
    gidx = @index(Global, Linear)
    n_k, n_b, n_j = size(h∇, 1), size(h∇, 2), size(h∇, 3)
    n_elements_per_node = n_k * n_b * n_j
    
    node_idx = (gidx - 1) ÷ n_elements_per_node + 1

    @inbounds if node_idx <= length(subtract_nodes)
        node = subtract_nodes[node_idx] # This is the right child
        if node > 0
            parent = node >> 1
            sibling = node - 1 # The left child sibling
            
            element_idx_in_node = (gidx - 1) % n_elements_per_node + 1
            
            h∇[element_idx_in_node, node] = h∇_parent[element_idx_in_node, parent] - h∇[element_idx_in_node, sibling]
        end
    end
end

#=
ARCHITECTURAL NOTE: The calling function `grow_tree!` is responsible for managing two histogram buffers.
Before calling this function in a loop for each depth > 1, you MUST save the parent histograms:
`copyto!(h∇_parent, h∇)`
This is essential for the subtraction trick to work correctly.
=#
function update_hist_gpu!(
    h∇, h∇_parent,
    gains, bins, feats, ∇, x_bin, nidx, js, is, depth, active_nodes,
    nodes_sum_gpu, params, left_nodes_buf, right_nodes_buf, build_mask,
    feattypes, monotone_constraints, K;
    is_mae::Bool=false, is_quantile::Bool=false, is_cred::Bool=false, is_mle2p::Bool=false
)
    backend = KernelAbstractions.get_backend(h∇)
    n_active = length(active_nodes)
    
    if depth == 1
        h∇ .= 0
        build_mask .= 1 # Build for all nodes (in this case, just the root)
        hist_kernel!(backend)(
            h∇, ∇, x_bin, nidx, js, is, build_mask, K, is_mle2p;
            ndrange = cld(length(is), 64) * length(js),
            workgroupsize = 256
        )
    else
        build_count = KernelAbstractions.zeros(backend, Int32, 1)
        subtract_count = KernelAbstractions.zeros(backend, Int32, 1)
        
        separate_nodes_kernel!(backend)(
            left_nodes_buf, build_count, right_nodes_buf, subtract_count, active_nodes;
            ndrange = n_active, workgroupsize = min(256, n_active)
        )
        KernelAbstractions.synchronize(backend)
        n_build = Array(build_count)[1]
        n_subtract = Array(subtract_count)[1]

        # Build histograms for left children from scratch
        if n_build > 0
            h∇ .= 0 # Zero out current buffer before building into it
            build_mask .= 0
            build_nodes_view = view(left_nodes_buf, 1:n_build)
            create_mask_kernel!(backend)(build_mask, build_nodes_view; ndrange=n_build, workgroupsize=min(256, n_build))

            hist_kernel!(backend)(
                h∇, ∇, x_bin, nidx, js, is, build_mask, K, is_mle2p;
                ndrange = cld(length(is), 64) * length(js),
                workgroupsize = 256
            )
        else
            # If there are no left children, the buffer still needs to be cleared for the subtraction step.
            h∇ .= 0
        end

        # Calculate histograms for right children using the parent and new left histograms
        if n_subtract > 0
            subtract_nodes_view = view(right_nodes_buf, 1:n_subtract)
            n_elems_per_node = size(h∇, 1) * size(h∇, 2) * size(h∇, 3)
            subtract_hist_kernel!(backend)(
                h∇, h∇_parent, subtract_nodes_view;
                ndrange = n_subtract * n_elems_per_node,
                workgroupsize = 256
            )
        end
    end
    
    find_best_split_from_hist_kernel!(backend)(
        gains, bins, feats, h∇, nodes_sum_gpu, active_nodes, js,
        feattypes, monotone_constraints,
        eltype(gains)(params.lambda), eltype(gains)(params.min_weight), eltype(gains)(params.L2),
        K, is_mae, is_quantile, is_mle2p;
        ndrange = n_active,
        workgroupsize = min(256, n_active)
    )
    
    KernelAbstractions.synchronize(backend)
end

