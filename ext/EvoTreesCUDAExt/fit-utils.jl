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
    K::Int,
    sums_temp::AbstractArray{T,2}
) where {T}
    n_idx = @index(Global)
    
    @inbounds if n_idx <= length(active_nodes)
        node = active_nodes[n_idx]
        if node == 0
            gains[n_idx], bins[n_idx], feats[n_idx] = T(-Inf), Int32(0), Int32(0)
        else
            nbins = size(h∇, 2)
            eps = T(1e-8)
            
            # Initialize node sums from first feature
            if !isempty(js)
                for k in 1:(2*K+1)
                    nodes_sum[k, node] = sum(@view h∇[k, :, js[1], node])
                end
            end
            
            # Pre-calculate parent gain and cache values
            w_p = nodes_sum[2*K+1, node]
            λw = lambda * w_p
            gain_p = K == 1 ? 
                nodes_sum[1, node]^2 / (nodes_sum[2, node] + λw + eps) :
                sum(nodes_sum[k, node]^2 / (nodes_sum[K+k, node] + λw/K + eps) for k in 1:K)
            
            g_best, b_best, f_best = T(-Inf), Int32(0), Int32(0)
            
            # Helper function for gain calculation
            calc_gain = (g, h, w) -> g^2 / (h + lambda * w + eps)
            calc_gain_k = (g, h, w, k) -> g^2 / (h + lambda * w / k + eps)
            
            for j_idx in 1:length(js)
                f = js[j_idx]
                is_numeric = feattypes[f]
                constraint = monotone_constraints[f]
                
                # Initialize accumulators
                acc = K == 1 ? zeros(T, 3) : (@inbounds sums_temp[:, n_idx] .= 0; sums_temp[:, n_idx])
                
                for b in 1:(nbins - 1)
                    # Update accumulator based on feature type
                    if is_numeric
                        K == 1 ? 
                            (acc .+= @view h∇[1:3, b, f, node]) :
                            (@inbounds acc .+= @view h∇[:, b, f, node])
                    else
                        K == 1 ?
                            (acc .= @view h∇[1:3, b, f, node]) :
                            (@inbounds acc .= @view h∇[:, b, f, node])
                    end
                    
                    # Get left/right weights
                    w_l = K == 1 ? acc[3] : acc[2*K+1]
                    w_r = w_p - w_l
                    
                    # Check minimum weight constraint
                    (w_l < min_weight || w_r < min_weight) && continue
                    
                    # Calculate gains based on K
                    if K == 1
                        g_l, h_l = acc[1], acc[2]
                        g_r, h_r = nodes_sum[1, node] - g_l, nodes_sum[2, node] - h_l
                        
                        # Check monotone constraint
                        if constraint != 0
                            pred_l, pred_r = -g_l/(h_l + lambda*w_l + eps), -g_r/(h_r + lambda*w_r + eps)
                            ((constraint == -1 && pred_l <= pred_r) || 
                             (constraint == 1 && pred_l >= pred_r)) && continue
                        end
                        
                        g = calc_gain(g_l, h_l, w_l) + calc_gain(g_r, h_r, w_r) - gain_p
                    else
                        # Multi-class: check constraint on first class
                        if constraint != 0
                            g_l1, h_l1 = acc[1], acc[K+1]
                            g_r1, h_r1 = nodes_sum[1, node] - g_l1, nodes_sum[K+1, node] - h_l1
                            pred_l, pred_r = -g_l1/(h_l1 + lambda*w_l/K + eps), -g_r1/(h_r1 + lambda*w_r/K + eps)
                            ((constraint == -1 && pred_l <= pred_r) || 
                             (constraint == 1 && pred_l >= pred_r)) && continue
                        end
                        
                        # Calculate total gain for all K classes
                        g = sum(calc_gain_k(acc[k], acc[K+k], w_l, K) + 
                               calc_gain_k(nodes_sum[k, node] - acc[k], 
                                          nodes_sum[K+k, node] - acc[K+k], w_r, K) 
                               for k in 1:K) - gain_p
                    end
                    
                    # Update best split if better
                    if g > g_best
                        g_best, b_best, f_best = g, Int32(b), Int32(f)
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
    feattypes, monotone_constraints, K, sums_temp=nothing
)
    backend = KernelAbstractions.get_backend(h∇)
    n_active = length(active_nodes)
    
    if sums_temp === nothing && K > 1
        sums_temp = similar(nodes_sum_gpu, 2*K+1, max(n_active, 1))
    elseif K == 1
        sums_temp = similar(nodes_sum_gpu, 1, 1)
    end
    
    h∇ .= 0
    
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
                eltype(gains)(params.lambda), eltype(gains)(params.min_weight), K, sums_temp;
                ndrange = max(n_active, 1), workgroupsize = min(256, max(64, n_active)))
    KernelAbstractions.synchronize(backend)
end

