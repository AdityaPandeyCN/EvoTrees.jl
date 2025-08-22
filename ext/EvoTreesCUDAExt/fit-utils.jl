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
    tid = @index(Local)
    gid = @index(Group)
    
    n_feats = length(js)
    n_obs = length(is)
    n_bins = size(h∇, 2)
    
    # Shared memory for local histogram accumulation
    # Size: 3 gradients × max_bins × 1 feature (process one feature at a time per block)
    shared_hist = @localmem T (3, n_bins)
    
    # Each block processes one feature
    if gid <= n_feats
        feat = js[gid]
        
        # Initialize shared memory to zero (each thread helps)
        for b in tid:@groupsize()[1]:n_bins
            shared_hist[1, b] = zero(T)
            shared_hist[2, b] = zero(T)
            shared_hist[3, b] = zero(T)
        end
        @synchronize()
        
        # Each thread processes a subset of observations
        for obs_idx in tid:@groupsize()[1]:n_obs
            if obs_idx <= n_obs
                obs = is[obs_idx]
                node = nidx[obs]
                
                @inbounds if node > 0 && node <= size(h∇, 4)
                    bin = x_bin[obs, feat]
                    if bin > 0 && bin <= n_bins
                        # Accumulate in shared memory (still needs atomics but much faster)
                        grad1 = ∇[1, obs]
                        grad2 = ∇[2, obs]
                        grad3 = ∇[3, obs]
                        Atomix.@atomic shared_hist[1, bin] += grad1
                        Atomix.@atomic shared_hist[2, bin] += grad2
                        Atomix.@atomic shared_hist[3, bin] += grad3
                    end
                end
            end
        end
        @synchronize()
        
        # Write shared memory to global (one thread per bin)
        for b in tid:@groupsize()[1]:n_bins
            if b <= n_bins
                for node in 1:size(h∇, 4)
                    @inbounds if shared_hist[1, b] != zero(T) || shared_hist[2, b] != zero(T) || shared_hist[3, b] != zero(T)
                        Atomix.@atomic h∇[1, b, feat, node] += shared_hist[1, b]
                        Atomix.@atomic h∇[2, b, feat, node] += shared_hist[2, b]
                        Atomix.@atomic h∇[3, b, feat, node] += shared_hist[3, b]
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
            nbins = size(h∇, 2)
            f_first = js[1]
            p_g1 = zero(T); p_g2 = zero(T); p_w = zero(T)
            @inbounds for b in 1:nbins
                p_g1 += h∇[1, b, f_first, node]
                p_g2 += h∇[2, b, f_first, node]
                p_w  += h∇[3, b, f_first, node]
            end
            nodes_sum[1, node] = p_g1
            nodes_sum[2, node] = p_g2
            nodes_sum[3, node] = p_w
            
            gain_p = p_g1^2 / (p_g2 + lambda * p_w + T(1e-8))
            
            g_best = T(-Inf)
            b_best = Int32(0)
            f_best = Int32(0)
            
            @inbounds for j_idx in 1:length(js)
                f = js[j_idx]
                s1 = zero(T); s2 = zero(T); s3 = zero(T)
                @inbounds for b in 1:(nbins - 1)
                    s1 += h∇[1, b, f, node]
                    s2 += h∇[2, b, f, node]
                    s3 += h∇[3, b, f, node]
                    l_w = s3
                    r_w = p_w - l_w
                    if l_w >= min_weight && r_w >= min_weight
                        l_g1 = s1
                        l_g2 = s2
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

function update_hist_gpu!(
    h∇, gains, bins, feats, ∇, x_bin, nidx, js, is, depth, active_nodes, nodes_sum_gpu, params,
    left_nodes_buf, right_nodes_buf, target_mask_buf
)
    backend = KernelAbstractions.get_backend(h∇)
    n_active = length(active_nodes)
    
    if n_active == 0
        return
    end
    
    h∇ .= 0
    
    # NEW: Launch one block per feature with 256 threads per block
    n_feats = length(js)
    hist_kernel_f! = hist_kernel!(backend)
    hist_kernel_f!(h∇, ∇, x_bin, nidx, js, is; 
                   ndrange = n_feats * 256,  # total threads
                   workgroupsize = 256)       # threads per block
    
    find_split! = find_best_split_from_hist_kernel!(backend)
    find_split!(gains, bins, feats, h∇, nodes_sum_gpu, active_nodes, js,
                eltype(gains)(params.lambda), eltype(gains)(params.min_weight);
                ndrange = n_active)
    
    KernelAbstractions.synchronize(backend)
end

