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

@kernel function hist_per_block_simple!(
    block_hists::AbstractArray{T,5},
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
    @Const(is),
    obs_per_block::Int32,
) where {T}
    block_id = @index(Group)
    thread_id = @index(Local)
    
    start_obs = (block_id - 1) * obs_per_block + 1
    end_obs = min(block_id * obs_per_block, length(is))
    
    @inbounds for obs_idx in (start_obs + thread_id - 1):@groupsize()[1]:end_obs
        if obs_idx <= length(is)
            obs = is[obs_idx]
            node = nidx[obs]
            if node > 0 && node <= size(block_hists, 4)
                @inbounds for j_idx in 1:length(js)
                    feat = js[j_idx]
                    bin = x_bin[obs, feat]
                    if (bin > 0 && bin <= size(block_hists, 2) && 
                        feat <= size(block_hists, 3))
                        Atomix.@atomic block_hists[1, bin, feat, node, block_id] += ∇[1, obs]
                        Atomix.@atomic block_hists[2, bin, feat, node, block_id] += ∇[2, obs]
                        Atomix.@atomic block_hists[3, bin, feat, node, block_id] += ∇[3, obs]
                    end
                end
            end
        end
    end
end

@kernel function accumulate_blocks_simple!(
    h∇::AbstractArray{T,4},
    @Const(block_hists),
) where {T}
    grad, bin, feat, node = @index(Global, NTuple)
    
    @inbounds if (grad <= size(h∇, 1) && bin <= size(h∇, 2) && 
                  feat <= size(h∇, 3) && node <= size(h∇, 4))
        total = zero(T)
        @inbounds for block_id in 1:size(block_hists, 5)
            total += block_hists[grad, bin, feat, node, block_id]
        end
        h∇[grad, bin, feat, node] = total
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
    h∇, h∇_parent, gains, bins, feats, ∇, x_bin, nidx, js, is, depth, active_nodes, nodes_sum_gpu, params,
    left_nodes_buf, right_nodes_buf, target_mask_buf
)
    backend = KernelAbstractions.get_backend(h∇)
    n_active = length(active_nodes)
    
    if n_active == 0
        return
    end
    
    max_blocks = 32
    obs_per_block = Int32(max(256, div(length(is), max_blocks)))
    num_blocks = min(max_blocks, Int32(ceil(length(is) / obs_per_block)))
    
    block_hists = similar(h∇, size(h∇, 1), size(h∇, 2), size(h∇, 3), size(h∇, 4), num_blocks)
    block_hists .= 0
    
    phase1! = hist_per_block_simple!(backend)
    phase1!(block_hists, ∇, x_bin, nidx, js, is, obs_per_block; 
            ndrange = (num_blocks * 256,), workgroupsize = (256,))
    
    phase2! = accumulate_blocks_simple!(backend)
    phase2!(h∇, block_hists; ndrange = size(h∇))
    
    find_split! = find_best_split_from_hist_kernel!(backend)
    find_split!(gains, bins, feats, h∇, nodes_sum_gpu, active_nodes, js,
                eltype(gains)(params.lambda), eltype(gains)(params.min_weight);
                ndrange = n_active)
    
    KernelAbstractions.synchronize(backend)
end

