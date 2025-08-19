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

# NVIDIA-style two-phase histogram: Phase 1 - per-block shared memory histograms
@kernel function hist_shared_phase1!(
    h∇_blocks::AbstractArray{T,4},  # [gradients, bins, features, blocks]
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
    @Const(is),
    @Const(target_nodes),
    obs_per_block::Int32,
) where {T}
    # NVIDIA-optimized thread layout
    block_id = @index(Group)
    local_id = @index(Local)
    
    # Each block processes obs_per_block observations
    start_obs = (block_id - 1) * obs_per_block + 1
    end_obs = min(block_id * obs_per_block, length(is))
    
    # Shared memory for per-block histogram
    shared_hist = @localmem T (3, size(h∇_blocks, 2), length(js))
    
    # Initialize shared memory (each thread initializes some bins)
    @inbounds for idx in local_id:@groupsize()[1]:length(shared_hist)
        shared_hist[idx] = zero(T)
    end
    @synchronize
    
    # Process observations assigned to this block
    @inbounds for obs_idx in start_obs:end_obs
        obs = is[obs_idx]
        node = nidx[obs]
        
        # Only process if node is in target set
        if node > 0 && target_nodes[node] != 0
            # Process all features for this observation
            @inbounds for j_idx in 1:length(js)
                feat = js[j_idx]
                bin = x_bin[obs, feat]
                
                if bin > 0 && bin <= size(h∇_blocks, 2)
                    # Atomic add to shared memory
                    Atomix.@atomic shared_hist[1, bin, j_idx] += ∇[1, obs]
                    Atomix.@atomic shared_hist[2, bin, j_idx] += ∇[2, obs]
                    Atomix.@atomic shared_hist[3, bin, j_idx] += ∇[3, obs]
                end
            end
        end
    end
    @synchronize
    
    # Copy shared histogram to global memory
    @inbounds for idx in local_id:@groupsize()[1]:length(shared_hist)
        h∇_blocks[idx + (block_id - 1) * length(shared_hist)] = shared_hist[idx]
    end
end

# Phase 2 - accumulate per-block histograms into final result
@kernel function hist_shared_phase2!(
    h∇::AbstractArray{T,4},
    @Const(h∇_blocks),
    @Const(target_nodes),
    num_blocks::Int32,
) where {T}
    grad_idx, bin_idx, feat_idx, node_idx = @index(Global, NTuple)
    
    @inbounds if (grad_idx <= 3 && bin_idx <= size(h∇, 2) && 
                  feat_idx <= size(h∇, 3) && node_idx <= size(h∇, 4))
        
        node = node_idx  # Direct mapping for now
        if node > 0 && target_nodes[node] != 0
            total = zero(T)
            @inbounds for block_id in 1:num_blocks
                total += h∇_blocks[grad_idx, bin_idx, feat_idx, block_id]
            end
            h∇[grad_idx, bin_idx, feat_idx, node_idx] = total
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
            # compute parent sums using first feature (identical across features)
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
    
    # Set target mask for active nodes
    target_mask_buf .= 0
    fill_mask! = fill_mask_kernel!(backend)
    fill_mask!(target_mask_buf, active_nodes; ndrange = n_active)
    
    # NVIDIA two-phase approach
    # Phase 1: Per-block shared memory histograms
    obs_per_block = max(256, div(length(is), 128))  # Optimize block size
    num_blocks = Int32(ceil(length(is) / obs_per_block))
    
    # Allocate temporary per-block histogram storage
    h∇_blocks = similar(h∇, 3, size(h∇, 2), length(js), num_blocks)
    h∇_blocks .= 0
    
    # Launch phase 1 with optimal thread layout
    phase1! = hist_shared_phase1!(backend, 256)  # 256 threads per block
    phase1!(h∇_blocks, ∇, x_bin, nidx, js, is, target_mask_buf, Int32(obs_per_block); 
            ndrange = (num_blocks * 256,), workgroupsize = (256,))
    
    # Phase 2: Accumulate per-block histograms
    phase2! = hist_shared_phase2!(backend)
    phase2!(h∇, h∇_blocks, target_mask_buf, num_blocks; 
            ndrange = (3, size(h∇, 2), length(js), size(h∇, 4)))
    
    # Find best splits
    find_split! = find_best_split_from_hist_kernel!(backend)
    find_split!(gains, bins, feats, h∇, nodes_sum_gpu, active_nodes, js,
                eltype(gains)(params.lambda), eltype(gains)(params.min_weight);
                ndrange = n_active)
    
    KernelAbstractions.synchronize(backend)
end

