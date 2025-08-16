%%writefile /content/EvoTrees.jl/ext/EvoTreesCUDAExt/fit-utils.jl
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
                nidx[obs] = (node << 1) + !is_left
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
) where {T}
    i, j = @index(Global, NTuple)
    @inbounds if i <= size(x_bin, 1) && j <= length(js)
        node = nidx[i]
        if node > 0
            jdx = js[j]
            bin = x_bin[i, jdx]
            if bin > 0 && bin <= size(h∇, 2)
                Atomix.@atomic h∇[1, bin, jdx, node] += ∇[1, i]
                Atomix.@atomic h∇[2, bin, jdx, node] += ∇[2, i]
                Atomix.@atomic h∇[3, bin, jdx, node] += ∇[3, i]
            end
        end
    end
end

@kernel function hist_kernel_selective!(
    h∇::AbstractArray{T,4},
    @Const(∇),
    @Const(x_bin),
    @Const(nidx),
    @Const(js),
    @Const(target_nodes),
) where {T}
    i, j = @index(Global, NTuple)
    @inbounds if i <= size(x_bin, 1) && j <= length(js)
        node = nidx[i]
        if node > 0 && node in target_nodes
            jdx = js[j]
            bin = x_bin[i, jdx]
            if bin > 0 && bin <= size(h∇, 2)
                Atomix.@atomic h∇[1, bin, jdx, node] += ∇[1, i]
                Atomix.@atomic h∇[2, bin, jdx, node] += ∇[2, i]
                Atomix.@atomic h∇[3, bin, jdx, node] += ∇[3, i]
            end
        end
    end
end

@kernel function subtract_hist_kernel!(
    h∇::AbstractArray{T,4},
    @Const(parent_nodes),
    @Const(left_nodes),
    @Const(right_nodes),
) where {T}
    idx, feat, bin = @index(Global, NTuple)
    
    @inbounds if idx <= length(parent_nodes) && feat <= size(h∇, 3) && bin <= size(h∇, 2)
        parent = parent_nodes[idx]
        left = left_nodes[idx]
        right = right_nodes[idx]
        
        h∇[1, bin, feat, right] = h∇[1, bin, feat, parent] - h∇[1, bin, feat, left]
        h∇[2, bin, feat, right] = h∇[2, bin, feat, parent] - h∇[2, bin, feat, left]
        h∇[3, bin, feat, right] = h∇[3, bin, feat, parent] - h∇[3, bin, feat, left]
    end
end

# Simplified scan kernel without shared memory issues
@kernel function scan_hist_kernel!(
    hL::AbstractArray{T,4},
    hR::AbstractArray{T,4},
    @Const(h∇),
    @Const(active_nodes),
) where {T}
    n_idx, feat = @index(Global, NTuple)
    tid = @index(Local)
    
    nbins = size(h∇, 2)
    
    @inbounds if n_idx <= length(active_nodes) && feat <= size(h∇, 3)
        node = active_nodes[n_idx]
        
        if tid <= nbins
            # Compute prefix sum directly
            sum_g1 = zero(T)
            sum_g2 = zero(T)
            sum_g3 = zero(T)
            
            for i in 1:tid
                sum_g1 += h∇[1, i, feat, node]
                sum_g2 += h∇[2, i, feat, node]
                sum_g3 += h∇[3, i, feat, node]
            end
            
            hL[1, tid, feat, node] = sum_g1
            hL[2, tid, feat, node] = sum_g2
            hL[3, tid, feat, node] = sum_g3
            
            if tid == nbins
                hR[1, nbins, feat, node] = sum_g1
                hR[2, nbins, feat, node] = sum_g2
                hR[3, nbins, feat, node] = sum_g3
            end
        end
    end
end

@kernel function find_best_split_kernel_parallel!(
    gains::AbstractVector{T},
    bins::AbstractVector{Int32},
    feats::AbstractVector{Int32},
    @Const(hL),
    @Const(hR),
    @Const(nodes_sum),
    @Const(active_nodes),
    lambda::T,
    min_weight::T,
) where {T}
    n_idx = @index(Global)
    
    @inbounds if n_idx <= length(active_nodes)
        node = active_nodes[n_idx]
        nbins = size(hL, 2)
        nfeats = size(hL, 3)
        
        g_best = T(-Inf)
        b_best = Int32(0)
        f_best = Int32(0)
        
        p_g1 = nodes_sum[1, node]
        p_g2 = nodes_sum[2, node]
        gain_p = p_g1^2 / (p_g2 + lambda)
        
        for f in 1:nfeats
            p_w = hR[3, nbins, f, node]
            for b in 1:(nbins - 1)
                l_w = hL[3, b, f, node]
                r_w = p_w - l_w
                if l_w >= min_weight && r_w >= min_weight
                    l_g1 = hL[1, b, f, node]
                    l_g2 = hL[2, b, f, node]
                    r_g1 = p_g1 - l_g1
                    r_g2 = p_g2 - l_g2
                    gain_l = l_g1^2 / (l_g2 + lambda)
                    gain_r = r_g1^2 / (r_g2 + lambda)
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

@kernel function fill_active_nodes_kernel!(active_nodes::AbstractVector{Int32}, offset::Int32)
    idx = @index(Global)
    @inbounds active_nodes[idx] = idx + offset
end

function update_hist_gpu!(
    h∇, hL, hR, gains, bins, feats, ∇, x_bin, nidx, js, depth, active_nodes_gpu, nodes_sum_gpu, params
)
    backend = KernelAbstractions.get_backend(h∇)
    n_nodes_level = 2^(depth - 1)
    dnodes = n_nodes_level:(2^depth - 1)
    
    active_nodes = view(active_nodes_gpu, 1:n_nodes_level)
    offset = Int32(2^(depth - 1) - 1)
    kernel_fill! = fill_active_nodes_kernel!(backend)
    kernel_fill!(active_nodes, offset; ndrange = n_nodes_level)
    
    if depth == 1
        h∇[:, :, :, dnodes] .= 0
        kernel_hist! = hist_kernel!(backend)
        kernel_hist!(h∇, ∇, x_bin, nidx, js; ndrange = (size(x_bin, 1), length(js)))
    else
        left_nodes = CuArray(collect(dnodes[1:2:end]))
        right_nodes = CuArray(collect(dnodes[2:2:end]))
        parent_nodes = CuArray(collect((n_nodes_level ÷ 2):(n_nodes_level - 1)))
        
        h∇[:, :, :, left_nodes] .= 0
        
        kernel_hist_selective! = hist_kernel_selective!(backend)
        kernel_hist_selective!(h∇, ∇, x_bin, nidx, js, left_nodes; 
                                ndrange = (size(x_bin, 1), length(js)))
        
        kernel_subtract! = subtract_hist_kernel!(backend)
        kernel_subtract!(h∇, parent_nodes, left_nodes, right_nodes; 
                        ndrange = (length(parent_nodes), size(h∇, 3), size(h∇, 2)))
    end
    
    nbins = size(h∇, 2)
    # Use nbins threads per node for the scan
    kernel_scan! = scan_hist_kernel!(backend, nbins)
    kernel_scan!(hL, hR, h∇, active_nodes; ndrange = (length(active_nodes), size(h∇, 3)))
    
    # Single thread per node for finding best split
    kernel_find_split! = find_best_split_kernel_parallel!(backend)
    kernel_find_split!(
        gains, bins, feats, hL, hR, nodes_sum_gpu, active_nodes,
        params.lambda, params.min_weight;
        ndrange = length(active_nodes)
    )
    
    KernelAbstractions.synchronize(backend)
    return nothing
end

