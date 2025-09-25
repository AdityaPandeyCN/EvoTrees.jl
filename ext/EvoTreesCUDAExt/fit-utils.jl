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
        obs = is[gidx]  # Get observation index
        node = nidx[obs]  # Current node assignment for this observation
        if node > 0
            # Get split condition for this node
            feat = cond_feats[node]  # Which feature was used for split
            bin = cond_bins[node]    # Split threshold (bin number)
            if bin == 0
                # No split - mark as inactive
                nidx[obs] = zero(T)
            else
                feattype = feattypes[feat]
                # Different logic for numeric vs categorical features
                # Numeric: check if value <= threshold
                # Categorical: check if value == specific category
                is_left = feattype ? (x_bin[obs, feat] <= bin) : (x_bin[obs, feat] == bin)
                # Binary tree navigation: left child = 2*parent, right child = 2*parent + 1
                # Use bit shift for efficiency: node << 1 = 2*node
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
    
    # Chunked processing strategy to improve GPU memory access patterns
    # Instead of one thread per observation, we use chunks to reduce atomic contention
    total_work_items = n_feats * cld(n_obs, chunk_size)
    
    if gidx <= total_work_items
        # 2D work distribution - threads process (feature, observation_chunk) pairs
        feat_idx = (gidx - 1) % n_feats + 1      # Which feature this thread handles
        obs_chunk = (gidx - 1) ÷ n_feats         # Which chunk of observations
        
        feat = js[feat_idx]
        
        # Calculate observation range for this chunk
        start_idx = obs_chunk * chunk_size + 1
        end_idx = min(start_idx + (chunk_size - 1), n_obs)
        
        # Process all observations in this chunk for this feature
        @inbounds for obs_idx in start_idx:end_idx
            obs = is[obs_idx]
            node = nidx[obs]  # Which tree node this observation belongs to
            if node > 0 && node <= size(h∇, 4)
                bin = x_bin[obs, feat]  # Binned feature value
                if bin > 0 && bin <= size(h∇, 2)
                    # ATOMIC UPDATE: Multiple threads may update same histogram bin
                    # h∇[gradient_index, bin, feature, node] accumulates gradients
                    # Layout: K gradients + K hessians + 1 weight = 2*K+1 total
                    for k in 1:(2*K+1)
                        grad = ∇[k, obs]
                        Atomix.@atomic h∇[k, bin, feat, node] += grad
                    end
                end
            end
        end
    end
end

@kernel function reduce_root_sums_kernel!(nodes_sum, @Const(∇), @Const(is))
    idx = @index(Global)
    if idx <= length(is)
        obs = is[idx]
        n_k = size(∇, 1)  # Number of gradient components (2*K+1)
        # ATOMIC REDUCTION: All threads contribute to root node (node 1) statistics
        @inbounds for k in 1:n_k
            Atomix.@atomic nodes_sum[k, 1] += ∇[k, obs]
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
    L2::T,
    min_weight::T,
    K::Int,
    loss_id::Int32,
    sums_temp::AbstractArray{T,2}
) where {T}
    n_idx = @index(Global)
    
    @inbounds if n_idx <= length(active_nodes)
        node = active_nodes[n_idx]
        if node == 0
            # Inactive node
            gains[n_idx] = T(-Inf)
            bins[n_idx] = Int32(0)
            feats[n_idx] = Int32(0)
        else
            nbins = size(h∇, 2)
            eps = T(1e-8)
            
            # Initialize node sums by summing all histogram bins for first feature
            # This gives us the total gradients/hessians for this node
            if !isempty(js)
                first_feat = js[1]
                for k in 1:(2*K+1)
                    sum_val = zero(T)
                    for b in 1:nbins
                        sum_val += h∇[k, b, first_feat, node]
                    end
                    nodes_sum[k, node] = sum_val
                end
            end
            
            # Pre-calculate parent node statistics for efficiency
            w_p = nodes_sum[2*K+1, node]  # Total weight in parent
            λw_p = lambda * w_p
            
            # Calculate parent gain using different formulas for different loss functions
            # This is the gain we need to beat with any split
            gain_p = zero(T)
            if loss_id == 1 # GradientRegression (MSE, Poisson, etc.)
                if K == 1  # Single output
                    g_p = nodes_sum[1, node]
                    h_p = nodes_sum[2, node]
                    denom_p = h_p + λw_p + L2
                    denom_p = denom_p < eps ? eps : denom_p
                    # Newton-Raphson gain formula: g²/(2h) with regularization
                    gain_p = g_p^2 / denom_p / 2
                else  # Multi-output
                    for k in 1:K
                        g_p = nodes_sum[k, node]
                        h_p = nodes_sum[K+k, node]
                        denom_p = h_p + λw_p + L2
                        denom_p = denom_p < eps ? eps : denom_p
                        gain_p += g_p^2 / denom_p / 2
                    end
                end
            elseif loss_id == 2 # MLE2P (Maximum Likelihood Estimation, 2 parameters)
                g1 = nodes_sum[1, node]
                g2 = nodes_sum[2, node]
                h1 = nodes_sum[3, node]
                h2 = nodes_sum[4, node]
                denom1 = h1 + λw_p + L2
                denom2 = h2 + λw_p + L2
                denom1 = denom1 < eps ? eps : denom1
                denom2 = denom2 < eps ? eps : denom2
                gain_p = (g1^2 / denom1 + g2^2 / denom2) / 2
            elseif loss_id == 3 # MLogLoss (Multiclass log loss)
                for k in 1:K
                    gk = nodes_sum[k, node]
                    hk = nodes_sum[K+k, node]
                    denom = hk + λw_p + L2
                    denom = denom < eps ? eps : denom
                    gain_p += gk^2 / denom / 2
                end
            elseif loss_id == 4 || loss_id == 5 # MAE or Quantile
                gain_p = zero(T)  # Different gain calculation for these losses
            elseif loss_id == 6 # Credibility-weighted loss
                # Credibility theory gain calculation
                μp = nodes_sum[1, node] / w_p
                VHM = μp^2  # Variance of Hypothetical Means
                EVPV = nodes_sum[2, node] / w_p - VHM  # Expected Value of Process Variance
                EVPV = EVPV < eps ? eps : EVPV
                Zp = VHM / (VHM + EVPV)  # Credibility factor
                gain_p = Zp * abs(nodes_sum[1, node]) / (1 + L2 / w_p)
            end
            
            # Initialize best split tracking
            g_best = T(-Inf)
            b_best = Int32(0)
            f_best = Int32(0)
            
            # Iterate through all features to find best split
            for j_idx in 1:length(js)
                f = js[j_idx]
                is_numeric = feattypes[f]  # Numeric vs categorical feature
                constraint = monotone_constraints[f]  # Monotonicity constraint (-1, 0, 1)
                
                # Initialize cumulative sum accumulators for left split
                if K == 1
                    acc1 = zero(T)  # Gradient accumulator
                    acc2 = zero(T)  # Hessian accumulator  
                    accw = zero(T)  # Weight accumulator
                else
                    # Multi-output case: use temporary storage
                    for kk in 1:(2*K+1)
                        sums_temp[kk, n_idx] = zero(T)
                    end
                end
                
                # Try each possible split point (bin threshold)
                for b in 1:(nbins - 1)  # Don't try last bin (would create empty right child)
                    # Update left child statistics by adding current bin
                    if K == 1
                        if is_numeric
                            # Numeric features: accumulate bins 1 through b
                            acc1 += h∇[1, b, f, node]
                            acc2 += h∇[2, b, f, node]
                            accw += h∇[3, b, f, node]
                        else
                            # Categorical features: only current bin
                            acc1 = h∇[1, b, f, node]
                            acc2 = h∇[2, b, f, node]
                            accw = h∇[3, b, f, node]
                        end
                        w_l = accw
                        w_r = w_p - w_l  # Right child weight by subtraction
                        # Skip splits that create children with too few samples
                        (w_l < min_weight || w_r < min_weight) && continue
                        
                        # Calculate gain for this split based on loss function
                        g_val = zero(T)
                        if loss_id == 1  # Gradient Regression
                            g_l = acc1
                            h_l = acc2
                            g_r = nodes_sum[1, node] - g_l  # Right stats by subtraction
                            h_r = nodes_sum[2, node] - h_l
                            d_l = h_l + lambda * w_l + L2
                            d_r = h_r + lambda * w_r + L2
                            d_l = d_l < eps ? eps : d_l
                            d_r = d_r < eps ? eps : d_r
                            # Split gain = left_gain + right_gain - parent_gain
                            g_val = (g_l^2 / d_l + g_r^2 / d_r) / 2 - gain_p
                            
                            # Check monotonicity constraints
                            # Ensure predictions satisfy monotonic relationship
                            if constraint != 0
                                pred_l = -g_l / d_l
                                pred_r = -g_r / d_r
                                if (constraint == -1 && pred_l <= pred_r) || (constraint == 1 && pred_l >= pred_r)
                                    continue  # Violates constraint, skip this split
                                end
                            end
                        elseif loss_id == 2  # MLE2P
                            # Two-parameter maximum likelihood
                            g1_l = acc1
                            h1_l = acc2
                            g1_r = nodes_sum[1, node] - g1_l
                            h1_r = nodes_sum[3, node] - h1_l
                            # Handle second parameter gradients/hessians
                            g2_l = is_numeric ? (sums_temp[2, n_idx] += h∇[2, b, f, node]; sums_temp[2, n_idx]) : h∇[2, b, f, node]
                            h2_l = is_numeric ? (sums_temp[4, n_idx] += h∇[4, b, f, node]; sums_temp[4, n_idx]) : h∇[4, b, f, node]
                            g2_r = nodes_sum[2, node] - g2_l
                            h2_r = nodes_sum[4, node] - h2_l
                            d1_l = h1_l + lambda * w_l + L2
                            d1_r = h1_r + lambda * w_r + L2
                            d2_l = h2_l + lambda * w_l + L2
                            d2_r = h2_r + lambda * w_r + L2
                            d1_l = d1_l < eps ? eps : d1_l
                            d1_r = d1_r < eps ? eps : d1_r
                            d2_l = d2_l < eps ? eps : d2_l
                            d2_r = d2_r < eps ? eps : d2_r
                            g_val = ((g1_l^2 / d1_l + g2_l^2 / d2_l) + (g1_r^2 / d1_r + g2_r^2 / d2_r)) / 2 - gain_p
                            if constraint != 0
                                pred_l = -g1_l / d1_l
                                pred_r = -g1_r / d1_r
                                if (constraint == -1 && pred_l <= pred_r) || (constraint == 1 && pred_l >= pred_r)
                                    continue
                                end
                            end
                        elseif loss_id == 4 || loss_id == 5  # MAE or Quantile
                            # Different gain for MAE/Quantile - based on mean differences
                            μp = nodes_sum[1, node] / w_p
                            μl = acc1 / w_l
                            μr = (nodes_sum[1, node] - acc1) / w_r
                            d_l = 1 + lambda + L2 / w_l
                            d_r = 1 + lambda + L2 / w_r
                            d_l = d_l < eps ? eps : d_l
                            d_r = d_r < eps ? eps : d_r
                            # Gain based on weighted absolute deviations from parent mean
                            g_val = abs(μl - μp) * w_l / d_l + abs(μr - μp) * w_r / d_r
                        elseif loss_id == 6  # Credibility
                            # Credibility-weighted gain calculation
                            μp = nodes_sum[1, node] / w_p
                            VHM_p = μp^2
                            EVPV_p = nodes_sum[2, node] / w_p - VHM_p
                            EVPV_p = EVPV_p < eps ? eps : EVPV_p
                            Zp = VHM_p / (VHM_p + EVPV_p)
                            # Left child credibility calculation
                            μl = acc1 / w_l
                            VHM_l = μl^2
                            EVPV_l = acc2 / w_l - VHM_l
                            EVPV_l = EVPV_l < eps ? eps : EVPV_l
                            Zl = VHM_l / (VHM_l + EVPV_l)
                            g_l = Zl * abs(acc1) / (1 + L2 / w_l)
                            # Right child credibility calculation
                            μr = (nodes_sum[1, node] - acc1) / w_r
                            VHM_r = μr^2
                            EVPV_r = (nodes_sum[2, node] - acc2) / w_r - VHM_r
                            EVPV_r = EVPV_r < eps ? eps : EVPV_r
                            Zr = VHM_r / (VHM_r + EVPV_r)
                            g_r = Zr * abs(nodes_sum[1, node] - acc1) / (1 + L2 / w_r)
                            g_val = g_l + g_r - Zp * abs(nodes_sum[1, node]) / (1 + L2 / w_p)
                        end
                    else  # K > 1 (Multi-output case)
                        # Handle multiple outputs simultaneously
                        if is_numeric
                            # Accumulate across bins for numeric features
                            for kk in 1:(2*K+1)
                                sums_temp[kk, n_idx] += h∇[kk, b, f, node]
                            end
                        else
                            # Single bin for categorical features
                            for kk in 1:(2*K+1)
                                sums_temp[kk, n_idx] = h∇[kk, b, f, node]
                            end
                        end
                        
                        w_l = sums_temp[2*K+1, n_idx]
                        w_r = w_p - w_l
                        (w_l < min_weight || w_r < min_weight) && continue
                        
                        # Check monotonicity constraint for first output only
                        if constraint != 0 && loss_id != 3
                            g_l1 = sums_temp[1, n_idx]
                            h_l1 = sums_temp[K+1, n_idx]
                            g_r1 = nodes_sum[1, node] - g_l1
                            h_r1 = nodes_sum[K+1, node] - h_l1
                            d1_l = h_l1 + lambda * w_l + L2
                            d1_r = h_r1 + lambda * w_r + L2
                            d1_l = d1_l < eps ? eps : d1_l
                            d1_r = d1_r < eps ? eps : d1_r
                            pred_l = -g_l1 / d1_l
                            pred_r = -g_r1 / d1_r
                            if (constraint == -1 && pred_l <= pred_r) || (constraint == 1 && pred_l >= pred_r)
                                continue
                            end
                        end
                        
                        # Calculate gain summed across all outputs
                        g_val = zero(T)
                        for k in 1:K
                            g_l = sums_temp[k, n_idx]
                            h_l = sums_temp[K+k, n_idx]
                            g_r = nodes_sum[k, node] - g_l
                            h_r = nodes_sum[K+k, node] - h_l
                            d_l = h_l + lambda * w_l + L2
                            d_r = h_r + lambda * w_r + L2
                            d_l = d_l < eps ? eps : d_l
                            d_r = d_r < eps ? eps : d_r
                            g_val += (g_l^2 / d_l + g_r^2 / d_r) / 2
                        end
                        g_val -= gain_p
                    end
                    
                    # Update best split if this one is better
                    if g_val > g_best
                        g_best = g_val
                        b_best = Int32(b)
                        f_best = Int32(f)
                    end
                end
            end
            
            # Store results for this node
            gains[n_idx] = g_best
            bins[n_idx] = b_best
            feats[n_idx] = f_best
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
        # OPTIMIZATION STRATEGY: Separate nodes into two categories
        # - Build nodes: compute histograms directly (typically smaller child)
        # - Subtract nodes: compute by parent - sibling subtraction (typically larger child)
        # This reduces histogram computation by ~50% since parent = left + right
        if idx % 2 == 1  # Odd indices go to build
            pos = Atomix.@atomic build_count[1] += 1
            build_nodes[pos] = node
        else  # Even indices go to subtract
            pos = Atomix.@atomic subtract_count[1] += 1
            subtract_nodes[pos] = node
        end
    end
end

@kernel function subtract_hist_kernel!(h∇, @Const(subtract_nodes))
    gidx = @index(Global)

    # 4D array indexing with manual linearization
    # h∇ has dimensions [gradient_component, bin, feature, node]
    n_k = size(h∇, 1)  # Number of gradient components (2*K+1)
    n_b = size(h∇, 2)  # Number of bins
    n_j = size(h∇, 3)  # Number of features
    n_elements_per_node = n_k * n_b * n_j
    
    # Convert linear thread index back to 4D coordinates
    # Each node gets n_elements_per_node threads to process all its histogram elements
    node_idx = (gidx - 1) ÷ n_elements_per_node + 1
    
    if node_idx <= length(subtract_nodes)
        # Extract 4D coordinates from linear index
        remainder = (gidx - 1) % n_elements_per_node
        j = remainder ÷ (n_k * n_b) + 1        # Feature index
        
        remainder = remainder % (n_k * n_b)
        b = remainder ÷ n_k + 1                # Bin index
        
        k = remainder % n_k + 1                # Gradient component index
        
        @inbounds node = subtract_nodes[node_idx]
        
        if node > 0
            # Binary tree relationships for histogram subtraction
            parent = node >> 1      # Parent node (bit shift right = divide by 2)
            sibling = node ⊻ 1      # Sibling node (XOR with 1 flips last bit)
            
            # Subtract sibling histogram from parent to get this node's histogram
            # This works because: parent_hist = left_child_hist + right_child_hist
            # Therefore: child_hist = parent_hist - sibling_hist
            @inbounds h∇[k, b, j, node] = h∇[k, b, j, parent] - h∇[k, b, j, sibling]
        end
    end
end

function update_hist_gpu!(
    h∇, gains::AbstractVector{T}, bins::AbstractVector{Int32}, feats::AbstractVector{Int32}, ∇, x_bin, nidx, js, is, depth, active_nodes, nodes_sum_gpu, params,
    feattypes, monotone_constraints, K, loss_id::Int32, L2::T, sums_temp=nothing
) where {T}
    backend = KernelAbstractions.get_backend(h∇)
    n_active = length(active_nodes)
    
    # MEMORY MANAGEMENT: Ensure temporary arrays are properly sized
    if sums_temp === nothing && K > 1
        sums_temp = similar(nodes_sum_gpu, 2*K+1, max(n_active, 1))
    elseif K == 1
        sums_temp = similar(nodes_sum_gpu, 1, 1)
    end
    
    # OPTIMIZATION: Clear only the histogram bins for nodes being processed
    # This avoids clearing the entire 4D histogram array
    if n_active > 0
        anodes_cpu = Array(view(active_nodes, 1:n_active))
        for node in anodes_cpu
            if node > 0
                @views h∇[:, :, :, node] .= zero(eltype(h∇))
            end
        end
        KernelAbstractions.synchronize(backend)
    end
    
    # Dynamic workgroup sizing for optimal GPU utilization
    n_feats = length(js)
    chunk_size = 64  # Process observations in chunks to reduce atomic contention
    n_obs_chunks = cld(length(is), chunk_size)
    num_threads = n_feats * n_obs_chunks
    
    hist_kernel_f! = hist_kernel!(backend)
    # Adaptive workgroup size based on problem size
    workgroup_size = min(256, max(64, num_threads))
    hist_kernel_f!(h∇, ∇, x_bin, nidx, js, is, K, chunk_size; ndrange = num_threads, workgroupsize = workgroup_size)
    KernelAbstractions.synchronize(backend)
    
    # Find best splits from the computed histograms
    find_split! = find_best_split_from_hist_kernel!(backend)
    find_split!(gains, bins, feats, h∇, nodes_sum_gpu, active_nodes, js, feattypes, monotone_constraints,
                eltype(gains)(params.lambda), L2, eltype(gains)(params.min_weight), K, loss_id, sums_temp;
                ndrange = max(n_active, 1), workgroupsize = min(256, max(64, n_active)))
    KernelAbstractions.synchronize(backend)
end
