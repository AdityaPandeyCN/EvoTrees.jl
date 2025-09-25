function EvoTrees.grow_evotree!(evotree::EvoTree{L,K}, cache::CacheGPU, params::EvoTrees.EvoTypes) where {L,K}
    # Update gradients based on current ensemble predictions
    EvoTrees.update_grads!(cache.∇, cache.pred, cache.y, L, params)
    
    # Bagging: train multiple trees per boosting round for ensemble diversity
    for _ in 1:params.bagging_size
        # Row subsampling: select random subset of training samples
        is = EvoTrees.subsample(cache.is_in, cache.is_out, cache.mask, params.rowsample, params.rng)
        
        # Feature subsampling: randomly select features for this tree
        # NOTE: This happens on CPU then copies to GPU - potential optimization point
        js_cpu = Vector{eltype(cache.js)}(undef, length(cache.js))
        EvoTrees.sample!(params.rng, cache.js_, js_cpu, replace=false, ordered=true)
        copyto!(cache.js, js_cpu)  # Copy selected features to GPU
        
        # Create new tree and grow it using GPU acceleration
        tree = EvoTrees.Tree{L,K}(params.max_depth)
        grow_tree!(tree, params, cache, is)
        push!(evotree.trees, tree)
        
        # Update ensemble predictions with this new tree's contributions
        EvoTrees.predict!(cache.pred, tree, cache.x_bin, cache.feattypes_gpu)
    end
    
    evotree.info[:nrounds] += 1
    return nothing
end

function grow_otree!(
    tree::EvoTrees.Tree{L,K},
    params::EvoTrees.EvoTypes,
    cache::CacheGPU,
    is::CuVector
) where {L,K}
    # Oblivious trees not implemented on GPU yet - fall back to standard trees
    @warn "Oblivious tree GPU implementation not yet available, using standard tree" maxlog=1
    grow_tree!(tree, params, cache, is)
end

function grow_tree!(
    tree::EvoTrees.Tree{L,K},
    params::EvoTrees.EvoTypes,
    cache::CacheGPU,
    is::CuVector
) where {L,K}

    backend = KernelAbstractions.get_backend(cache.x_bin)

    # Special handling for MAE and Quantile losses
    # These require modified gradients where second derivatives are set to 1
    ∇_gpu = cache.∇
    if L <: Union{EvoTrees.MAE, EvoTrees.Quantile}
        ∇_gpu = copy(cache.∇)  # OPTIMIZATION: This full copy could be avoided
        ∇_gpu[2, :] .= 1.0f0   # Set hessian to 1 for MAE/Quantile losses
    end

    # Initialize all GPU cache arrays to clean state
    # These arrays store tree structure as it's being built
    cache.tree_split_gpu .= false      # Whether each node splits or is a leaf
    cache.tree_cond_bin_gpu .= 0       # Split condition (bin threshold)
    cache.tree_feat_gpu .= 0           # Feature used for split
    cache.tree_gain_gpu .= 0           # Information gain from split
    cache.tree_pred_gpu .= 0           # Leaf predictions
    cache.nodes_sum_gpu .= 0           # Gradient/hessian sums per node
    cache.anodes_gpu .= 0              # Active nodes at current depth
    cache.n_next_gpu .= 0              # Next level's active nodes
    cache.n_next_active_gpu .= 0       # Count of next level's active nodes
    cache.best_gain_gpu .= 0           # Best split gain found per node
    cache.best_bin_gpu .= 0            # Best split bin per node
    cache.best_feat_gpu .= 0           # Best split feature per node
    
    # Initialize node indices: all samples start at root (node 1)
    cache.nidx .= 1
    view(cache.anodes_gpu, 1:1) .= 1   # Root node is initially active

    # Map loss function types to integer IDs for GPU kernel dispatch
    # This avoids complex type dispatch within GPU kernels
    loss_id::Int32 = if L <: EvoTrees.GradientRegression
        Int32(1)
    elseif L <: EvoTrees.MLE2P
        Int32(2)
    elseif L == EvoTrees.MLogLoss
        Int32(3)
    elseif L == EvoTrees.MAE
        Int32(4)
    elseif L == EvoTrees.Quantile
        Int32(5)
    elseif L <: EvoTrees.Cred
        Int32(6)
    else
        Int32(1)
    end

    # Special case: if max_depth=1, just compute root node statistics
    if params.max_depth == 1
        reduce_root_sums_kernel!(backend)(cache.nodes_sum_gpu, ∇_gpu, is; ndrange=length(is), workgroupsize=256)
        KernelAbstractions.synchronize(backend)
    else
        # Build histograms and find best splits for root node
        # This is the core of the histogram-based tree building algorithm
        update_hist_gpu!(
            cache.h∇, cache.best_gain_gpu, cache.best_bin_gpu, cache.best_feat_gpu,
            ∇_gpu, cache.x_bin, cache.nidx, cache.js, is,
            1, view(cache.anodes_gpu, 1:1), cache.nodes_sum_gpu, params,
            cache.feattypes_gpu, cache.monotone_constraints_gpu, cache.K, loss_id, Float32(params.L2), view(cache.sums_temp_gpu, 1:(2*cache.K+1), 1:1)
        )
    end

    # Track number of active nodes (nodes that can still be split)
    n_active = params.max_depth == 1 ? 0 : 1

    # Main tree growing loop: process each depth level
    for depth in 1:params.max_depth
        !iszero(n_active) || break  # Stop if no nodes can be split
        
        view(cache.n_next_active_gpu, 1:1) .= 0  # Reset next level counter
        
        # Calculate how many nodes exist at this depth (2^(depth-1))
        n_nodes_level = 2^(depth - 1)
        active_nodes_full = view(cache.anodes_gpu, 1:n_nodes_level)
        
        # Zero out inactive nodes in the active nodes array
        if n_active < n_nodes_level
            view(cache.anodes_gpu, n_active+1:n_nodes_level) .= 0
        end

        # Views for best split information at current depth
        view_gain = view(cache.best_gain_gpu, 1:n_nodes_level)
        view_bin  = view(cache.best_bin_gpu, 1:n_nodes_level)
        view_feat = view(cache.best_feat_gpu, 1:n_nodes_level)
        
        # For depths > 1, we need to build histograms for child nodes
        if depth > 1
            active_nodes_act = view(active_nodes_full, 1:n_active)

            # Reset counters for histogram building strategy
            cache.build_nodes_gpu .= 0      # Nodes that need histograms built
            cache.subtract_nodes_gpu .= 0   # Nodes that use subtraction method
            cache.build_count .= 0
            cache.subtract_count .= 0

            # OPTIMIZATION: Separate nodes into two categories:
            # 1. "Build" nodes: compute histograms directly (usually smaller child)
            # 2. "Subtract" nodes: compute by subtracting from parent (usually larger child)
            # This reduces computation by ~50% since parent = left_child + right_child
            separate_kernel! = separate_nodes_kernel!(backend)
            separate_kernel!(
                cache.build_nodes_gpu, cache.build_count,
                cache.subtract_nodes_gpu, cache.subtract_count,
                active_nodes_act;
                ndrange=n_active, workgroupsize=256
            )
            KernelAbstractions.synchronize(backend)
            
            # Get counts from GPU to CPU for control flow
            build_count_val = Array(cache.build_count)[1]
            subtract_count_val = Array(cache.subtract_count)[1]
            
            # Build histograms directly for "build" nodes
            if build_count_val > 0
                update_hist_gpu!(
                    cache.h∇, cache.best_gain_gpu, cache.best_bin_gpu, cache.best_feat_gpu,
                    ∇_gpu, cache.x_bin, cache.nidx, cache.js, is,
                    depth, view(cache.build_nodes_gpu, 1:build_count_val), cache.nodes_sum_gpu, params,
                    cache.feattypes_gpu, cache.monotone_constraints_gpu, cache.K, loss_id, Float32(params.L2), view(cache.sums_temp_gpu, 1:(2*cache.K+1), 1:max(build_count_val,1))
                )
            end
            
            # Compute histograms by subtraction for "subtract" nodes
            # child_histogram = parent_histogram - sibling_histogram
            if subtract_count_val > 0
                subtract_hist_kernel!(backend)(
                    cache.h∇, view(cache.subtract_nodes_gpu, 1:subtract_count_val);
                    ndrange = subtract_count_val * size(cache.h∇, 1) * size(cache.h∇, 2) * size(cache.h∇, 3),
                    workgroupsize=256
                )
                KernelAbstractions.synchronize(backend)
            end
        end

        # Apply the best splits found and create child nodes
        # This kernel also computes leaf predictions for terminal nodes
        apply_splits_kernel!(backend)(
            cache.tree_split_gpu, cache.tree_cond_bin_gpu, cache.tree_feat_gpu, cache.tree_gain_gpu, cache.tree_pred_gpu,
            cache.nodes_sum_gpu,
            cache.n_next_gpu, cache.n_next_active_gpu,
            view_gain, view_bin, view_feat,
            cache.h∇,
            active_nodes_full,
            depth, params.max_depth, Float32(params.lambda), Float32(params.gamma), Float32(params.L2), cache.K;
            ndrange = max(n_active, 1), workgroupsize = 256
        )
        KernelAbstractions.synchronize(backend)
        
        # Update active nodes for next iteration
        n_active_val = Array(cache.n_next_active_gpu)[1]
        n_active = n_active_val
        if n_active > 0
            copyto!(view(cache.anodes_gpu, 1:n_active), view(cache.n_next_gpu, 1:n_active))
        end

        # Update sample-to-node assignments for next depth level
        # This tracks which node each sample belongs to after splits
        if depth < params.max_depth && n_active > 0
            update_nodes_idx_kernel!(backend)(
                cache.nidx, is, cache.x_bin, cache.tree_feat_gpu, cache.tree_cond_bin_gpu, cache.feattypes_gpu;
                ndrange = length(is), workgroupsize=256
            )
            KernelAbstractions.synchronize(backend)
        end
    end

    # Copy final tree structure from GPU back to CPU
    copyto!(tree.split, Array(cache.tree_split_gpu))
    copyto!(tree.feat, Array(cache.tree_feat_gpu))
    copyto!(tree.cond_bin, Array(cache.tree_cond_bin_gpu))
    copyto!(tree.gain, Array(cache.tree_gain_gpu))

    leaf_nodes = findall(!, tree.split)  # Find all leaf nodes

    # Special handling for MAE and Quantile losses
    # These require computing medians/quantiles which is difficult on GPU
    if L <: Union{EvoTrees.MAE, EvoTrees.Quantile}
        # OPTIMIZATION OPPORTUNITY: This transfers large arrays from GPU to CPU
        # defeating the purpose of GPU acceleration for leaf prediction
        nidx_cpu = Array(cache.nidx)
        is_cpu = Array(is)
        ∇_cpu = Array(cache.∇)
        
        # Build mapping from leaf nodes to their sample indices
        # This is needed for quantile/median computation
        leaf_map = Dict{Int, Vector{UInt32}}()
        sizehint!(leaf_map, length(leaf_nodes))
        for i in 1:length(is_cpu)
            leaf_id = nidx_cpu[is_cpu[i]]
            if !haskey(leaf_map, leaf_id)
                leaf_map[leaf_id] = UInt32[]
            end
            push!(leaf_map[leaf_id], is_cpu[i])
        end
        
        nodes_sum_cpu = Array(cache.nodes_sum_gpu)
        # Compute leaf predictions using CPU (required for quantile computation)
        for n in leaf_nodes
            node_sum_cpu_view = view(nodes_sum_cpu, :, n)
            if L <: EvoTrees.Quantile
                node_is = get(leaf_map, n, UInt32[])
                if !isempty(node_is)
                    EvoTrees.pred_leaf_cpu!(tree.pred, n, node_sum_cpu_view, L, params, ∇_cpu, node_is)
                else
                    # Fallback to MAE if no samples in leaf
                    EvoTrees.pred_leaf_cpu!(tree.pred, n, node_sum_cpu_view, EvoTrees.MAE, params)
                end
            else
                EvoTrees.pred_leaf_cpu!(tree.pred, n, node_sum_cpu_view, L, params)
            end
        end
    else
        # Standard leaf prediction computation (much simpler)
        nodes_sum_cpu = Array(cache.nodes_sum_gpu)
        for n in leaf_nodes
            node_sum_cpu_view = view(nodes_sum_cpu, :, n)
            EvoTrees.pred_leaf_cpu!(tree.pred, n, node_sum_cpu_view, L, params)
        end
    end
    
    return nothing
end

@kernel function apply_splits_kernel!(
    tree_split, tree_cond_bin, tree_feat, tree_gain, tree_pred,
    nodes_sum,
    n_next, n_next_active,
    best_gain, best_bin, best_feat,
    h∇,
    active_nodes,
    depth, max_depth, lambda, gamma, L2,
    K_val
)
    n_idx = @index(Global)  # Thread index
    node = active_nodes[n_idx]  # Node this thread is processing

    eps = eltype(tree_pred)(1e-8)  # Small epsilon to prevent division by zero

    @inbounds if depth < max_depth && best_gain[n_idx] > gamma
        # This node will split - store split information
        tree_split[node] = true
        tree_cond_bin[node] = best_bin[n_idx]
        tree_feat[node] = best_feat[n_idx]
        tree_gain[node] = best_gain[n_idx]

        # Create child nodes using bit shifting (efficient for binary tree indexing)
        # Left child = 2*parent, Right child = 2*parent + 1
        child_l, child_r = node << 1, (node << 1) + 1
        feat, bin = Int(tree_feat[node]), Int(tree_cond_bin[node])

        # Distribute gradient/hessian sums to child nodes based on split
        # Left child gets sum of bins 1 to split_bin
        # Right child gets remaining sum (parent - left)
        @inbounds for kk in 1:(2*K_val+1)  # K gradients + K hessians + 1 weight sum
            sum_val = zero(eltype(nodes_sum))
            for b in 1:bin
                sum_val += h∇[kk, b, feat, node]
            end
            nodes_sum[kk, child_l] = sum_val
            nodes_sum[kk, child_r] = nodes_sum[kk, node] - sum_val  # Subtraction trick
        end
        
        # Extract sample weights for regularization
        w_l = nodes_sum[2*K_val+1, child_l]
        w_r = nodes_sum[2*K_val+1, child_r]
        
        # Compute leaf predictions using Newton-Raphson update: -gradient/hessian
        # Formula: pred = -g / (h + lambda*w + L2) where lambda and L2 are regularization terms
        if K_val == 1  # Single output case (regression/binary classification)
            g_l = nodes_sum[1, child_l]
            h_l = nodes_sum[2, child_l]
            d_l = max(eps, h_l + lambda * w_l + L2)
            
            g_r = nodes_sum[1, child_r]
            h_r = nodes_sum[2, child_r]
            d_r = max(eps, h_r + lambda * w_r + L2)
            
            tree_pred[1, child_l] = -g_l / d_l
            tree_pred[1, child_r] = -g_r / d_r
        else  # Multi-output case (multiclass/multi-target)
            @inbounds for k in 1:K_val
                g_l = nodes_sum[k, child_l]
                h_l = nodes_sum[K_val+k, child_l]
                d_l = max(eps, h_l + lambda * w_l + L2)
                tree_pred[k, child_l] = -g_l / d_l
                
                g_r = nodes_sum[k, child_r]
                h_r = nodes_sum[K_val+k, child_r]
                d_r = max(eps, h_r + lambda * w_r + L2)
                tree_pred[k, child_r] = -g_r / d_r
            end
        end
        
        # ATOMIC OPERATION: Add child nodes to next level's active list
        # This is thread-safe since multiple threads may create children simultaneously
        idx_base = Atomix.@atomic n_next_active[1] += 2
        n_next[idx_base - 1] = child_l
        n_next[idx_base] = child_r
    else
        # This node becomes a leaf - compute final prediction
        # Same Newton-Raphson formula but for the entire node
        if K_val == 1
            g = nodes_sum[1, node]
            h = nodes_sum[2, node]
            w = nodes_sum[2*K_val+1, node]
            d = h + lambda * w + L2
            # Handle edge cases where regularization makes denominator invalid
            if w <= zero(w) || d <= zero(h)
                tree_pred[1, node] = 0.0f0
            else
                tree_pred[1, node] = -g / max(eps, d)
            end
        else
            w = nodes_sum[2*K_val+1, node]
            @inbounds for k in 1:K_val
                g = nodes_sum[k, node]
                h = nodes_sum[K_val+k, node]
                d = h + lambda * w + L2
                if w <= zero(w) || d <= zero(h)
                    tree_pred[k, node] = 0.0f0
                else
                    tree_pred[k, node] = -g / max(eps, d)
                end
            end
        end
    end
end

# Define that GPU arrays should use CuArray type
EvoTrees.device_array_type(::Type{EvoTrees.GPU}) = CuArray
