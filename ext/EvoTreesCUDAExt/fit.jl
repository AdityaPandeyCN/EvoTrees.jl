# fit.jl - Optimized version utilizing all available resources

function EvoTrees.grow_evotree!(evotree::EvoTree{L,K}, cache::CacheGPU, params::EvoTrees.EvoTypes) where {L,K}
    EvoTrees.update_grads!(cache.∇, cache.pred, cache.y, L, params)
    
    for _ in 1:params.bagging_size
        is = EvoTrees.subsample(cache.is_in, cache.is_out, cache.mask, params.rowsample, params.rng)
        
        # Efficient feature sampling
        js_cpu = Vector{eltype(cache.js)}(undef, length(cache.js))
        EvoTrees.sample!(params.rng, cache.js_, js_cpu, replace=false, ordered=true)
        copyto!(cache.js, js_cpu)
        
        tree = EvoTrees.Tree{L,K}(params.max_depth)
        grow! = params.tree_type == :oblivious ? grow_otree! : grow_tree!
        grow!(tree, params, cache, is)
        
        push!(evotree.trees, tree)
        EvoTrees.predict!(cache.pred, tree, cache.x_bin, cache.feattypes_gpu)
    end
    
    evotree.info[:nrounds] += 1
    return nothing
end

function grow_tree!(
    tree::EvoTrees.Tree{L,K},
    params::EvoTrees.EvoTypes,
    cache::CacheGPU,
    is::CuVector
) where {L,K}
    backend = KernelAbstractions.get_backend(cache.x_bin)
    
    # Reset all GPU buffers efficiently
    reset_tree_buffers!(cache)
    cache.nidx .= 1
    
    # Initialize root node processing using a kernel
    @kernel function init_root_kernel!(anodes)
        idx = @index(Global)
        @inbounds if idx == 1
            anodes[idx] = 1
        end
    end
    
    init_root_kernel!(backend)(
        cache.anodes_gpu;
        ndrange = 1, workgroupsize = 1
    )
    KernelAbstractions.synchronize(backend)
    n_active = 1
    
    # Build histogram for root using h∇L and h∇R for split-build pattern
    build_root_histogram!(cache, is, params, backend)
    
    # Tree construction loop
    for depth in 1:params.max_depth
        !iszero(n_active) || break
        
        n_nodes_level = 2^(depth - 1)
        active_nodes = view(cache.anodes_gpu, 1:n_nodes_level)
        
        # Clear inactive nodes
        if n_active < n_nodes_level
            active_nodes[n_active+1:end] .= 0
        end
        
        # Use split-build pattern with h∇L and h∇R for efficiency
        if depth > 1
            n_active = process_level_with_subtraction!(
                cache, active_nodes, n_active, depth, is, params, backend
            )
        end
        
        # Find and apply best splits
        n_active = apply_level_splits!(
            cache, active_nodes, n_active, depth, params, backend
        )
        
        # Update node indices for next level
        if depth < params.max_depth && n_active > 0
            update_nodes_idx_kernel!(backend)(
                cache.nidx, is, cache.x_bin, 
                cache.tree_feat_gpu, cache.tree_cond_bin_gpu, 
                cache.feattypes_gpu;
                ndrange = length(is), workgroupsize = 256
            )
            KernelAbstractions.synchronize(backend)
        end
    end
    
    # Finalize tree
    finalize_tree!(tree, cache, params.eta)
    return nothing
end

# Efficiently reset all buffers
function reset_tree_buffers!(cache::CacheGPU)
    fill!(cache.tree_split_gpu, false)
    fill!(cache.tree_cond_bin_gpu, 0)
    fill!(cache.tree_feat_gpu, 0)
    fill!(cache.tree_gain_gpu, 0)
    fill!(cache.tree_pred_gpu, 0)
    fill!(cache.nodes_sum_gpu, 0)
    fill!(cache.nodes_gain_gpu, 0)
    fill!(cache.anodes_gpu, 0)
    fill!(cache.n_next_gpu, 0)
    fill!(cache.n_next_active_gpu, 0)
    fill!(cache.best_gain_gpu, 0)
    fill!(cache.best_bin_gpu, 0)
    fill!(cache.best_feat_gpu, 0)
end

# Build root histogram
function build_root_histogram!(cache, is, params, backend)
    # Use h∇ for main histogram computation
    cache.h∇ .= 0
    
    hist_kernel!(backend)(
        cache.h∇, cache.∇, cache.x_bin, cache.nidx, cache.js, is, cache.K;
        ndrange = cld(length(is), 8) * length(cache.js),
        workgroupsize = 256
    )
    KernelAbstractions.synchronize(backend)
    
    # Find best split for root
    find_best_split_from_hist_kernel!(backend)(
        cache.best_gain_gpu, cache.best_bin_gpu, cache.best_feat_gpu,
        cache.h∇, cache.nodes_sum_gpu, view(cache.anodes_gpu, 1:1),
        cache.js, cache.feattypes_gpu, cache.monotone_constraints_gpu,
        Float32(params.lambda), Float32(params.min_weight), cache.K;
        ndrange = 1, workgroupsize = 1
    )
    KernelAbstractions.synchronize(backend)
end

# Process level using histogram subtraction pattern
function process_level_with_subtraction!(cache, active_nodes, n_active, depth, is, params, backend)
    # Use h∇L and h∇R for split-build pattern to avoid recomputation
    cache.build_count .= 0
    cache.subtract_count .= 0
    
    # Separate nodes into build and subtract sets
    separate_nodes_kernel!(backend)(
        cache.build_nodes_gpu, cache.build_count,
        cache.subtract_nodes_gpu, cache.subtract_count,
        view(active_nodes, 1:n_active);
        ndrange = n_active, workgroupsize = 256
    )
    KernelAbstractions.synchronize(backend)
    
    build_count_val = Array(cache.build_count)[1]
    subtract_count_val = Array(cache.subtract_count)[1]
    
    # Build histograms for build nodes using h∇
    if build_count_val > 0
        build_nodes = view(cache.build_nodes_gpu, 1:build_count_val)
        cache.h∇ .= 0
        
        hist_kernel!(backend)(
            cache.h∇, cache.∇, cache.x_bin, cache.nidx, cache.js, is, cache.K;
            ndrange = cld(length(is), 8) * length(cache.js),
            workgroupsize = 256
        )
    end
    
    # Use h∇L for subtract nodes - compute via parent-sibling subtraction
    if subtract_count_val > 0
        subtract_nodes = view(cache.subtract_nodes_gpu, 1:subtract_count_val)
        
        subtract_hist_kernel!(backend)(
            cache.h∇L, cache.h∇, subtract_nodes;
            ndrange = subtract_count_val * prod(size(cache.h∇)[1:3]),
            workgroupsize = 256
        )
    end
    
    # Merge results back to h∇ for split finding
    if subtract_count_val > 0
        merge_histograms!(cache.h∇, cache.h∇L, cache.subtract_nodes_gpu, subtract_count_val)
    end
    
    KernelAbstractions.synchronize(backend)
    
    # Find best splits for all active nodes
    find_best_split_from_hist_kernel!(backend)(
        cache.best_gain_gpu, cache.best_bin_gpu, cache.best_feat_gpu,
        cache.h∇, cache.nodes_sum_gpu, view(active_nodes, 1:n_active),
        cache.js, cache.feattypes_gpu, cache.monotone_constraints_gpu,
        Float32(params.lambda), Float32(params.min_weight), cache.K;
        ndrange = n_active, workgroupsize = min(256, n_active)
    )
    KernelAbstractions.synchronize(backend)
    
    return n_active
end

# Apply splits and prepare next level
function apply_level_splits!(cache, active_nodes, n_active, depth, params, backend)
    cache.n_next_active_gpu .= 0
    
    apply_splits_kernel!(backend)(
        cache.tree_split_gpu, cache.tree_cond_bin_gpu, cache.tree_feat_gpu,
        cache.tree_gain_gpu, cache.tree_pred_gpu,
        cache.nodes_sum_gpu, cache.nodes_gain_gpu,
        cache.n_next_gpu, cache.n_next_active_gpu,
        cache.best_gain_gpu, cache.best_bin_gpu, cache.best_feat_gpu,
        cache.h∇, active_nodes,
        depth, params.max_depth, Float32(params.lambda), 
        Float32(params.gamma), cache.K;
        ndrange = n_active, workgroupsize = min(256, n_active)
    )
    KernelAbstractions.synchronize(backend)
    
    # Get number of active nodes for next level
    n_next = Array(cache.n_next_active_gpu)[1]
    n_active_next = min(n_next, 2^depth)
    
    # Copy next level nodes
    if n_active_next > 0
        copyto!(view(cache.anodes_gpu, 1:n_active_next), 
                view(cache.n_next_gpu, 1:n_active_next))
    end
    
    return n_active_next
end

# Optimized apply splits kernel using available resources
@kernel function apply_splits_kernel!(
    tree_split, tree_cond_bin, tree_feat, tree_gain, tree_pred,
    nodes_sum, nodes_gain,
    n_next, n_next_active,
    best_gain, best_bin, best_feat,
    h∇, active_nodes,
    depth, max_depth, lambda, gamma, K
)
    idx = @index(Global)
    @inbounds if idx <= length(active_nodes)
        node = active_nodes[idx]
        if node > 0
        
        eps = eltype(tree_pred)(1e-8)
        
        # Check split condition
        if depth < max_depth && best_gain[idx] > gamma && best_bin[idx] > 0
            # Apply split
            tree_split[node] = true
            tree_cond_bin[node] = best_bin[idx]
            tree_feat[node] = best_feat[idx]
            tree_gain[node] = best_gain[idx]
            
            # Prepare children
            child_l = node << 1
            child_r = child_l + 1
            
            # Get split parameters
            feat_idx = Int(best_feat[idx])
            bin = Int(best_bin[idx])
            
            # Accumulate child statistics from histogram
            for k in 1:(2*K+1)
                sum_l = zero(eltype(nodes_sum))
                for b in 1:bin
                    sum_l += h∇[k, b, feat_idx, node]
                end
                nodes_sum[k, child_l] = sum_l
                nodes_sum[k, child_r] = nodes_sum[k, node] - sum_l
            end
            
            # Compute gains and predictions for children
            for k in 1:K
                g_l = nodes_sum[k, child_l]
                h_l = nodes_sum[K+k, child_l]
                w_l = nodes_sum[2*K+1, child_l]
                
                g_r = nodes_sum[k, child_r]
                h_r = nodes_sum[K+k, child_r]
                w_r = nodes_sum[2*K+1, child_r]
                
                # Child gains
                nodes_gain[child_l] += g_l^2 / (h_l + lambda * w_l / K + eps)
                nodes_gain[child_r] += g_r^2 / (h_r + lambda * w_r / K + eps)
                
                # Child predictions
                tree_pred[k, child_l] = -g_l / (h_l + lambda * w_l / K + eps)
                tree_pred[k, child_r] = -g_r / (h_r + lambda * w_r / K + eps)
            end
            
            # Add children to next level
            pos = Atomix.@atomic n_next_active[1] += 2
            n_next[pos - 1] = child_l
            n_next[pos] = child_r
        else
            # Terminal node
            for k in 1:K
                g = nodes_sum[k, node]
                h = nodes_sum[K+k, node]
                w = nodes_sum[2*K+1, node]
                
                if w > 0 && h + lambda * w / K > 0
                    tree_pred[k, node] = -g / (h + lambda * w / K + eps)
                else
                    tree_pred[k, node] = zero(eltype(tree_pred))
                end
            end
        end
        end
    end
end

# Helper to merge histograms efficiently
function merge_histograms!(h∇_main, h∇_subtract, subtract_nodes, count)
    @kernel function merge_kernel!(h∇_main, h∇_subtract, nodes)
        idx = @index(Global)
        node_idx = (idx - 1) ÷ prod(size(h∇_main)[1:3]) + 1
        
        @inbounds if node_idx <= length(nodes)
            node = nodes[node_idx]
            if node > 0
                elem_idx = (idx - 1) % prod(size(h∇_main)[1:3]) + 1
                k = (elem_idx - 1) % size(h∇_main, 1) + 1
                remainder = (elem_idx - 1) ÷ size(h∇_main, 1)
                b = remainder % size(h∇_main, 2) + 1
                j = remainder ÷ size(h∇_main, 2) + 1
                
                h∇_main[k, b, j, node] = h∇_subtract[k, b, j, node]
            end
        end
    end
    
    backend = KernelAbstractions.get_backend(h∇_main)
    merge_kernel!(backend)(
        h∇_main, h∇_subtract, view(subtract_nodes, 1:count);
        ndrange = count * prod(size(h∇_main)[1:3]),
        workgroupsize = 256
    )
end

# Finalize tree
function finalize_tree!(tree, cache, eta)
    copyto!(tree.split, Array(cache.tree_split_gpu))
    copyto!(tree.cond_bin, Array(cache.tree_cond_bin_gpu))
    copyto!(tree.feat, Array(cache.tree_feat_gpu))
    copyto!(tree.gain, Array(cache.tree_gain_gpu))
    
    # Apply learning rate
    tree_pred_cpu = Array(cache.tree_pred_gpu)
    tree_pred_cpu .*= eta
    copyto!(tree.pred, tree_pred_cpu)
end

# Oblivious tree placeholder
function grow_otree!(tree::EvoTrees.Tree{L,K}, params, cache, is) where {L,K}
    @warn "Oblivious tree GPU implementation pending, using standard tree" maxlog=1
    grow_tree!(tree, params, cache, is)
end

