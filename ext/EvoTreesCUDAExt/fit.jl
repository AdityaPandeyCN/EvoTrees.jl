function EvoTrees.grow_evotree!(evotree::EvoTree{L,K}, cache::CacheGPU, params::EvoTrees.EvoTypes) where {L,K}
    EvoTrees.update_grads!(cache.∇, cache.pred, cache.y, L, params)
    
    for _ in 1:params.bagging_size
        is = EvoTrees.subsample(cache.is_in, cache.is_out, cache.mask, params.rowsample, params.rng)
        
        # Sample features on CPU, then copy indices to GPU
        js_cpu = Vector{eltype(cache.js)}(undef, length(cache.js))
        EvoTrees.sample!(params.rng, cache.js_, js_cpu, replace=false, ordered=true)
        copyto!(cache.js, js_cpu)
        
        tree = EvoTrees.Tree{L,K}(params.max_depth)
        grow_tree!(tree, params, cache, is)
        push!(evotree.trees, tree)
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

    ∇_gpu = cache.∇
    if L <: EvoTrees.MAE
        ∇_gpu = copy(cache.∇)
        ∇_gpu[2, :] .= 1.0f0
    elseif L <: EvoTrees.Quantile
        ∇_gpu = cache.∇
    end

    # Clear cache arrays - keep on GPU
    cache.tree_split_gpu .= false
    cache.tree_cond_bin_gpu .= 0
    cache.tree_feat_gpu .= 0
    cache.tree_gain_gpu .= 0
    cache.tree_pred_gpu .= 0
    cache.nodes_sum_gpu .= 0
    cache.anodes_gpu .= 0
    cache.n_next_gpu .= 0
    cache.n_next_active_gpu .= 0
    cache.best_gain_gpu .= 0
    cache.best_bin_gpu .= 0
    cache.best_feat_gpu .= 0
    
    cache.nidx .= 1
    view(cache.anodes_gpu, 1:1) .= 1

    if params.max_depth == 1
        reduce_root_sums_kernel!(backend)(cache.nodes_sum_gpu, ∇_gpu, is; ndrange=length(is), workgroupsize=256)
        KernelAbstractions.synchronize(backend)
    else
        update_hist_gpu!(
            L,
            cache.h∇, cache.best_gain_gpu, cache.best_bin_gpu, cache.best_feat_gpu,
            ∇_gpu, cache.x_bin, cache.nidx, cache.js, is,
            1, view(cache.anodes_gpu, 1:1), cache.nodes_sum_gpu, params,
            cache.feattypes_gpu, cache.monotone_constraints_gpu, cache.K, Float32(params.L2), view(cache.sums_temp_gpu, 1:(2*cache.K+1), 1:1)
        )
    end

    n_active = params.max_depth == 1 ? 0 : 1
    
    # Preallocate small CPU buffer for reading build/subtract counters
    count_buffer = zeros(Int32, 2)  # For build_count and subtract_count
    
    # Statistics tracking for subtraction usage
    total_build = 0
    total_subtract = 0

    for depth in 1:params.max_depth
        !iszero(n_active) || break
        
        view(cache.n_next_active_gpu, 1:1) .= 0
        
        n_nodes_level = 2^(depth - 1)
        active_nodes_full = view(cache.anodes_gpu, 1:n_nodes_level)
        
        if n_active < n_nodes_level
            view(cache.anodes_gpu, n_active+1:n_nodes_level) .= 0
        end

        view_gain = view(cache.best_gain_gpu, 1:n_nodes_level)
        view_bin  = view(cache.best_bin_gpu, 1:n_nodes_level)
        view_feat = view(cache.best_feat_gpu, 1:n_nodes_level)
        
        if depth > 1
            active_nodes_act = view(active_nodes_full, 1:n_active)

            cache.build_nodes_gpu .= 0
            cache.subtract_nodes_gpu .= 0
            cache.build_count .= 0
            cache.subtract_count .= 0

            separate_kernel! = separate_nodes_kernel!(backend)
            separate_kernel!(
                cache.build_nodes_gpu, cache.build_count,
                cache.subtract_nodes_gpu, cache.subtract_count,
                active_nodes_act;
                ndrange=n_active, workgroupsize=256
            )
            KernelAbstractions.synchronize(backend)
            
            # Read both counters with one host transfer
            copyto!(count_buffer, 1, cache.build_count, 1, 1)
            copyto!(count_buffer, 2, cache.subtract_count, 1, 1)
            build_count_val = count_buffer[1]
            subtract_count_val = count_buffer[2]
            
            # Print statement: Show split between build and subtract
            println("📊 Depth $depth: n_active=$n_active | BUILD=$build_count_val | SUBTRACT=$subtract_count_val")
            total_build += build_count_val
            total_subtract += subtract_count_val
            
            # Build histograms for "build" nodes
            if build_count_val > 0
                println("  ⚙️  Building histograms for $build_count_val nodes (scanning $(length(is)) observations)")
                update_hist_gpu_optimized!(
                    L,
                    cache.h∇, cache.best_gain_gpu, cache.best_bin_gpu, cache.best_feat_gpu,
                    ∇_gpu, cache.x_bin, cache.nidx, cache.js, is,
                    depth, view(cache.build_nodes_gpu, 1:build_count_val), cache.nodes_sum_gpu, params,
                    cache.feattypes_gpu, cache.monotone_constraints_gpu, cache.K, Float32(params.L2), 
                    view(cache.sums_temp_gpu, 1:(2*cache.K+1), 1:max(build_count_val,1)),
                    backend
                )
                println("  ✅ Build complete")
            end
            
            # Subtract histograms for "subtract" nodes
            if subtract_count_val > 0
                println("  ➖ Subtracting histograms for $subtract_count_val nodes (parent - sibling)")
                
                # Debug: Check histogram values before subtraction
                subtract_nodes_cpu = Array(view(cache.subtract_nodes_gpu, 1:subtract_count_val))
                println("    Subtract nodes: $subtract_nodes_cpu")
                
                subtract_hist_kernel!(backend)(
                    cache.h∇, view(cache.subtract_nodes_gpu, 1:subtract_count_val);
                    ndrange = subtract_count_val * size(cache.h∇, 1) * size(cache.h∇, 2) * size(cache.h∇, 3),
                    workgroupsize=256
                )
                KernelAbstractions.synchronize(backend)
                
                println("  ✅ Subtraction complete")
                
                # FIX: Find best splits for subtract nodes
                println("  🔍 Finding best splits for $subtract_count_val subtract nodes")
                find_split! = find_best_split_from_hist_kernel!(backend)
                find_split!(
                    L, 
                    cache.best_gain_gpu, 
                    cache.best_bin_gpu, 
                    cache.best_feat_gpu,
                    cache.h∇, 
                    cache.nodes_sum_gpu, 
                    view(cache.subtract_nodes_gpu, 1:subtract_count_val),
                    cache.js, 
                    cache.feattypes_gpu, 
                    cache.monotone_constraints_gpu,
                    eltype(cache.best_gain_gpu)(params.lambda), 
                    Float32(params.L2), 
                    eltype(cache.best_gain_gpu)(params.min_weight), 
                    cache.K, 
                    view(cache.sums_temp_gpu, 1:(2*cache.K+1), 1:max(subtract_count_val,1));
                    ndrange = max(subtract_count_val, 1),
                    workgroupsize = min(256, max(64, subtract_count_val))
                )
                KernelAbstractions.synchronize(backend)
                println("  ✅ Split finding complete for subtract nodes")
            end
            
            # Verify all nodes have splits computed
            gains_cpu = Array(view(cache.best_gain_gpu, 1:n_active))
            bins_cpu = Array(view(cache.best_bin_gpu, 1:n_active))
            feats_cpu = Array(view(cache.best_feat_gpu, 1:n_active))
            
            println("  🔎 Verification: gains range [$(minimum(gains_cpu)), $(maximum(gains_cpu))]")
            n_zero_gain = count(==(0.0f0), gains_cpu)
            n_neg_inf = count(==(-Inf32), gains_cpu)
            if n_zero_gain > 0 && n_neg_inf == 0
                println("  ⚠️  Warning: $n_zero_gain nodes have zero gain (may be legitimate)")
            end
        end

        apply_splits_kernel!(backend)(
            cache.tree_split_gpu, cache.tree_cond_bin_gpu, cache.tree_feat_gpu, cache.tree_gain_gpu, cache.tree_pred_gpu,
            cache.nodes_sum_gpu,
            cache.n_next_gpu, cache.n_next_active_gpu,
            view_gain, view_bin, view_feat,
            cache.h∇,
            active_nodes_full,
            cache.feattypes_gpu,
            depth, params.max_depth, Float32(params.lambda), Float32(params.gamma), Float32(params.gamma), Float32(params.L2), cache.K;
            ndrange = max(n_active, 1), workgroupsize = 256
        )
        KernelAbstractions.synchronize(backend)
        
        # Read next active count from device
        n_active_val = Array(cache.n_next_active_gpu)[1]
        n_active = n_active_val
        if n_active > 0
            copyto!(view(cache.anodes_gpu, 1:n_active), view(cache.n_next_gpu, 1:n_active))
        end

        if depth < params.max_depth && n_active > 0
            update_nodes_idx_kernel!(backend)(
                cache.nidx, is, cache.x_bin, cache.tree_feat_gpu, cache.tree_cond_bin_gpu, cache.feattypes_gpu;
                ndrange = length(is), workgroupsize=256
            )
            KernelAbstractions.synchronize(backend)
        end
    end

    # Print final statistics
    total_nodes = total_build + total_subtract
    if total_nodes > 0
        subtract_pct = round(100 * total_subtract / total_nodes, digits=1)
        println("🎯 SUBTRACTION STATS: Total nodes=$total_nodes | Build=$total_build | Subtract=$total_subtract ($subtract_pct%)")
        if total_subtract > 0
            println("✅ Histogram subtraction IS WORKING - saved $(total_subtract) histogram builds!")
        else
            println("⚠️  WARNING: No histogram subtraction occurred - may indicate a bug")
        end
    end

    # Copy tree structure back to CPU
    copyto!(tree.split, Array(cache.tree_split_gpu))
    copyto!(tree.feat, Array(cache.tree_feat_gpu))
    copyto!(tree.cond_bin, Array(cache.tree_cond_bin_gpu))
    copyto!(tree.gain, Array(cache.tree_gain_gpu))

    leaf_nodes = findall(!, tree.split)

    # Batch CPU transfers for MAE/Quantile
    if L <: Union{EvoTrees.MAE, EvoTrees.Quantile}
        # Batch all CPU transfers together
        cpu_data = (
            nidx = Array(cache.nidx),
            is = Array(is),
            ∇ = Array(cache.∇),
            nodes_sum = Array(cache.nodes_sum_gpu)
        )
        
        leaf_map = Dict{Int, Vector{UInt32}}()
        sizehint!(leaf_map, length(leaf_nodes))
        for i in 1:length(cpu_data.is)
            leaf_id = cpu_data.nidx[cpu_data.is[i]]
            if !haskey(leaf_map, leaf_id)
                leaf_map[leaf_id] = UInt32[]
            end
            push!(leaf_map[leaf_id], cpu_data.is[i])
        end
        
        for n in leaf_nodes
            node_sum_cpu_view = view(cpu_data.nodes_sum, :, n)
            if L <: EvoTrees.Quantile
                node_is = get(leaf_map, n, UInt32[])
                if !isempty(node_is)
                    EvoTrees.pred_leaf_cpu!(tree.pred, n, node_sum_cpu_view, L, params, cpu_data.∇, node_is)
                else
                    EvoTrees.pred_leaf_cpu!(tree.pred, n, node_sum_cpu_view, EvoTrees.MAE, params)
                end
            else
                EvoTrees.pred_leaf_cpu!(tree.pred, n, node_sum_cpu_view, L, params)
            end
        end
    else
        nodes_sum_cpu = Array(cache.nodes_sum_gpu)
        for n in leaf_nodes
            node_sum_cpu_view = view(nodes_sum_cpu, :, n)
            EvoTrees.pred_leaf_cpu!(tree.pred, n, node_sum_cpu_view, L, params)
        end
    end
    
    return nothing
end

# Histogram update using device-side clear and kernels
function update_hist_gpu_optimized!(
    ::Type{L},
    h∇, gains::AbstractVector{T}, bins::AbstractVector{Int32}, feats::AbstractVector{Int32}, 
    ∇, x_bin, nidx, js, is, depth, active_nodes, nodes_sum_gpu, params,
    feattypes, monotone_constraints, K, L2::T, sums_temp, backend
) where {T,L}
    
    n_active = length(active_nodes)
    
    if sums_temp === nothing && K > 1
        sums_temp = similar(nodes_sum_gpu, 2*K+1, max(n_active, 1))
    elseif K == 1
        sums_temp = similar(nodes_sum_gpu, 1, 1)
    end
    
    # OPTIMIZATION: Use GPU kernel to clear histogram instead of CPU loop
    if n_active > 0
        clear_hist_kernel!(backend)(
            h∇, active_nodes, n_active;
            ndrange = n_active * size(h∇, 1) * size(h∇, 2) * size(h∇, 3),
            workgroupsize = 256
        )
        KernelAbstractions.synchronize(backend)
    end
    
    # Continue with existing histogram building logic
    n_feats = length(js)
    chunk_size = 64
    n_obs_chunks = cld(length(is), chunk_size)
    num_threads = n_feats * n_obs_chunks
    
    hist_kernel_f! = hist_kernel!(backend)
    workgroup_size = min(256, max(64, num_threads))
    hist_kernel_f!(h∇, ∇, x_bin, nidx, js, is, K, chunk_size; ndrange = num_threads, workgroupsize = workgroup_size)
    KernelAbstractions.synchronize(backend)
    
    find_split! = find_best_split_from_hist_kernel!(backend)
    find_split!(L, gains, bins, feats, h∇, nodes_sum_gpu, active_nodes, js, feattypes, monotone_constraints,
                eltype(gains)(params.lambda), L2, eltype(gains)(params.min_weight), K, sums_temp;
                ndrange = max(n_active, 1), workgroupsize = min(256, max(64, n_active)))
    KernelAbstractions.synchronize(backend)
end

# Apply splits and write children/leaf predictions
@kernel function apply_splits_kernel!(
    tree_split, tree_cond_bin, tree_feat, tree_gain, tree_pred,
    nodes_sum,
    n_next, n_next_active,
    best_gain, best_bin, best_feat,
    h∇,
    active_nodes,
    feattypes,
    depth, max_depth, lambda, gamma, L2,
    K_val
)
    n_idx = @index(Global)
    
    if n_idx <= length(active_nodes)
        node = active_nodes[n_idx]

        eps = eltype(tree_pred)(1e-8)

        @inbounds if depth < max_depth && best_gain[n_idx] > gamma
            tree_split[node] = true
            tree_cond_bin[node] = best_bin[n_idx]
            tree_feat[node] = best_feat[n_idx]
            tree_gain[node] = best_gain[n_idx]

            child_l, child_r = node << 1, (node << 1) + 1
            feat, bin = Int(tree_feat[node]), Int(tree_cond_bin[node])
            is_numeric = feattypes[feat]

            @inbounds for kk in 1:(2*K_val+1)
                sum_val = zero(eltype(nodes_sum))
                if is_numeric
                    for b in 1:bin
                        sum_val += h∇[kk, b, feat, node]
                    end
                else
                    sum_val = h∇[kk, bin, feat, node]
                end
                nodes_sum[kk, child_l] = sum_val
                nodes_sum[kk, child_r] = nodes_sum[kk, node] - sum_val
            end
            
            w_l = nodes_sum[2*K_val+1, child_l]
            w_r = nodes_sum[2*K_val+1, child_r]
            
            if K_val == 1
                g_l = nodes_sum[1, child_l]
                h_l = nodes_sum[2, child_l]
                d_l = max(eps, h_l + lambda * w_l + L2)
                
                g_r = nodes_sum[1, child_r]
                h_r = nodes_sum[2, child_r]
                d_r = max(eps, h_r + lambda * w_r + L2)
                
                tree_pred[1, child_l] = -g_l / d_l
                tree_pred[1, child_r] = -g_r / d_r
            else
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
            
            idx_base = Atomix.@atomic n_next_active[1] += 2
            n_next[idx_base - 1] = child_l
            n_next[idx_base] = child_r
        else
            if K_val == 1
                g = nodes_sum[1, node]
                h = nodes_sum[2, node]
                w = nodes_sum[2*K_val+1, node]
                d = h + lambda * w + L2
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
end

