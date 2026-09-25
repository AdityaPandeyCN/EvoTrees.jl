function EvoTrees.grow_evotree!(m::EvoTree{L,K}, cache::EvoTrees.CacheGPU, params::EvoTrees.EvoTypes) where {L,K}

    EvoTrees.update_grads!(cache.∇, cache.pred, cache.y, L, params, cache.group)

    for _ in 1:params.bagging_size
        is = isnothing(cache.group) ?
             EvoTrees.subsample(cache.is_full, cache.mask_cpu, cache.mask_gpu, params.rowsample, cache.rng) :
             EvoTrees.subsample(cache.is_full, cache.mask_cpu, cache.mask_gpu, params.rowsample, cache.rng, cache.group)

        js_cpu = Vector{eltype(cache.js)}(undef, length(cache.js))
        EvoTrees.sample!(cache.rng, cache.js_, js_cpu, replace=false, ordered=true)
        copyto!(cache.js, js_cpu)

        tree = EvoTrees.Tree{L,K}(params.max_depth)
        grow! = params.tree_type == :oblivious ? grow_otree! : grow_tree!
        grow!(tree, params, cache, is, js_cpu)
        push!(m.trees, tree)
        EvoTrees.predict!(cache.pred, tree, cache.x_bin, cache.feattypes_gpu)
    end

    m.info[:nrounds] += 1
    return nothing
end

"""
	grow_otree!(tree, params, cache, is, js_cpu)

Grow an oblivious tree on GPU (one shared split per depth).

Mutates:
- `tree`: the resulting tree structure and leaf predictions
- `cache`: internal GPU working buffers used during growth
"""
function grow_otree!(
    tree::EvoTrees.Tree{L,K},
    params::EvoTrees.EvoTypes,
    cache::EvoTrees.CacheGPU,
    is::CuVector,
    js_cpu::AbstractVector,
) where {L,K}
    grow_tree!(tree, params, cache, is, js_cpu, Val(true))
end

"""
	_select_binary_split!(cache, backend, L, params, active_nodes, n_feats, n_active)

Best split per active node into `best_gain` / `best_bin` / `best_feat`.
"""
function _select_binary_split!(
    cache::EvoTrees.CacheGPU, backend, ::Type{L}, params::EvoTrees.EvoTypes,
    active_nodes, n_feats::Integer, n_active::Integer,
) where {L}
    gains = view(cache.gains_per_feat_gpu, 1:n_feats, 1:n_active)
    bins = view(cache.bins_per_feat_gpu, 1:n_feats, 1:n_active)

    find_best_split_parallel_kernel!(backend)(
        L, gains, bins,
        cache.h∇, cache.nodes_sum_gpu, active_nodes,
        cache.js, cache.feattypes_gpu, cache.monotone_constraints_gpu,
        params.lambda, params.L2, params.min_weight,
        cache.K, n_feats, cache.split_sums_temp_gpu;
        ndrange=n_active * n_feats,
    )

    reduce_best_split_kernel!(backend)(
        view(cache.best_gain_gpu, 1:n_active),
        view(cache.best_bin_gpu, 1:n_active),
        view(cache.best_feat_gpu, 1:n_active),
        gains, bins, cache.js, n_feats;
        ndrange=n_active,
    )
    return nothing
end

"""
	_select_obliv_split!(cache, backend, L, params, active_nodes, n_feats, n_active, js_cpu)

One shared split for the depth, broadcast into every active-node `best_*` slot.
`js_cpu` is the host copy of `cache.js` for this tree.
"""
function _select_obliv_split!(
    cache::EvoTrees.CacheGPU, backend, ::Type{L}, params::EvoTrees.EvoTypes,
    active_nodes, n_feats::Integer, n_active::Integer, js_cpu,
) where {L}
    gains = view(cache.obliv_gains_gpu, :, 1:n_feats)
    counts = view(cache.obliv_count_gpu, :, 1:n_feats)
    gains .= 0
    counts .= 0

    accumulate_obliv_gains_kernel!(backend)(
        L, gains, counts,
        cache.h∇, cache.nodes_sum_gpu, active_nodes,
        cache.js, cache.feattypes_gpu, cache.monotone_constraints_gpu,
        params.lambda, params.L2, params.min_weight,
        cache.K, n_feats, cache.split_sums_temp_gpu;
        ndrange=n_active * n_feats,
    )

    # nbins × n_feats is small; host findmax is simpler than a device reduce.
    g_host = Array(gains)
    c_host = Array(counts)
    @inbounds for i in eachindex(g_host)
        c_host[i] == Int32(n_active) || (g_host[i] = -Inf)
    end

    best_gain, idx = findmax(g_host)
    best_bin = Int32(idx[1])
    best_feat = Int32(js_cpu[idx[2]])
    if !isfinite(best_gain)
        best_gain, best_bin, best_feat = -Inf, Int32(0), Int32(0)
    end

    broadcast_obliv_split_kernel!(backend)(
        view(cache.best_gain_gpu, 1:n_active),
        view(cache.best_bin_gpu, 1:n_active),
        view(cache.best_feat_gpu, 1:n_active),
        Float64(best_gain), best_bin, best_feat;
        ndrange=n_active,
    )
    return nothing
end

"""
	grow_tree!(tree, params, cache, is, js_cpu)
	grow_tree!(tree, params, cache, is, js_cpu, ::Val{oblivious})

Grow a tree on GPU depth by depth, following the CPU `grow_tree!`: build the smaller child's
histogram, subtract for its sibling, find splits, then partition each node's rows into its children.
Each node's rows are the slice `is[start[n]+1:start[n]+len[n]]`, tracked on the host.
Pass `Val(true)` for oblivious (shared split per depth).
"""
function grow_tree!(
    tree::EvoTrees.Tree{L,K},
    params::EvoTrees.EvoTypes,
    cache::EvoTrees.CacheGPU,
    is::CuVector,
    js_cpu::AbstractVector,
) where {L,K}
    grow_tree!(tree, params, cache, is, js_cpu, Val(false))
end

function grow_tree!(
    tree::EvoTrees.Tree{L,K},
    params::EvoTrees.EvoTypes,
    cache::EvoTrees.CacheGPU,
    is::CuVector,
    js_cpu::AbstractVector,
    ::Val{OBLIVIOUS},
) where {L,K,OBLIVIOUS}

    backend = KernelAbstractions.get_backend(cache.x_bin)

    ∇_gpu = cache.∇
    if L <: EvoTrees.MAE
        ∇_gpu = copy(cache.∇)
        ∇_gpu[(cache.K+1):(2*cache.K), :] .= 1.0f0
    end
    # 64-bit fixed point for `hist_kernel!`, following XGBoost's quantiser: a power-of-two scale with
    # 2^62 / scale >= sum(|∇|) per channel, so no partial sum of rows can overflow an Int64.
    scale = map(s -> iszero(s) ? 1.0 : 2.0^62 / nextpow(2, s), vec(Float64.(sum(abs, ∇_gpu; dims=2))))
    ∇q = round.(Int64, ∇_gpu .* scale) # once per tree, not per row × feature in the kernel

    cache.tree_split_gpu .= false
    cache.tree_cond_bin_gpu .= 0
    cache.tree_feat_gpu .= 0
    cache.tree_gain_gpu .= 0
    cache.nodes_sum_gpu .= 0

    n_feats = length(cache.js)
    start, len = zeros(Int, length(tree.split)), zeros(Int, length(tree.split))
    len[1] = length(is)
    out = similar(is)
    leaf_is = Dict{Int,Vector{UInt32}}()
    leaf_rows!(nodes) = L <: EvoTrees.Quantile && for n in nodes
        leaf_is[n] = Array(view(is, (start[n]+1):(start[n]+len[n])))
    end

    n_current = [1]
    depth = 0
    while !isempty(n_current) && depth <= params.max_depth
        if depth == params.max_depth
            leaf_rows!(n_current)
            break
        end
        n_active = length(n_current)
        active_nodes = view(cache.anodes_gpu, 1:n_active)
        copyto!(active_nodes, Int32.(n_current))

        # smaller child of each pair is built, its sibling is parent - smaller
        EvoTrees.update_hist!(cache.h∇, ∇q, cache.x_bin, cache.js, is, n_current[1:2:end], start, len, scale, backend)
        n_active > 1 && EvoTrees.subtract_hist!(cache.h∇, _to_device(backend, Int32.(n_current[2:2:end])), cache.js)
        depth == 0 && root_sum_kernel!(backend)(cache.nodes_sum_gpu, cache.h∇, cache.js; ndrange=2K + 1)

        if OBLIVIOUS
            _select_obliv_split!(cache, backend, L, params, active_nodes, n_feats, n_active, js_cpu)
        else
            _select_binary_split!(cache, backend, L, params, active_nodes, n_feats, n_active)
        end

        apply_splits_kernel!(backend)(
            cache.tree_split_gpu, cache.tree_cond_bin_gpu, cache.tree_feat_gpu,
            cache.tree_gain_gpu, cache.nodes_sum_gpu,
            view(cache.best_gain_gpu, 1:n_active),
            view(cache.best_bin_gpu, 1:n_active),
            view(cache.best_feat_gpu, 1:n_active),
            cache.h∇, active_nodes, cache.feattypes_gpu, Float32(params.gamma);
            ndrange=n_active,
        )
        split = Array(cache.tree_split_gpu)
        leaf_rows!(filter(n -> !split[n], n_current))
        n_split = filter(n -> split[n], n_current)

        # children's rows are only needed for their histograms, or for quantile leaves
        if !isempty(n_split) && (depth + 1 < params.max_depth || L <: EvoTrees.Quantile)
            is, out = EvoTrees.split_set!(
                out, is, cache.x_bin, cache.tree_feat_gpu, cache.tree_cond_bin_gpu,
                cache.feattypes_gpu, n_split, start, len, backend,
            )
        end
        n_current = Int[]
        for n in n_split
            l, r = n << 1, n << 1 + 1
            append!(n_current, len[r] >= len[l] ? (l, r) : (r, l))
        end
        depth += 1
    end

    # Copy tree to CPU and compute leaf predictions
    copyto!(tree.split, cache.tree_split_gpu)
    copyto!(tree.feat, cache.tree_feat_gpu)
    copyto!(tree.cond_bin, cache.tree_cond_bin_gpu)
    copyto!(tree.gain, cache.tree_gain_gpu)
    # An oblivious depth broadcasts its summed gain into every node, and `importance` adds
    # `tree.gain` once per split node, so each node keeps its depth's share. This is done here
    # rather than in the kernel because the gamma check reads the undivided value there.
    if OBLIVIOUS
        lo = 1
        while lo <= length(tree.gain)
            hi = min(2lo - 1, length(tree.gain))
            m = count(view(tree.split, lo:hi))
            m > 1 && (view(tree.gain, lo:hi) ./= m)
            lo <<= 1
        end
    end
    nodes_sum_cpu = Array(cache.nodes_sum_gpu)
    copyto!(tree.w, view(nodes_sum_cpu, size(nodes_sum_cpu, 1), 1:length(tree.w)))

    leaf_nodes = findall(!, tree.split)
    if L <: EvoTrees.Quantile
        ∇_cpu = Array(cache.∇)
        Threads.@threads for n in leaf_nodes
            node_is = get(leaf_is, n, UInt32[])
            if !isempty(node_is)
                EvoTrees.pred_leaf_cpu!(tree.pred, n, view(nodes_sum_cpu, :, n), L, params, ∇_cpu, node_is)
            else
                tree.pred[:, n] .= 0
            end
        end
    else
        Threads.@threads for n in leaf_nodes
            EvoTrees.pred_leaf_cpu!(tree.pred, n, view(nodes_sum_cpu, :, n), L, params)
        end
    end

    return nothing
end

"""
    apply_splits_kernel!(tree_split, tree_cond_bin, tree_feat, tree_gain, nodes_sum,
                         best_gain, best_bin, best_feat, h∇, active_nodes, feattypes, gamma)

For each active node whose `best_gain` exceeds `gamma`, record the split and write both
children's totals into `nodes_sum`: left from the parent histogram, right as `parent - left`.
"""
@kernel function apply_splits_kernel!(
    tree_split, tree_cond_bin, tree_feat, tree_gain, nodes_sum,
    @Const(best_gain), @Const(best_bin), @Const(best_feat), @Const(h∇), @Const(active_nodes), @Const(feattypes),
    gamma,
)
    n_idx = @index(Global)
    node = active_nodes[n_idx]

    @inbounds if best_gain[n_idx] > gamma
        feat, bin = Int(best_feat[n_idx]), Int(best_bin[n_idx])
        tree_split[node] = true
        tree_cond_bin[node] = bin
        tree_feat[node] = feat
        tree_gain[node] = best_gain[n_idx]

        for kk in axes(h∇, 1)
            sum_val = zero(eltype(nodes_sum))
            for b in (feattypes[feat] ? (1:bin) : (bin:bin))
                sum_val += h∇[kk, b, feat, node]
            end
            nodes_sum[kk, node<<1] = sum_val
            nodes_sum[kk, node<<1+1] = nodes_sum[kk, node] - sum_val
        end
    end
end
