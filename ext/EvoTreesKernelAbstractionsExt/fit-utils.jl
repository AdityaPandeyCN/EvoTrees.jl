using KernelAbstractions
using Atomix

# Rows of a node read by one histogram workgroup, and by one partition workgroup (one row per thread).
const HIST_ROWS = 4096
const SPLIT_ROWS = 256
# Slots in a workgroup's local histogram tile (32 KB: one 64-bit integer per slot, as two UInt32 words).
const HIST_LOCAL = 4096

"""
    hist_kernel!(h∇, ∇, x_bin, js, is, items, scale, ck, nf)

One workgroup per `(item, tile)`. Item `items[:, i] = (node, lo, hi)` is the rows `is[lo+1:hi]`
of `node`. A tile is `nf` features by `ck` of the `2K+1` channels. The workgroup sums its rows in
local memory, then adds the tile into `h∇` once (two-phase privatized histogram). `∇` is the
gradients in 64-bit fixed point, `round(∇ * scale)`; integer sums are exact and independent of
order. Each 64-bit add is done as two native 32-bit atomics with the carry propagated, as in
XGBoost's `AtomicAdd64As32`.
"""
@kernel function hist_kernel!(h∇, @Const(∇), @Const(x_bin), @Const(js), @Const(is), @Const(items), @Const(scale), ck::Int, nf::Int)
    sh = @localmem UInt32 (2 * HIST_LOCAL,)
    tid = @index(Local, Linear)
    wg = @groupsize()[1]
    for i in tid:wg:(2*HIST_LOCAL)
        sh[i] = 0
    end
    @synchronize
    # indices are re-read after each @synchronize: the CPU backend does not carry locals across it
    tid = @index(Local, Linear)
    wg = @groupsize()[1]
    it, t = @index(Group, NTuple)
    NK, NB, nct = size(h∇, 1), size(h∇, 2), cld(size(h∇, 1), ck)
    ct, ft = (t - 1) % nct, (t - 1) ÷ nct
    @inbounds for r in (items[2, it]+tid):wg:items[3, it]
        obs = is[r]
        for f in 1:nf
            jj = ft * nf + f
            jj > length(js) && break
            bin = x_bin[obs, js[jj]]
            for c in 1:ck
                k = ct * ck + c
                k > NK && break
                i, x = c + ck * (bin - 1 + NB * (f - 1)), reinterpret(UInt64, ∇[k, obs])
                lo, hi = x % UInt32, (x >> 32) % UInt32
                low = Atomix.@atomic sh[2i-1] += lo # returns the new low word
                Atomix.@atomic sh[2i] += hi + UInt32(low < lo) # plus the carry out of the low word
            end
        end
    end
    @synchronize
    tid = @index(Local, Linear)
    wg = @groupsize()[1]
    it, t = @index(Group, NTuple)
    NK, NB, nct = size(h∇, 1), size(h∇, 2), cld(size(h∇, 1), ck)
    ct, ft = (t - 1) % nct, (t - 1) ÷ nct
    @inbounds for i in tid:wg:(ck*NB*nf)
        c, b, f = (i - 1) % ck + 1, (i - 1) ÷ ck % NB + 1, (i - 1) ÷ (ck * NB) + 1
        jj, k, v = ft * nf + f, ct * ck + c, reinterpret(Int64, UInt64(sh[2i]) << 32 | UInt64(sh[2i-1]))
        if jj <= length(js) && k <= NK && !iszero(v)
            Atomix.@atomic h∇[k, b, js[jj], items[1, it]] += v / scale[k]
        end
    end
end

@kernel function clear_hist_kernel!(h, @Const(js), @Const(nodes))
    i, jj, nn = @index(Global, NTuple)
    @inbounds h[i, js[jj], nodes[nn]] = 0
end

# Row slices `(node, lo, hi)` of `step` rows covering each node in `nodes`, one column per slice.
function _slices(nodes, start, len, step, nrows)
    plan, c = zeros(Int32, nrows, sum(n -> cld(len[n], step), nodes; init=0)), 0
    for n in nodes, lo in start[n]:step:(start[n]+len[n]-1)
        c += 1
        plan[1, c], plan[2, c], plan[3, c] = n, lo, min(lo + step, start[n] + len[n])
    end
    return plan
end

"""
    update_hist!(h∇, ∇, x_bin, js, is, nodes, start, len, scale, backend)

Build the histograms of `nodes`, whose rows are `is[start[n]+1:start[n]+len[n]]`.
"""
function EvoTrees.update_hist!(h∇, ∇, x_bin, js, is, nodes, start, len, scale, backend)
    NK, NB = size(h∇, 1), size(h∇, 2)
    ck = min(NK, HIST_LOCAL ÷ NB)
    nf = ck == NK ? HIST_LOCAL ÷ (NK * NB) : 1
    h = reshape(h∇, :, size(h∇, 3), size(h∇, 4))
    clear_hist_kernel!(backend)(h, js, _to_device(backend, Int32.(nodes)); ndrange=(size(h, 1), length(js), length(nodes)))
    items = _slices(nodes, start, len, HIST_ROWS, 3)
    hist_kernel!(backend, (256, 1))(
        h∇, ∇, x_bin, js, is, _to_device(backend, items), scale, ck, nf;
        ndrange=(256 * size(items, 2), cld(NK, ck) * cld(length(js), nf)),
    )
end

@inline _goes_left(x_bin, i, f, b, numeric) = numeric ? x_bin[i, f] <= b : x_bin[i, f] == b

# One workgroup per slice `(node, lo, hi, left_at, right_at)`, one row per thread. Rows going left set
# a bit in a local mask. The count pass writes the slice's left count; the scatter pass writes each row
# after the left or right rows before it, so both children keep row order.
@kernel function split_kernel!(out, lefts, @Const(is), @Const(x_bin), @Const(plan), @Const(feat), @Const(cond_bin), @Const(feattypes), scatter::Bool)
    bits = @localmem UInt32 (SPLIT_ROWS ÷ 32,)
    tid = @index(Local, Linear)
    tid <= SPLIT_ROWS ÷ 32 && (bits[tid] = 0)
    @synchronize
    tid = @index(Local, Linear)
    s = @index(Group, Linear)
    @inbounds begin
        n, p = plan[1, s], plan[2, s] + tid
        if p <= plan[3, s] && _goes_left(x_bin, is[p], feat[n], cond_bin[n], feattypes[feat[n]])
            Atomix.@atomic bits[(tid-1)>>5+1] += UInt32(1) << ((tid - 1) & 31)
        end
    end
    @synchronize
    tid = @index(Local, Linear)
    s = @index(Group, Linear)
    w, b = (tid - 1) >> 5 + 1, (tid - 1) & 31
    l = 0
    @inbounds for k in 1:(scatter ? w - 1 : SPLIT_ROWS ÷ 32)
        l += count_ones(bits[k])
    end
    @inbounds if !scatter
        tid == 1 && (lefts[s] = l)
    elseif plan[2, s] + tid <= plan[3, s]
        l += count_ones(bits[w] & ((UInt32(1) << b) - UInt32(1)))
        at = isodd(bits[w] >> b) ? plan[4, s] + l : plan[5, s] + tid - 1 - l
        out[at+1] = is[plan[2, s]+tid]
    end
end

"""
    split_set!(out, is, x_bin, feat, cond_bin, feattypes, nodes, start, len, backend)

Stable partition of the rows of each node in `nodes` into its left then right child, written to
`out`, like the CPU `split_set!`: count left rows per slice, prefix-sum the counts on the host,
then scatter. Sets `start` and `len` of the children.
"""
function EvoTrees.split_set!(out, is, x_bin, feat, cond_bin, feattypes, nodes, start, len, backend)
    plan = _slices(nodes, start, len, SPLIT_ROWS, 5)
    plan_gpu = _to_device(backend, plan)
    lefts_gpu = KA.allocate(backend, Int32, size(plan, 2))
    split_kernel!(backend, SPLIT_ROWS)(out, lefts_gpu, is, x_bin, plan_gpu, feat, cond_bin, feattypes, false; ndrange=SPLIT_ROWS * size(plan, 2))
    lefts = Array(lefts_gpu)
    for n in nodes
        len[2n] = 0
    end
    for c in axes(plan, 2)
        len[2plan[1, c]] += lefts[c]
    end
    l, r = copy(start), copy(start)
    for c in axes(plan, 2)
        n = plan[1, c]
        plan[4, c], plan[5, c] = l[n], r[n] + len[2n]
        l[n] += lefts[c]
        r[n] += plan[3, c] - plan[2, c] - lefts[c]
    end
    for n in nodes
        start[2n], start[2n+1] = start[n], start[n] + len[2n]
        len[2n+1] = len[n] - len[2n]
    end
    split_kernel!(backend, SPLIT_ROWS)(out, lefts_gpu, is, x_bin, _to_device(backend, plan), feat, cond_bin, feattypes, true; ndrange=SPLIT_ROWS * size(plan, 2))
    return out, is
end

"""
	subtract_hist_kernel!(h, js, nodes)

Sibling subtraction over `h` reshaped to `(2K+1)*nbins × nfeats × nnodes`.
The 3D ndrange drops the per-element index decode.
"""
@kernel function subtract_hist_kernel!(h, @Const(js), @Const(nodes))
    i, jj, nn = @index(Global, NTuple)
    @inbounds begin
        n = nodes[nn]
        j = js[jj]
        h[i, j, n] = h[i, j, n>>1] - h[i, j, n⊻1]
    end
end

function EvoTrees.subtract_hist!(h∇::GPUArraysCore.AbstractGPUArray{<:Any,4}, nodes, js)
    backend = get_backend(h∇)
    h = reshape(h∇, :, size(h∇, 3), size(h∇, 4))
    subtract_hist_kernel!(backend)(h, js, nodes; ndrange=(size(h, 1), length(js), length(nodes)))
end

# Root totals: every row falls in one bin of each feature, so any one feature's bins sum to the node.
@kernel function root_sum_kernel!(nodes_sum, @Const(h∇), @Const(js))
    k = @index(Global)
    s = zero(eltype(nodes_sum))
    @inbounds for b in axes(h∇, 2)
        s += h∇[k, b, js[1], 1]
    end
    @inbounds nodes_sum[k, 1] = s
end

"""
    check_monotone(L, constraint, g_l, h_l, g_r, h_r, w_l, w_r, lambda, L2, ε) -> Bool

Return `true` if the split violates `constraint` and should be skipped.
Always `false` when `constraint == 0`, and for losses that do not support
monotone constraints.
"""
@inline function check_monotone(::Type{L}, constraint, g_l, h_l, g_r, h_r, w_l, w_r, lambda, L2, ε) where {L<:EvoTrees.GradientRegression}
    constraint == 0 && return false
    d_l = max(h_l + lambda * w_l + L2, ε)
    d_r = max(h_r + lambda * w_r + L2, ε)
    pred_l = -g_l / d_l
    pred_r = -g_r / d_r
    return (constraint == -1 && pred_l <= pred_r) || (constraint == 1 && pred_l >= pred_r)
end

@inline function check_monotone(::Type{L}, constraint, g_l, h_l, g_r, h_r, w_l, w_r, lambda, L2, ε) where {L<:EvoTrees.MLE2P}
    constraint == 0 && return false
    d_l = max(h_l + lambda * w_l + L2, ε)
    d_r = max(h_r + lambda * w_r + L2, ε)
    pred_l = -g_l / d_l
    pred_r = -g_r / d_r
    return (constraint == -1 && pred_l <= pred_r) || (constraint == 1 && pred_l >= pred_r)
end

@inline check_monotone(::Type{EvoTrees.MLogLoss}, constraint, args...) = false
@inline check_monotone(::Type{EvoTrees.MAE}, constraint, args...) = false
@inline check_monotone(::Type{<:EvoTrees.Quantile}, constraint, args...) = false
@inline check_monotone(::Type{L}, constraint, args...) where {L<:EvoTrees.Cred} = false

"""
    _eval_split_bin(L, h∇, nodes_sum, node, f, b, ...) -> (gain, acc1, acc2, accw)

Advance left-side histogram sums to bin `b` and return net split gain
(`split_gain - gain_p`).

`K == 1` keeps `g`, `h`, `w` in `acc1`, `acc2`, `accw`. `K > 1` writes column
`temp_idx` of `sums_temp`. Ineligible bins return `-Inf` but still update the
accumulators.
"""
Base.@propagate_inbounds function _eval_split_bin(
    ::Type{L},
    h∇,
    nodes_sum,
    node,
    f,
    b,
    is_numeric,
    constraint,
    acc1::T,
    acc2::T,
    accw::T,
    w_p::T,
    gain_p::T,
    lambda::T,
    L2::T,
    min_weight::T,
    K::Int,
    sums_temp,
    temp_idx::Int,
    ε::T,
) where {T,L}
    if K == 1
        acc1, acc2, accw = EvoTrees._accumulate_hist_k1(
            h∇, f, b, node, is_numeric, acc1, acc2, accw,
        )
        w_l, w_r = accw, w_p - accw
        (w_l <= min_weight || w_r <= min_weight) && return (T(-Inf), acc1, acc2, accw)
        check_monotone(
            L, constraint,
            acc1, acc2,
            nodes_sum[1, node] - acc1, nodes_sum[2, node] - acc2,
            w_l, w_r, lambda, L2, ε,
        ) && return (T(-Inf), acc1, acc2, accw)
        ∑ = (nodes_sum[1, node], nodes_sum[2, node], nodes_sum[3, node])
        ∑L = (acc1, acc2, accw)
        gain = EvoTrees.split_gain(L, ∑, ∑L, w_l, w_r, lambda, L2, ε) - gain_p
        return (gain, acc1, acc2, accw)
    else
        EvoTrees._acc_left!(sums_temp, temp_idx, h∇, f, b, node, 2 * K + 1, is_numeric)
        w_l = sums_temp[2*K+1, temp_idx]
        w_r = w_p - w_l
        (w_l <= min_weight || w_r <= min_weight) && return (T(-Inf), acc1, acc2, accw)
        check_monotone(
            L, constraint,
            sums_temp[1, temp_idx], sums_temp[K+1, temp_idx],
            nodes_sum[1, node] - sums_temp[1, temp_idx],
            nodes_sum[K+1, node] - sums_temp[K+1, temp_idx],
            w_l, w_r, lambda, L2, ε,
        ) && return (T(-Inf), acc1, acc2, accw)
        gain = EvoTrees.split_gain(
            L, nodes_sum, node, sums_temp, temp_idx, K, w_l, w_r, lambda, L2, ε,
        ) - gain_p
        return (gain, acc1, acc2, accw)
    end
end

"""
    find_best_split_parallel_kernel!(L, gains, bins, h∇, nodes_sum, active_nodes, js, feattypes, monotone_constraints, lambda, L2, min_weight, K, n_feats, sums_temp)

One thread per `(active node, feature)`. Write the best bin into `gains[f, n]`
and `bins[f, n]` (`0` if none).
"""
@kernel function find_best_split_parallel_kernel!(
    ::Type{L},
    gains::AbstractMatrix{T},
    bins::AbstractMatrix{Int32},
    @Const(h∇),
    @Const(nodes_sum),
    @Const(active_nodes),
    @Const(js),
    @Const(feattypes),
    @Const(monotone_constraints),
    lambda::T,
    L2::T,
    min_weight::T,
    K::Int,
    n_feats::Int,
    sums_temp::AbstractArray{T,2},
) where {T,L}
    gidx = @index(Global)
    n_active = length(active_nodes)
    ε = T(1e-8)

    @inbounds if gidx <= n_active * n_feats
        n_idx = (gidx - 1) ÷ n_feats + 1
        f_idx = (gidx - 1) % n_feats + 1
        node = active_nodes[n_idx]

        if node == 0
            gains[f_idx, n_idx] = T(-Inf)
            bins[f_idx, n_idx] = Int32(0)
        else
            f, is_numeric, constraint, w_p, gain_p, b_max = EvoTrees._init_split_scan(
                L, h∇, nodes_sum, node, js, f_idx, feattypes, monotone_constraints,
                lambda, L2, K, ε,
            )
            temp_idx = (n_idx - 1) * n_feats + f_idx
            EvoTrees._clear_split_sums!(sums_temp, temp_idx, K)

            g_best, b_best = T(-Inf), Int32(0)
            acc1, acc2, accw = zero(T), zero(T), zero(T)
            for b in 1:b_max
                g_val, acc1, acc2, accw = _eval_split_bin(
                    L, h∇, nodes_sum, node, f, b, is_numeric, constraint,
                    acc1, acc2, accw, w_p, gain_p,
                    lambda, L2, min_weight, K, sums_temp, temp_idx, ε,
                )
                if g_val > g_best
                    g_best = g_val
                    b_best = Int32(b)
                end
            end

            gains[f_idx, n_idx] = g_best
            bins[f_idx, n_idx] = b_best
        end
    end
end

"""
    accumulate_obliv_gains_kernel!(L, gains_accum, count_accum, h∇, nodes_sum, active_nodes, js, feattypes, monotone_constraints, lambda, L2, min_weight, K, n_feats, sums_temp)

Sum eligible bin gains across active nodes into `gains_accum[bin, f]` and
increment `count_accum[bin, f]`. A split is valid only when
`count_accum[bin, f] == n_active`.
"""
@kernel function accumulate_obliv_gains_kernel!(
    ::Type{L},
    gains_accum::AbstractMatrix{T},
    count_accum::AbstractMatrix{Int32},
    @Const(h∇),
    @Const(nodes_sum),
    @Const(active_nodes),
    @Const(js),
    @Const(feattypes),
    @Const(monotone_constraints),
    lambda::T,
    L2::T,
    min_weight::T,
    K::Int,
    n_feats::Int,
    sums_temp::AbstractArray{T,2},
) where {T,L}
    gidx = @index(Global)
    n_active = length(active_nodes)
    ε = T(1e-8)

    @inbounds if gidx <= n_active * n_feats
        n_idx = (gidx - 1) ÷ n_feats + 1
        f_idx = (gidx - 1) % n_feats + 1
        node = active_nodes[n_idx]

        if node != 0
            f, is_numeric, constraint, w_p, gain_p, b_max = EvoTrees._init_split_scan(
                L, h∇, nodes_sum, node, js, f_idx, feattypes, monotone_constraints,
                lambda, L2, K, ε,
            )
            temp_idx = (n_idx - 1) * n_feats + f_idx
            EvoTrees._clear_split_sums!(sums_temp, temp_idx, K)

            acc1, acc2, accw = zero(T), zero(T), zero(T)
            for b in 1:b_max
                g_val, acc1, acc2, accw = _eval_split_bin(
                    L, h∇, nodes_sum, node, f, b, is_numeric, constraint,
                    acc1, acc2, accw, w_p, gain_p,
                    lambda, L2, min_weight, K, sums_temp, temp_idx, ε,
                )
                if g_val > zero(T)
                    Atomix.@atomic gains_accum[b, f_idx] += g_val
                    Atomix.@atomic count_accum[b, f_idx] += Int32(1)
                end
            end
        end
    end
end

"""
	broadcast_obliv_split_kernel!(best_gain, best_bin, best_feat, gain, bin, feat)

Write the shared level split into every active-node `best_*` slot.
"""
@kernel function broadcast_obliv_split_kernel!(best_gain, best_bin, best_feat, gain, bin, feat)
    i = @index(Global)
    @inbounds if i <= length(best_gain)
        best_gain[i] = gain
        best_bin[i] = bin
        best_feat[i] = feat
    end
end

"""
	reduce_best_split_kernel!(best_gain, best_bin, best_feat, gains, bins, js, n_feats)

For each node-column in `gains`, find the feature index with maximum gain and output:
- `best_gain[n_idx]`
- `best_bin[n_idx]`
- `best_feat[n_idx]` (actual feature id from `js`)
"""
@kernel function reduce_best_split_kernel!(
    best_gain,
    best_bin,
    best_feat,
    @Const(gains),
    @Const(bins),
    @Const(js),
    n_feats::Int
)
    n_idx = @index(Global)

    @inbounds if n_idx <= size(gains, 2)
        best_f_idx = 1
        best_g = gains[1, n_idx]

        for f_idx in 2:n_feats
            g = gains[f_idx, n_idx]
            if g > best_g
                best_g = g
                best_f_idx = f_idx
            end
        end

        best_gain[n_idx] = best_g
        best_bin[n_idx] = bins[best_f_idx, n_idx]
        best_feat[n_idx] = js[best_f_idx]
    end
end
