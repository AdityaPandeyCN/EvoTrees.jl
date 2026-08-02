# bench_subtract.jl
#
# Legitimizes step 2: replacing the per-node nested-loop sibling subtraction with
# a single reshaped, threaded pass sharing one signature with the GPU method.
#
# Run once, on a quiet machine:
#   julia -t auto benchmarks/bench_subtract.jl
#
# Part A is self-contained (no EvoTrees import): it times the old and new CPU
# bodies against each other on the exact 4D layout and asserts bitwise equality.
# Part B is the end-to-end gate: it needs EvoTrees dev'd to the branch under
# test, and is run twice (baseline commit, then this commit) and diffed.

using Base.Threads: @threads
using Random
using Printf
using Statistics

# --- implementations under test -------------------------------------------

# current (post step-1): one call per node, @threads at the call site,
# @simd over axis 1 which is length 2K+1 (3 for K=1)
function subtract_old!(h∇::Array{Float64,4}, n::Integer, np::Integer, ns::Integer, js)
    @inbounds for j in js
        for b in axes(h∇, 2)
            @simd for k in axes(h∇, 1)
                h∇[k, b, j, n] = h∇[k, b, j, np] - h∇[k, b, j, ns]
            end
        end
    end
    return nothing
end

function subtract_old_level!(h∇, nodes, js)
    @threads for n in nodes
        sib = n % 2 == 0 ? n + 1 : n - 1
        subtract_old!(h∇, n, n >> 1, sib, js)
    end
    return nothing
end

# proposed: (2K+1, nbins) collapsed into one contiguous run, @threads inside
function subtract_new!(h∇::Array{Float64,4}, nodes, js)
    isempty(nodes) && return nothing
    h = reshape(h∇, :, size(h∇, 3), size(h∇, 4))
    @threads for n in nodes
        n > 1 || continue
        np, ns = n >> 1, n ⊻ 1
        @inbounds for j in js
            @simd for i in axes(h, 1)
                h[i, j, n] = h[i, j, np] - h[i, j, ns]
            end
        end
    end
    return nothing
end

# --- harness ---------------------------------------------------------------

function make_hist(K, nbins, nfeats, nnodes; seed=1)
    rng = Xoshiro(seed)
    return rand(rng, Float64, 2K + 1, nbins, nfeats, nnodes)
end

function level_nodes(depth)
    first_node = 1 << (depth - 1)
    n_current = collect(first_node:(1<<depth)-1)
    return n_current[2:2:end]
end

function timeit(f, args...; reps=20)
    f(args...)                      # warm up / compile
    ts = Float64[]
    for _ in 1:reps
        t = time_ns()
        f(args...)
        push!(ts, (time_ns() - t) / 1e6)
    end
    return median(ts)
end

println("threads = ", Threads.nthreads())

# --- A1. bitwise equality --------------------------------------------------

println("\n=== A1. bitwise equality (old vs new) ===")
let allok = true
    for K in (1, 2), nbins in (32, 64), nfeats in (10, 100), depth in (2, 6, 11)
        h1 = make_hist(K, nbins, nfeats, 1 << (depth + 1))
        h2 = copy(h1)
        js = collect(UInt32.(1:max(1, round(Int, 0.9nfeats))))
        nodes = level_nodes(depth)
        subtract_old_level!(h1, nodes, js)
        subtract_new!(h2, nodes, js)
        ok = h1 == h2
        allok &= ok
        ok || @printf("  MISMATCH K=%d nbins=%d nfeats=%d depth=%d\n", K, nbins, nfeats, depth)
    end
    println(allok ? "  all configurations bitwise identical" : "  FAILED")
end

# --- A2. isolated subtraction timing ---------------------------------------

println("\n=== A2. subtraction cost per level (median ms of 20) ===")
@printf("%3s %6s %7s %6s %8s | %9s %9s %7s\n",
    "K", "nbins", "nfeats", "depth", "nodes", "old_ms", "new_ms", "speedup")

for K in (1, 2), nbins in (32, 64), nfeats in (10, 100), depth in (2, 6, 9, 11)
    nnodes = 1 << (depth + 1)
    nodes = level_nodes(depth)
    js = collect(UInt32.(1:max(1, round(Int, 0.9nfeats))))
    h = make_hist(K, nbins, nfeats, nnodes)
    t_old = timeit(subtract_old_level!, h, nodes, js)
    t_new = timeit(subtract_new!, h, nodes, js)
    @printf("%3d %6d %7d %6d %8d | %9.3f %9.3f %7.2fx\n",
        K, nbins, nfeats, depth, length(nodes), t_old, t_new, t_old / t_new)
end

# --- B. end-to-end gate ----------------------------------------------------
# Only meaningful when run twice and diffed. Guarded so Part A stays usable
# without a dev'd EvoTrees.

if get(ENV, "BENCH_E2E", "1") == "1"
    using EvoTrees

    println("\n=== B1. prediction fingerprint (must match baseline exactly) ===")
    let
        rng = Xoshiro(123)
        x = randn(rng, Float32, 50_000, 20)
        y = Float32.(sin.(x[:, 1]) .+ 0.5f0 .* x[:, 2] .+ 0.1f0 .* randn(rng, Float32, 50_000))
        for depth in (6, 11)
            config = EvoTreeRegressor(; loss=:mse, nrounds=50, max_depth=depth,
                nbins=64, eta=0.05, rowsample=0.5, colsample=0.9, seed=42)
            m = EvoTrees.fit(config; x_train=x, y_train=y)
            p = m(x)
            @printf("depth=%2d  sum=%.13e  norm=%.13e\n",
                depth, sum(Float64, p), sqrt(sum(abs2, Float64.(p))))
        end
    end

    println("\n=== B2. fit wall time (median of 3) ===")
    @printf("%8s %7s %6s | %10s %11s\n", "nobs", "nfeats", "depth", "median_s", "alloc_MiB")
    for (nobs, nfeats, depth) in [
        (100_000, 100, 6),
        (100_000, 100, 11),
        (1_000_000, 100, 6),
        (1_000_000, 100, 11),
    ]
        rng = Xoshiro(123)
        x = randn(rng, Float32, nobs, nfeats)
        y = Float32.(sin.(x[:, 1]) .+ 0.5f0 .* x[:, 2] .+ 0.1f0 .* randn(rng, Float32, nobs))
        config = EvoTreeRegressor(; loss=:mse, nrounds=100, max_depth=depth,
            nbins=64, eta=0.05, rowsample=0.5, colsample=0.9, seed=42)
        EvoTrees.fit(EvoTreeRegressor(; loss=:mse, nrounds=5, max_depth=depth, nbins=64,
                eta=0.05, rowsample=0.5, colsample=0.9, seed=42); x_train=x, y_train=y)
        ts, as = Float64[], Float64[]
        for _ in 1:3
            GC.gc()
            st = @timed EvoTrees.fit(config; x_train=x, y_train=y)
            push!(ts, st.time)
            push!(as, st.bytes / 2^20)
        end
        @printf("%8d %7d %6d | %10.3f %11.1f\n", nobs, nfeats, depth, median(ts), median(as))
    end
end
