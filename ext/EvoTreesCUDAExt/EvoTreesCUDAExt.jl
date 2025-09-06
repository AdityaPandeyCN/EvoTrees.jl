module EvoTreesGPU

using EvoTrees
using KernelAbstractions
using Adapt
using Tables
using Random
using StatsBase 

# Use KernelAbstractions to create arrays on the default GPU backend
function EvoTrees.device_ones(::Type{<:EvoTrees.GPU}, ::Type{T}, n::Int) where {T}
    backend = KernelAbstractions.GPU()
    return KernelAbstractions.ones(backend, T, n)
end

# Get the array type (e.g., CuArray, ROCArray) from the default GPU backend
function EvoTrees.device_array_type(::Type{<:EvoTrees.GPU})
    backend = KernelAbstractions.GPU()
    return KernelAbstractions.get_array_type(backend)
end

# Perform a generic post-fit cleanup
function EvoTrees.post_fit_gc(::Type{<:EvoTrees.GPU})
    GC.gc(true)
    # Synchronize the default GPU to ensure all work is complete.
    # Backend-specific reclaim (like CUDA.reclaim()) isn't portable.
    backend = KernelAbstractions.GPU()
    KernelAbstractions.synchronize(backend)
end

include("structs.jl")
include("loss.jl")
include("eval.jl")
include("predict.jl")
include("init.jl")
include("subsample.jl")
include("fit-utils.jl")
include("fit.jl")

end

