module EvoTreesCUDAExt

using EvoTrees
using CUDA
using KernelAbstractions

EvoTrees.gpu_backend(::Type{<:EvoTrees.CUDADevice}) = CUDA.CUDABackend()
EvoTrees.device_array_type(::Type{<:EvoTrees.CUDADevice}) = CuArray
function EvoTrees.post_fit_gc(::Type{<:EvoTrees.CUDADevice}, cache)
    for f in fieldnames(typeof(cache))
        x = getfield(cache, f)
        x isa CuArray && CUDA.unsafe_free!(x)
    end
    g = cache.group
    if !isnothing(g)
        CUDA.unsafe_free!(g.group_gpu)
        CUDA.unsafe_free!(g.mask_gpu)
    end
    return nothing
end

end
