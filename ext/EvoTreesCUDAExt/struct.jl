struct CacheGPU
    info::Dict
    x_bin::CuMatrix
    y::CuArray
    w::Union{Nothing, CuVector}
    K::Int
    nodes::Vector  # Keep CPU nodes for compatibility
    pred::CuMatrix
    nidx::CuVector{UInt32}
    is_in::CuVector{UInt32}
    is_out::CuVector{UInt32}
    mask::CuVector{UInt8}
    js_::Vector{UInt32}
    js::CuVector{UInt32}
    ∇::CuMatrix
    h∇::CuArray
    h∇L::CuArray
    h∇R::CuArray
    fnames::Vector{String}
    edges::Vector
    featbins::Vector
    feattypes_gpu::CuVector{Bool}
    cond_feats::Vector{Int}
    cond_feats_gpu::CuVector
    cond_bins::Vector{UInt8}
    cond_bins_gpu::CuVector
    monotone_constraints_gpu::CuVector{Int32}
    left_nodes_buf::CuVector{Int32}
    right_nodes_buf::CuVector{Int32}
    target_mask_buf::CuVector{UInt8}
end

