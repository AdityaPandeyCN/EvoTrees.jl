using KernelAbstractions
using Random

@kernel function subsample_step_1_kernel!(is_in, mask, cond::Float32, counts, chunk_size::Int)
    bid = @index(Global)
    gdim = length(counts)

    i_start = chunk_size * (bid - 1) + 1
    i_stop = bid == gdim ? length(is_in) : i_start + chunk_size - 1
    count = 0

    @inbounds for i = i_start:i_stop
        if mask[i] <= cond
            is_in[i_start+count] = i
            count += 1
        end
    end
    counts[bid] = count
end

@kernel function subsample_step_2_kernel!(is_in, is_out, counts, counts_cum, chunk_size::Int)
    bid = @index(Global)
    count_cum = counts_cum[bid]
    i_start = chunk_size * (bid - 1)
    @inbounds for i = 1:counts[bid]
        is_out[count_cum+i] = is_in[i_start+i]
    end
end

# Override the main subsample function for GPU arrays - THIS IS THE KEY FIX
function EvoTrees.subsample(left::CuVector{UInt32}, is::CuVector{UInt32}, 
                           mask_cond::CuVector{UInt8}, rowsample::AbstractFloat, rng)
    backend = KernelAbstractions.get_backend(is)

    # Generate random mask directly on GPU
    Random.rand!(rng, mask_cond)
    cond = round(UInt8, 255 * rowsample)

    chunk_size = cld(length(is), min(cld(length(is), 128), 2048))
    nblocks = cld(length(is), chunk_size)
    counts = KernelAbstractions.zeros(backend, Int, nblocks)

    step1! = subsample_step_1_kernel!(backend)
    step1!(is, mask_cond, Float32(cond), counts, chunk_size; ndrange=nblocks, workgroupsize=min(256, nblocks))
    KernelAbstractions.synchronize(backend)

    # Compute cumulative counts on host for compatibility
    counts_host = Array(counts)
    counts_cum_host = cumsum(counts_host) .- counts_host
    counts_cum = similar(counts)
    copyto!(counts_cum, counts_cum_host)

    step2! = subsample_step_2_kernel!(backend)
    step2!(is, left, counts, counts_cum, chunk_size; ndrange=nblocks, workgroupsize=min(256, nblocks))
    KernelAbstractions.synchronize(backend)

    # Get total count
    counts_sum = sum(counts_host)

    if counts_sum == 0
        @error "no subsample observation - choose larger rowsample"
        return view(left, 1:1)  # Return at least one element to avoid errors
    else
        return view(left, 1:counts_sum)
    end
end

# Also handle the case with Float32 mask if needed
function EvoTrees.subsample(left::CuVector{UInt32}, is::CuVector{UInt32}, 
                           mask::CuVector{Float32}, rowsample::AbstractFloat, rng)
    backend = KernelAbstractions.get_backend(is)

    # Generate random mask directly on GPU
    Random.rand!(rng, mask)
    cond = Float32(rowsample)

    chunk_size = cld(length(is), min(cld(length(is), 128), 2048))
    nblocks = cld(length(is), chunk_size)
    counts = KernelAbstractions.zeros(backend, Int, nblocks)

    step1! = subsample_step_1_kernel!(backend)
    step1!(is, mask, cond, counts, chunk_size; ndrange=nblocks, workgroupsize=min(256, nblocks))
    KernelAbstractions.synchronize(backend)

    # Compute cumulative counts on host for compatibility
    counts_host = Array(counts)
    counts_cum_host = cumsum(counts_host) .- counts_host
    counts_cum = similar(counts)
    copyto!(counts_cum, counts_cum_host)

    step2! = subsample_step_2_kernel!(backend)
    step2!(is, left, counts, counts_cum, chunk_size; ndrange=nblocks, workgroupsize=min(256, nblocks))
    KernelAbstractions.synchronize(backend)

    # Get total count
    counts_sum = sum(counts_host)

    if counts_sum == 0
        @error "no subsample observation - choose larger rowsample"
        return view(left, 1:1)  # Return at least one element to avoid errors
    else
        return view(left, 1:counts_sum)
    end
end

