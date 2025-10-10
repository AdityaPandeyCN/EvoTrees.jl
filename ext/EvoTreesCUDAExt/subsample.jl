using CUDA

function EvoTrees.subsample(
	left::CuVector{T},
	is::CuVector{T},
	mask_cond::CuVector{UInt8},
	rowsample::AbstractFloat,
	rng
) where {T}
	n = length(left)
	threshold = UInt8(round(255 * rowsample))

	rand!(mask_cond)

	mask = CuArray{Int32}(undef, n)

	function mark_kernel!(mask, mask_cond, threshold)
		i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
		if i <= length(mask)
			@inbounds mask[i] = mask_cond[i] <= threshold ? Int32(1) : Int32(0)
		end
		return nothing
	end

	threads = 256
	blocks = cld(n, threads)
	@cuda threads = threads blocks = blocks mark_kernel!(mask, mask_cond, threshold)

	positions = cumsum(mask)
	count = Array(positions[end:end])[1]

	if count == 0
		@error "no subsample observation - choose larger rowsample"
		return view(is, 1:1)
	end

	function compact_kernel!(is, left, mask, positions)
		i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
		if i <= length(left)
			@inbounds if mask[i] == 1
				is[positions[i]] = left[i]
			end
		end
		return nothing
	end

	@cuda threads = threads blocks = blocks compact_kernel!(is, left, mask, positions)

	return view(is, 1:count)
end