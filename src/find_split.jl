#############################################
# Get the braking points
#############################################
function get_edges(X, nbins=250)
    edges = Vector{Vector}(undef, size(X,2))
    @threads for i in 1:size(X, 2)
        edges[i] = unique(quantile(view(X, :,i), (0:nbins)/nbins))[2:end]
        if length(edges[i]) == 0
            edges[i] = [minimum(view(X, :,i))]
        end
    end
    return edges
end

####################################################
# Transform X matrix into a UInt8 binarized matrix
####################################################
function binarize(X, edges)
    X_bin = zeros(UInt8, size(X))
    @threads for i in 1:size(X, 2)
        X_bin[:,i] = searchsortedlast.(Ref(edges[i][1:end-1]), view(X,:,i)) .+ 1
    end
    X_bin
end

function find_bags(x_bin::Vector{T}) where T <: Real
    𝑖 = 1:length(x_bin) |> collect
    bags = [BitSet() for _ in 1:maximum(x_bin)]
    for bag in 1:length(bags)
        bags[bag] = BitSet(𝑖[x_bin .== bag])
    end
    return bags
end

function update_bags!(bins, set)
    for bin in bins
        intersect!(bin, set)
    end
end


function find_split_static!(hist_δ, hist_δ², hist_𝑤, bins::Vector{BitSet}, X_bin, δ, δ², 𝑤, ∑δ, ∑δ², ∑𝑤, params::EvoTreeRegressor, info::SplitInfo{S, Int}, edges, set::BitSet) where {S<:AbstractFloat}

    # initialize histogram
    hist_δ .*= 0.0
    hist_δ² .*= 0.0
    hist_𝑤 .*= 0.0

    # initialize tracking
    ∑δL = SVector{params.K,Float64}(0.0)
    ∑δ²L = SVector{params.K,Float64}(0.0)
    ∑𝑤L = SVector{1,Float64}(0.0)
    ∑δR = ∑δ
    ∑δ²R = ∑δ²
    ∑𝑤R = ∑𝑤

    # build histogram
    @inbounds for i in set
        hist_δ[X_bin[i]] += δ[i]
        hist_δ²[X_bin[i]] += δ²[i]
        hist_𝑤[X_bin[i]] += 𝑤[i]
    end

    @inbounds for bin in 1:(length(bins)-1)
        ∑δL += hist_δ[bin]
        ∑δ²L += hist_δ²[bin]
        ∑𝑤L += hist_𝑤[bin]
        ∑δR -= hist_δ[bin]
        ∑δ²R -= hist_δ²[bin]
        ∑𝑤R -= hist_𝑤[bin]

        gainL, gainR = get_gain(params.loss, ∑δL, ∑δ²L, ∑𝑤L, params.λ), get_gain(params.loss, ∑δR, ∑δ²R, ∑𝑤R, params.λ)
        gain = gainL + gainR

        if gain > info.gain && ∑𝑤L[1] >= params.min_weight && ∑𝑤R[1] >= params.min_weight
            info.gain = gain
            info.gainL = gainL
            info.gainR = gainR
            info.∑δL = ∑δL
            info.∑δ²L = ∑δ²L
            info.∑𝑤L = ∑𝑤L
            info.∑δR = ∑δR
            info.∑δ²R = ∑δ²R
            info.∑𝑤R = ∑𝑤R
            info.cond = edges[bin]
            info.𝑖 = bin
        end
    end
    return
end
