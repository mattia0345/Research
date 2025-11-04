export ons, offs, flip, reaction_network

ons(x::BitStr) = baddrs(x)

offs(x::BitStr) = baddrs(neg(x))

# the "flip" method in BitBasis only works on masks, not indices
function flip(x::BitStr{N, T}, idx::Integer) where {N, T}
    1 <= idx <= N ||
        error("Index $idx out of bounds for bit strings of length $N.")
    mask = bmask(BitStr{N, T}, idx)
    return BitBasis.flip(x, mask)
end

# function bitarray_to_uint64(arr::BitArray)::UInt64
#     # Ensure the BitArray is not too large for UInt64
#     if length(arr) > 64
#         error("BitArray too large for UInt64 conversion.")
#     end

#     val = UInt64(0)
#     # Iterate from the least significant bit (rightmost in a typical representation)
#     # to the most significant bit (leftmost).
#     for i in 1:length(arr)
#         if arr[i]
#             val |= (UInt64(1) << (i - 1))
#         end
#     end
#     return val
# end


function reaction_network(rate_constants, schemes; species=String[], parameters=String[])
    reactions = String[]

    for (constant, scheme) in zip(rate_constants, schemes)
        push!(reactions, constant*", "*scheme*" \n")
    end 

    network = "@reaction_network begin \n"

    if !isempty(species)
        species_block = "@species begin \n"
        for s in species
            species_block *= "$s(t)\n"
        end
        species_block *= "end \n"
        network *= species_block
    end

    if !isempty(parameters)
        network *= "@parameters begin \n"
        network *= join(parameters, "\n")
        network *= "\nend \n"
    end

    for reaction in reactions
        network *= reaction
    end

    complete_network = network*"end "

    # for constant in rate_constants
    #     complete_network *= constant*" "
    # end
    
    parsed_network = Meta.parse(complete_network)

    rn = eval(parsed_network)    

    return rn
end