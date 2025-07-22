"""
    AbstractPolesSum

Supertype which represents a (block) function on the real axis as a sum of poles.
"""
abstract type AbstractPolesSum <: AbstractPoles end

Base.eachindex(P::AbstractPolesSum) = eachindex(locations(P))

"""
    moment(P::AbstractPolesSum, n::Int=0)

Return the `n`-th moment.
"""
function moment end

"""
    moments(P::AbstractPolesSum, ns)

Return the `n`-th moment for each `n` in `ns`.
"""
function moments(P::AbstractPolesSum, ns)
    return map(i -> moment(P, i), ns)
end

"""
    remove_zero_weight!(P::AbstractPolesSum, remove_zero::Bool=true)

Remove all poles which have zero weight.

If `remove_zero`, the pole at ``a_i = 0`` with zero weight is also removed.

See also [`remove_zero_weight`](@ref).
"""
function remove_zero_weight!(P::AbstractPolesSum, remove_zero::Bool=true)
    i = 1
    while i <= length(P)
        if iszero(locations(P)[i]) && !remove_zero
            # keep pole at origin
            i += 1
            continue
        end

        if iszero(weights(P)[i])
            deleteat!(locations(P), i)
            deleteat!(weights(P), i)
        else
            i += 1
        end
    end
    return P
end

"""
    remove_zero_weight(P::AbstractPolesSum, remove_zero::Bool=true)

Remove all poles which have zero weight.

If `remove_zero`, the pole at ``a_i = 0`` with zero weight is also removed.

See also [`remove_zero_weight!`](@ref).
"""
function remove_zero_weight(P::AbstractPolesSum, remove_zero::Bool=true)
    return remove_zero_weight!(copy(P), remove_zero)
end

function Base.allunique(P::AbstractPolesSum)
    loc = locations(P)
    # allunique discrimates between ±zero(Float64)
    return allunique(loc) && length(findall(iszero, loc)) <= 1
end

function Base.issorted(P::AbstractPolesSum, args...; kwargs...)
    return issorted(locations(P), args...; kwargs...)
end

Base.reverse(P::AbstractPolesSum) = reverse!(copy(P))

Base.sort(P::AbstractPolesSum) = sort!(copy(P))
