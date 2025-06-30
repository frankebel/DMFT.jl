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
