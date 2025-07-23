"""
    AbstractPoles

Supertype which represents a function on the real axis as a collection of poles.
"""
abstract type AbstractPoles end

"""
    locations(P::AbstractPoles)

Return the locations of `P`.
"""
locations(P::AbstractPoles) = P.locations

"""
    amplitudes(P::AbstractPoles)

Return the amplitudes of `P`.
"""
function amplitudes end

"""
    weights(P::AbstractPoles)

Return the weights of `P`.
"""
function weights end

"""
    weight(P::AbstractPoles, i::Integer)

Return the weight of `P` at index `i`.

See also [`weights`](@ref).
"""
function weight end

"""
    evaluate_lorentzian(P::AbstractPoles, ω, δ)

Evaluate `P` with Lorentzian broadening ``P(ω + \\mathrm{i}δ)``.
"""
function evaluate_lorentzian end

function evaluate_lorentzian(P::AbstractPoles, ω::Vector{<:Real}, δ)
    # map for each point in given grid
    return map(i -> evaluate_lorentzian(P, i, δ), ω)
end

"""
    evaluate_gaussian(P::AbstractPoles, ω, σ)

Evaluate `P` with Gaussian broadening ``σ``.
"""
function evaluate_gaussian end

function evaluate_gaussian(P::AbstractPoles, ω::Vector{<:Real}, σ)
    # map for each point in given grid
    return map(i -> evaluate_gaussian(P, i, σ), ω)
end

Base.length(P::AbstractPoles) = length(locations(P))
