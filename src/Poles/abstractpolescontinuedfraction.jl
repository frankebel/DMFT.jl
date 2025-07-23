"""
    AbstractPolesContinuedFraction

Supertype which represents a (block) function on the real axis as a continued fraction.
"""
abstract type AbstractPolesContinuedFraction <: AbstractPoles end

amplitude(P::AbstractPolesContinuedFraction, i::Integer) = amplitudes(P)[i]

amplitudes(P::AbstractPolesContinuedFraction) = P.amplitudes

"""
    scale(P::AbstractPolesContinuedFraction)

Return the scale of `P`.
"""
scale(P::AbstractPolesContinuedFraction) = P.scale
