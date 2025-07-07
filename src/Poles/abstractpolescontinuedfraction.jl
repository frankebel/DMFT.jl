"""
    AbstractPolesContinuedFraction

Supertype which represents a (block) function on the real axis as a continued fraction.
"""
abstract type AbstractPolesContinuedFraction <: AbstractPoles end

amplitudes(P::AbstractPolesContinuedFraction) = P.amp

"""
    scale(P::AbstractPolesContinuedFraction)

Return the scale of `P`.
"""
scale(P::AbstractPolesContinuedFraction) = P.scl
