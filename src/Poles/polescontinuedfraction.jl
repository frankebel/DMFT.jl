"""
    PolesContinuedFraction{A<:Real,B<:Real} <: AbstractPolesContinuedFraction

Representation of poles on the real axis as a continued fraction with
locations ``a_i`` of type `A` and amplitudes ``b_i`` of type `B`.

The scale factor ``s`` of type `B` rescales the whole object.

```math
P(ω) = \\frac{s^2}{ω - a_1 - \\frac{b_1^2}{ω - a_2 - …}}
```
"""
struct PolesContinuedFraction{A<:Real,B<:Real} <: AbstractPolesContinuedFraction
    loc::Vector{A} # locations of poles
    amp::Vector{B} # amplitudes of poles
    scl::B # scale

    function PolesContinuedFraction{A,B}(loc, amp, scl) where {A,B}
        length(loc) == length(amp) + 1 || throw(ArgumentError("length mismatch"))
        return new{A,B}(loc, amp, scl)
    end
end

"""
    PolesContinuedFraction(
        loc::AbstractVector{A}, amp::AbstractVector{B}, scl=one(B)
    ) where {A,B}

Create a new instance of [`PolesContinuedFraction`](@ref) by supplying locations `loc`,
amplitudes `amp`, and scale `scl`.

By default the scale is set to ``1``.
"""
function PolesContinuedFraction(
    loc::AbstractVector{A}, amp::AbstractVector{B}, scl=one(B)
) where {A,B}
    return PolesContinuedFraction{A,B}(loc, amp, scl)
end

# convert type
function PolesContinuedFraction{A,B}(P::PolesContinuedFraction) where {A,B}
    return PolesContinuedFraction{A,B}(
        Vector{A}(locations(P)), Vector{B}(amplitudes(P)), B(scale(P))
    )
end

function evaluate_lorentzian(P::PolesContinuedFraction, ω::Real, δ::Real)
    result = zero(ComplexF64)
    loc = Iterators.reverse(locations(P))
    amp = Iterators.reverse(amplitudes(P))
    for (a, b) in zip(loc, amp)
        result = abs2(b) / (ω + im * δ - a - result)
    end
    result = abs2(scale(P)) / (ω + im * δ - locations(P)[1] - result)
    return result
end

weight(P::PolesContinuedFraction, i::Integer) = abs2(amplitudes(P)[i])

weights(P::PolesContinuedFraction) = abs2.(amplitudes(P))

function Core.Array(P::PolesContinuedFraction)
    # lose information about scale
    return Array(SymTridiagonal(P))
end

Base.eltype(::Type{<:PolesContinuedFraction{A,B}}) where {A,B} = promote_type(A, B)

function Base.show(io::IO, P::PolesContinuedFraction)
    return print(io, summary(P), " with ", length(P), " poles")
end

function LinearAlgebra.SymTridiagonal(P::PolesContinuedFraction)
    # lose information about scale
    return SymTridiagonal(locations(P), amplitudes(P))
end
