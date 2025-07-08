"""
    PolesContinuedFractionBlock{A<:Number,B<:Number} <: AbstractPolesContinuedFraction

Representation of poles on the real axis with as a continued fraction with
locations ``A_i`` of type `A` and amplitudes ``B_i`` of type `B`.

All matrices must be hermitian.
The scale factor ``S`` rescales the whole object.

```math
P(ω) = S \\frac{1}{ω-A_1-B_1\\frac{1}{ω-A_2-…}B_1} S
```
"""
struct PolesContinuedFractionBlock{A<:Number,B<:Number} <: AbstractPolesContinuedFraction
    loc::Vector{Matrix{A}} # locations of poles
    amp::Vector{Matrix{B}} # amplitudes of poles
    scl::Matrix{B} # scale

    function PolesContinuedFractionBlock{A,B}(loc, amp, scl) where {A,B}
        length(loc) == length(amp) + 1 || throw(ArgumentError("length mismatch"))
        # hermitian
        all(ishermitian, loc) || throw(ArgumentError("locations are not hermitian"))
        all(ishermitian, amp) || throw(ArgumentError("amplitudes are not hermitian"))
        ishermitian(scl) || throw(ArgumentError("scale is not hermitian"))
        # size
        s = size(first(loc))
        all(i -> size(i) == s, loc) ||
            throw(DimensionMismatch("locations do not have matching size"))
        all(i -> size(i) == s, amp) ||
            throw(DimensionMismatch("amplitudes do not have matching size"))
        size(scl) == s || throw(DimensionMismatch("scale does not have matching size"))
        return new{A,B}(loc, amp, scl)
    end
end

"""
    PolesContinuedFractionBlock(
        loc::AbstractVector{<:AbstractMatrix{<:A}},
        amp::AbstractVector{<:AbstractMatrix{<:B}},
        [scl::AbstractMatrix{<:B},]
    ) where {A,B}

Create a new instance of [`PolesContinuedFraction`](@ref) by supplying locations `loc`,
amplitudes `amp`, and scale `scl`.

By default the scale is set to the identiy matrix ``1``.
"""
function PolesContinuedFractionBlock(
    loc::AbstractVector{<:AbstractMatrix{<:A}},
    amp::AbstractVector{<:AbstractMatrix{<:B}},
    scl::AbstractMatrix{<:B},
) where {A,B}
    return PolesContinuedFractionBlock{A,B}(loc, amp, scl)
end

# convert type
function PolesContinuedFractionBlock{A,B}(P::PolesContinuedFractionBlock) where {A,B}
    return PolesContinuedFractionBlock{A,B}(
        map(i -> Matrix{A}(i), locations(P)),
        map(i -> Matrix{B}(i), amplitudes(P)),
        Matrix{B}(scale(P)),
    )
end

# scale is identity matrix
function PolesContinuedFractionBlock(
    loc::AbstractVector{<:AbstractMatrix{<:A}}, amp::AbstractVector{<:AbstractMatrix{<:B}}
) where {A,B}
    scl = LinearAlgebra.I(size(first(loc), 1))
    return PolesContinuedFractionBlock{A,B}(loc, amp, scl)
end

function evaluate_lorentzian(P::PolesContinuedFractionBlock, ω::Real, δ::Real)
    result = zeros(ComplexF64, size(P))
    loc = Iterators.reverse(locations(P))
    amp = Iterators.reverse(amplitudes(P))
    for (A, B) in zip(loc, amp)
        result = B * inv((ω + im * δ) * I - A - result) * B
    end
    result = scale(P) * inv((ω + im * δ) * I - locations(P)[1] - result) * scale(P)
    return result
end

function Core.Array(P::PolesContinuedFractionBlock)
    n1 = length(P)
    n2 = size(P, 1)
    n = n1 * n2
    result = zeros(eltype(P), n, n)
    for i in 1:(n1 - 1)
        i1 = 1 + (i - 1) * n2
        i2 = i * n2
        result[i1:i2, (i1 + n2):(i2 + n2)] = amplitudes(P)[i] # upper diagonal
        result[i1:i2, i1:i2] = locations(P)[i] # main diagonal
        result[(i1 + n2):(i2 + n2), i1:i2] = amplitudes(P)[i] # lower diagonal
    end
    result[(end - n2 + 1):end, (end - n2 + 1):end] = locations(P)[end] # last element
    return result
end

Base.eltype(::Type{<:PolesContinuedFractionBlock{A,B}}) where {A,B} = promote_type(A, B)

Base.size(P::PolesContinuedFractionBlock) = size(scale(P))
Base.size(P::PolesContinuedFractionBlock, i) = size(scale(P), i)
