"""
    PolesSumBlock{A<:Real,B<:Number} <: AbstractPolesSum

Representation of block of poles on the real axis with locations ``a_i`` of type `A`
and weights ``B_i`` of type `Matrix{B}`.

```math
P(ω) = ∑_i \\frac{B_i}{ω-a_i}.
```

For a scalar variant see [`PolesSum`](@ref).
"""
struct PolesSumBlock{A<:Real,B<:Number} <: AbstractPolesSum
    loc::Vector{A} # locations of poles
    wgt::Vector{Matrix{B}} # weights of poles

    function PolesSumBlock{A,B}(loc, wgt) where {A,B}
        length(loc) == length(wgt) || throw(DimensionMismatch("length mismatch"))
        all(ishermitian, wgt) || throw(ArgumentError("weights are not hermitian"))
        allequal(size, wgt) || throw(DimensionMismatch("weights do not have matching size"))
        return new{A,B}(loc, wgt)
    end
end

"""
    PolesSumBlock(loc::AbstractVector{A}, wgt::Vector{<:AbstractMatrix{B}}) where {A,B}

Create a new instance of [`PolesSumBlock`](@ref) by supplying locations `loc`
and weights `wgt`.

```jldoctest
julia> loc = collect(0:2);

julia> wgt = [[1 0; 0 1], [2 1; 1 2], [2 -1; -1 2]];

julia> P = PolesSumBlock(loc, wgt)
PolesSumBlock{Int64, Int64} with 3 poles of size 2×2

julia> locations(P) === loc
true

julia> weights(P) === wgt
true
```
"""
PolesSumBlock(loc::AbstractVector{A}, wgt::Vector{<:AbstractMatrix{B}}) where {A,B} =
    PolesSumBlock{A,B}(loc, wgt)

"""
    PolesSumBlock(loc::AbstractVector, amp::AbstractMatrix{B}) where {B}

Create a new instance of [`PolesSumBlock`](@ref) by supplying locations `loc`
and amplitudes `amp`.

The ``i``-th column of `amp` is interpreted as the vector ``\\vec{b}_i`` and the weight
as ``B_i = \\vec{b}_i \\vec{b}^†_i``.

```jldoctest
julia> loc = collect(0:1);

julia> amp = [1+2im 3im; 4 5+6im];

julia> P = PolesSumBlock(loc, amp)
PolesSumBlock{Int64, Complex{Int64}} with 2 poles of size 2×2

julia> locations(P) === loc
true

julia> weights(P) == [[5 4+8im; 4-8im 16], [9 18+15im; 18-15im 61]]
true
```
"""
function PolesSumBlock(loc::AbstractVector, amp::AbstractMatrix{B}) where {B}
    wgt = Vector{Matrix{B}}(undef, size(amp, 2))
    for i in axes(amp, 2)
        foo = view(amp, :, i)
        wgt[i] = foo * foo'
    end
    return PolesSumBlock(loc, wgt)
end

function PolesSumBlock{A,B}(P::PolesSumBlock) where {A,B}
    return PolesSumBlock(Vector{A}(locations(P)), map(i -> Matrix{B}(i), weights(P)))
end

# `sqrt(matrix)` would work but is very slow on first run.
# Annotating as symmetric or hermitian makes it much faster.
# But general Julia code still does not know about semipositive eigenvalues
# and gives eltype as union of Float64 and ComplexF64.
# Therefore, decompose by hand and apply square root in-place.
function amplitude(P::PolesSumBlock{<:Real,<:Real}, i::Integer)
    m = Symmetric(Matrix{Float64}(weights(P)[i])) # symmetric and Float64
    F = eigen!(m)
    tol = maximum(F.values) * sqrt(eps())
    # NOTE: use `map!` in Julia 1.12 or higher
    for i in eachindex(F.values)
        # set small eigenvalues to zero
        F.values[i] = F.values[i] > tol ? sqrt(F.values[i]) : 0
    end
    result = F.vectors * Diagonal(F.values) * F.vectors'
    hermitianpart!(result)
    return result
end
function amplitude(P::PolesSumBlock, i::Integer)
    # use Hermitian matrix
    m = Hermitian(Matrix{ComplexF64}(weights(P)[i])) # hermitian and ComplexF64
    F = eigen!(m)
    tol = maximum(F.values) * sqrt(eps())
    # NOTE: use `map!` in Julia 1.12 or higher
    for i in eachindex(F.values)
        # set small eigenvalues to zero
        F.values[i] = F.values[i] > tol ? sqrt(F.values[i]) : 0
    end
    result = F.vectors * Diagonal(F.values) * F.vectors'
    hermitianpart!(result)
    return result
end

amplitudes(P::PolesSumBlock) = map(i -> amplitude(P, i), eachindex(P))

function evaluate_gaussian(P::PolesSumBlock, ω::Real, σ::Real)
    d = size(P, 1)
    real = zeros(Float64, d, d)
    imag = zero(real)
    for i in eachindex(P)
        w = weights(P)[i]
        real .+= w .* sqrt(2) ./ (π * σ) .* dawson((ω - locations(P)[i]) / (sqrt(2) * σ))
        imag .+= w .* pdf(Normal(locations(P)[i], σ), ω)
    end
    result = real - im * imag
    result .*= π # not spectral function
    return result
end

function evaluate_lorentzian(P::PolesSumBlock, ω::Real, δ::Real)
    d = size(P, 1)
    result = zeros(ComplexF64, d, d)
    for i in eachindex(P)
        w = weights(P)[i]
        result .+= w ./ (ω + im * δ - locations(P)[i])
    end
    return result
end

function moment(P::PolesSumBlock, n::Int=0)
    return sum(i -> i[1]^n * i[2], zip(locations(P), weights(P)))
end

weights(P::PolesSumBlock) = P.wgt

function Base.copy(P::PolesSumBlock)
    return PolesSumBlock(copy(locations(P)), map(i -> copy(i), weights(P)))
end

Base.eltype(::Type{<:PolesSumBlock{A,B}}) where {A,B} = promote_type(A, B)

function Base.show(io::IO, P::PolesSumBlock)
    return print(
        io, summary(P), " with ", length(P), " poles of size ", size(P, 1), "×", size(P, 2)
    )
end

Base.size(P::PolesSumBlock) = size(first(weights(P)))
Base.size(P::PolesSumBlock, i) = size(first(weights(P)), i)

function Base.transpose(P::PolesSumBlock)
    return PolesSumBlock(locations(P), map(transpose, weights(P)))
end
