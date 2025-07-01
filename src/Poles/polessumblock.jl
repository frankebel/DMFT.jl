"""
    PolesSumBlock{A<:Real,B<:Number} <: AbstractPolesSum

Representation of block of poles on the real axis with locations ``a_i`` of type `A`
and amplitudes ``b_i`` of type `B`

```math
P(ω) = ∑_i \\frac{\\vec{b}_i \\vec{b}^†_i}{ω-a_i}.
```

For a scalar variant see [`PolesSum`](@ref).
"""
struct PolesSumBlock{A<:Real,B<:Number} <: AbstractPolesSum
    loc::Vector{A} # locations of poles
    amp::Matrix{B} # amplitudes of poles

    function PolesSumBlock{A,B}(loc, amp) where {A,B}
        length(loc) == size(amp, 2) || throw(DimensionMismatch("length mismatch"))
        return new{A,B}(loc, amp)
    end
end

"""
    PolesSumBlock(loc::Vector{A}, amp::Matrix{B}) where {A,B}

Create a new instance of [`PolesSumBlock`](@ref) by supplying locations `loc`
and amplitudes `amp`.

```jldoctest
julia> loc = collect(0:5);

julia> amp = reshape(collect(-5:6), (2, 6));

julia> P = PolesSumBlock(loc, amp)
2×5 PolesSum{Int64, Int64}

julia> locations(P) === loc
true

julia> amplitudes(P) === amp
true
```
"""
PolesSumBlock(loc::Vector{A}, amp::Matrix{B}) where {A,B} = PolesSumBlock{A,B}(loc, amp)

function PolesSumBlock{A,B}(P::PolesSumBlock) where {A,B}
    return PolesSumBlock(Vector{A}(locations(P)), Matrix{B}(amplitudes(P)))
end

amplitudes(P::PolesSumBlock) = P.amp

function evaluate_gaussian(P::PolesSumBlock, ω::Real, σ::Real)
    d = size(P, 1)
    real = zeros(Float64, d, d)
    imag = zero(real)
    for i in eachindex(P)
        w = weight(P, i)
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
        w = weight(P, i)
        result .+= w ./ (ω + im * δ - locations(P)[i])
    end
    return result
end

function moment(P::PolesSumBlock, n::Int=0)
    return sum(i -> i[1]^n * i[2], zip(locations(P), weights(P)))
end

function weight(P::PolesSumBlock, i::Integer)
    foo = view(amplitudes(P), :, i)
    return foo * foo'
end

weights(P::PolesSumBlock) = map(i -> weight(P, i), eachindex(P))

function Base.show(io::IO, P::PolesSumBlock)
    return print(io, size(P, 1), "×", size(P, 2), " ", summary(P))
end

Base.size(P::PolesSumBlock) = size(amplitudes(P))
Base.size(P::PolesSumBlock, i) = size(amplitudes(P), i)
