"""
    Pole{A<:AbstractVector{<:Real},B<:AbstractVecOrMat}

Representation of poles on the real axis with locations `a::A` and weights `b::B`.

If both are `A` and `B` are vectors, it is just a sum:

```math
P(z) = ∑_i \\frac{|b_i|^2}{z-a_i}
```

If `B` is a matrix, its ``i``-th column is interpreted as a vector ``\\vec{b_i}`` with

```math
P(z) = \\sum_i \\frac{\\vec{b}_i\\vec{b}_i^\\dagger}{z-a_i}
```

Can be evaluated at
- point `z` in the upper complex plane: `P(z)`
- vector of points `Z` in the upper complex plane: `P(Z)`
- real point `ω` with Gaussian broadening `σ`: `P(ω, σ)`
- vector of real points `W` with Gaussian broadening `σ`: `P(W, σ)`
"""
struct Pole{A<:AbstractVector{<:Real},B<:AbstractVecOrMat}
    a::A
    b::B

    # both are vectors
    function Pole{A,B}(a, b) where {A<:AbstractVector{<:Number},B<:AbstractVector{<:Number}}
        length(a) == length(b) || throw(DimensionMismatch("length mismatch"))
        return new{A,B}(a, b)
    end

    # `b` is a matrix
    function Pole{A,B}(a, b) where {A<:AbstractVector{<:Number},B<:AbstractMatrix{<:Number}}
        length(a) == size(b, 2) || throw(DimensionMismatch("length mismatch"))
        return new{A,B}(a, b)
    end
end

Pole(a::A, b::B) where {A,B} = Pole{A,B}(a, b)

Pole{A,B}(P::Pole) where {A,B} = Pole(A(P.a), B(P.b))

# evaluate with Lorentzian broadening at complex value `z`

function (P::Pole{<:Any,<:AbstractVector})(z::Complex)
    result = zero(z)
    for i in eachindex(P.a)
        result += abs2(P.b[i]) / (z - P.a[i])
    end
    return result
end

function (P::Pole{<:Any,<:AbstractMatrix})(z::Complex)
    d = size(P.b, 1)
    result = zeros(ComplexF64, d, d)
    for i in eachindex(P.a)
        b = view(P.b, :, i)
        result .+= b * b' ./ (z - P.a[i])
    end
    return result
end

(P::Pole)(Z::AbstractVector{<:Complex}) = map(P, Z)

# evaluate with Gaussian broadening `σ`

function (P::Pole{<:Any,<:AbstractVector})(ω::R, σ::R) where {R<:Real}
    real = zero(R)
    imag = zero(R)
    for i in eachindex(P.a)
        real += abs2(P.b[i]) * sqrt(2) / (π * σ) * dawson((ω - P.a[i]) / (sqrt(2) * σ))
        imag += abs2(P.b[i]) * pdf(Normal(P.a[i], σ), ω)
    end
    result = real - im * imag
    return π .* result # not spectral function
end

function (P::Pole{<:Any,<:AbstractMatrix})(ω::R, σ::R) where {R<:Real}
    d = size(P.b, 1)
    real = zeros(R, d, d)
    imag = zero(real)
    for i in eachindex(P.a)
        b = view(P.b, :, i)
        real .+= b * b' .* sqrt(2) ./ (π * σ) .* dawson((ω - P.a[i]) / (sqrt(2) * σ))
        imag .+= b * b' .* pdf(Normal(P.a[i], σ), ω)
    end
    result = real - im * imag
    return π .* result # not spectral function
end

(P::Pole)(ω::AbstractVector{<:R}, σ::R) where {R<:Real} = map(w -> P(w, σ), ω)

function Core.Array(P::Pole{<:V,<:V}) where {V<:AbstractVector{<:Real}}
    result = Matrix(Diagonal([0; P.a]))
    result[1, 2:end] .= P.b
    result[2:end, 1] .= P.b
    return result
end

# PERF: use convolution theorem and FFT for higher performance

"""
    realKK(A::V, ω::V) where {V<:AbstractVector{<:Real}}

Calculate the real part of a function given its imaginary part `A`
using Kramers-Kronig relations

```math
\\mathrm{Re}~A(ω) = \\frac{1}{π} 𝒫 ∫_{-∞}^∞ \\frac{A(ω')}{ω' - ω} \\mathrm{d}ω'.
```

See also: [`imagKK`](@ref).
"""
function realKK(A::V, ω::V) where {V<:AbstractVector{<:Real}}
    length(A) == length(ω) || throw(DimensionMismatch("length mismatch"))
    result = zero(A)
    # all frequencies ω
    for i in eachindex(A)
        # all frequencies ω'
        for j in eachindex(ω)
            i == j && continue # exclude ω' == ω
            # trapezoidal rule
            j == 1 && continue # skip first index
            result[i] += 0.5 * (A[j - 1] + A[j]) / (ω[j] - ω[i]) * (ω[j] - ω[j - 1])
        end
    end

    result .*= 1 / π
    return result
end

"""
    imagKK(A::V, ω::V) where {V<:AbstractVector{<:Real}}

Calculate the imaginary part of a function given its real part `A`
using Kramers-Kronig relations

```math
\\mathrm{Im}~A(ω) = -\\frac{1}{π} 𝒫 ∫_{-∞}^∞ \\frac{A(ω')}{ω' - ω} \\mathrm{d}ω'.
```

See also: [`realKK`](@ref).
"""
function imagKK(A::V, ω::V) where {V<:AbstractVector{<:Real}}
    return -realKK(A, ω)
end
