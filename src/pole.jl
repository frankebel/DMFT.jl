"""
    Pole{A<:AbstractVector,B<:AbstractVecOrMat}

Pole representation in the upper complex plane with locations `a::A` and weights `b::B`.

If both are `A` and `B` are vectors, it is just a sum:

```math
P(z) = ∑_i \\frac{|b_i|^2}{z-a_i}
```

If `B` is a matrix, its ``i``-th column is interpreted as the vector ``\\vec{b_i}`` with

```math
P(z) = \\sum_i \\frac{\\vec{b}_i\\vec{b}_i^\\dagger}{z-a_i}
```

Can also be evaluated by Gaussian broadening `P(ω, σ)` with a fixed or variable broadening.
"""
struct Pole{A<:AbstractVector,B<:AbstractVecOrMat}
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

# variable broadening
function (P::Pole{<:Any,<:AbstractVector})(
    ω::AbstractVector{<:R}, σ::AbstractVector{<:R}
) where {R<:Real}
    length(ω) == length(σ) || throw(DimensionMismatch("length mismatch"))
    result = similar(ω, Complex{R})
    for i in eachindex(ω)
        result[i] = P(ω[i], σ[i])
    end
    return result
end

function (P::Pole{<:Any,<:AbstractMatrix})(
    ω::AbstractVector{<:R}, σ::AbstractVector{<:R}
) where {R<:Real}
    length(ω) == length(σ) || throw(DimensionMismatch("length mismatch"))
    result = similar(ω, Matrix{Complex{R}})
    for i in eachindex(ω)
        result[i] = P(ω[i], σ[i])
    end
    return result
end

# write to a filepath, overwriting contents
function Base.write(s::AbstractString, P::Pole{<:Any,<:AbstractArray{<:Number}})
    h5open(s, "w") do fid
        fid["a"] = P.a
        fid["b"] = P.b
    end
    return nothing
end

# read from a filepath
function Pole{A,B}(s::AbstractString) where {A,B}
    return h5open(s, "r") do fid
        a::A = read(fid, "a")
        b::B = read(fid, "b")
        return Pole{A,B}(a, b)
    end
end

"""
    get_hyb(n_bath::Int, t::Real=1.0)

Return the semicircular density of states with hopping `t` discretized on `n_bath` poles.
"""
function get_hyb(n_bath::Int, t::Real=1.0)
    α = zeros(n_bath)
    β = fill(t, n_bath - 1)
    H0 = SymTridiagonal(α, β)
    E, T = eigen(H0)

    return Pole(E, abs.(T[:, 1]))
end

"""
    get_hyb_equal(n_bath::Int, t::Real=1.0)

Return the semicircular density of states with hopping `t` discretized on `n_bath` poles.

Each site has the same hybridization `V_k^2 = 1/n_bath`.
"""
function get_hyb_equal(n_bath::Int, t::Real=1.0)
    isodd(n_bath) || throw(ArgumentError("n_bath must be odd"))

    D = 2 * t # half-bandwidth
    V_sqr = 1 / n_bath
    V = sqrt(V_sqr)
    s = Semicircle(D)

    # calculate only negative half, mirror due to symmetry
    q = collect(0:V_sqr:0.5) # equal weight for each pole
    v = quantile.(Semicircle(D), q) # I_l

    # ϵ_l = 1/V_sqr ∫_{I_l} dω ω f(ω)
    # trapezoid rule with `n_p` points
    a = Float64[]
    n_p = 128 # arbitrary number
    for i in eachindex(v)
        i == length(v) && break
        # subtract half of border values
        α = -v[i] * pdf(s, v[i]) - v[i + 1] * pdf(s, v[i + 1])
        α /= 2
        for j in LinRange(v[i], v[i + 1], n_p)
            α += j * pdf(s, j)
        end
        α *= (v[i + 1] - v[i]) / n_p # Δω = I_l/n_p
        push!(a, α)
    end
    a .*= n_bath # a .*= 1/V_sqr

    a = [a; 0; -reverse(a)]
    b = fill(V, n_bath)
    return Pole(a, b)
end

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
