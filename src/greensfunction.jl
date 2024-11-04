"""
    Greensfunction(a::Vector{A}, b::B) where {A,B<:AbstractVecOrMat}

Green's function `G` represented by a sum over poles `G.a` and "residues" `G.b`.

```math
G(z) = ∑_i β_i^† (z𝟏 - α_i)^{-1} β_i
```

with ``z = ω + \\mathrm{i}η ∈ ℂ``.
The coefficients `α` are stored in `G.a` and `β` in `G.b`.
In the most general case `α` and `β` are matrices.
``(z𝟏 - α)^{-1}`` denotes the resolvent.

Can be evaluated at any complex number ``z``.
"""
struct Greensfunction{A,B<:AbstractVecOrMat}
    a::Vector{A}
    b::B

    # both contain scalars
    function Greensfunction{A,B}(a, b) where {A<:Real,B<:AbstractVector{<:Number}}
        length(a) == length(b) || throw(DimensionMismatch("length mismatch"))
        return new{A,B}(a, b)
    end

    # `a` contains scalars, `b` is a matrix
    function Greensfunction{A,B}(a, b) where {A<:Real,B<:AbstractMatrix{<:Number}}
        axes(a, 1) == axes(b, 2) || throw(DimensionMismatch("length mismatch"))
        return new{A,B}(a, b)
    end
end

Greensfunction(a::Vector{A}, b::B) where {A,B} = Greensfunction{A,B}(a, b)

function Greensfunction{A,B}(G::Greensfunction{A,B}) where {A,B}
    return Greensfunction(deepcopy(G.a), deepcopy(G.b))
end

# evaluate with Lorentzian broadening at complex value `z`

function (G::Greensfunction{<:Any,<:AbstractVector})(z::Complex)
    result = zero(z)
    for i in eachindex(G.a)
        result += abs2(G.b[i]) / (z - G.a[i])
    end
    return result
end

# interpret each column in `b` as a row vector $v_i = [b_1i b_2i …]$.
# Then $G_i(z) = v_i^† (z - a_i)^{-1} v_i$ which is a matrix.
# or interpret each column in `b` as a column vector $v_i = [b_1i, b_2i …]$.
# Then $G_i(z) = v_i (z - a_i)^{-1} v_i^†$.
function (G::Greensfunction{<:Any,<:AbstractMatrix})(z::Complex)
    d = size(G.b, 1)
    res = zeros(ComplexF64, d, d)
    for m in axes(G.b, 2), i in axes(G.b, 1), j in axes(G.b, 1)
        res[i, j] += G.b[i, m] * conj(G.b[j, m]) / (z - G.a[m])
    end
    return res
end

(G::Greensfunction)(Z::AbstractVector{<:Complex}) = map(G, Z)

# evaluate with Gaussian broadening `σ`

function (G::Greensfunction{<:Any,<:AbstractVector})(ω::R, σ::R) where {R<:Real}
    real = zero(R)
    imag = zero(R)
    for i in eachindex(G.a)
        real += abs2(G.b[i]) * sqrt(2) / (π * σ) * dawson((ω - G.a[i]) / (sqrt(2) * σ))
        imag += abs2(G.b[i]) * pdf(Normal(G.a[i], σ), ω)
    end
    result = real - im * imag
    return π .* result # not spectral function
end

function (G::Greensfunction{<:Any,<:AbstractMatrix})(ω::R, σ::R) where {R<:Real}
    d = size(G.b, 1)
    real = zeros(R, d, d)
    imag = zeros(R, d, d)
    for m in eachindex(G.a), i in axes(G.b, 1), j in axes(G.b, 1)
        real[i, j] +=
            G.b[i, m] * conj(G.b[j, m]) * sqrt(2) / (π * σ) *
            dawson((ω - G.a[m]) / (sqrt(2) * σ))
        imag[i, j] += G.b[i, m] * conj(G.b[j, m]) * pdf(Normal(G.a[m], σ), ω)
    end
    result = real - im * imag
    return π .* result # not spectral function
end

(G::Greensfunction)(ω::AbstractVector{<:R}, σ::R) where {R<:Real} = map(w -> G(w, σ), ω)

# variable broadening
function (G::Greensfunction{<:Any,<:AbstractVector})(
    ω::AbstractVector{<:R}, σ::AbstractVector{<:R}
) where {R<:Real}
    length(ω) == length(σ) || throw(ArgumentError("length mismatch"))
    result = similar(ω, Complex{R})
    for i in eachindex(ω)
        result[i] = G(ω[i], σ[i])
    end
    return result
end

function (G::Greensfunction{<:Any,<:AbstractMatrix})(
    ω::AbstractVector{<:R}, σ::AbstractVector{<:R}
) where {R<:Real}
    length(ω) == length(σ) || throw(ArgumentError("length mismatch"))
    result = similar(ω, Matrix{Complex{R}})
    for i in eachindex(ω)
        result[i] = G(ω[i], σ[i])
    end
    return result
end

# write to a filepath, overwriting contents
function Base.write(
    s::AbstractString, G::Greensfunction{<:Number,<:AbstractArray{<:Number}}
)
    h5open(s, "w") do fid
        fid["a"] = G.a
        fid["b"] = G.b
    end
    return nothing
end

# read from a filepath
function Greensfunction{A,B}(s::AbstractString) where {A<:Number,B<:AbstractArray{<:Number}}
    return h5open(s, "r") do fid
        a::Vector{A} = read(fid, "a")
        b::B = read(fid, "b")
        return Greensfunction(a, b)
    end
end

"""
    get_hyb(n_bath::Int, t::Real=1.0)

Return non-interacting `Greensfunction` for hybridization with hopping `t`.
"""
function get_hyb(n_bath::Int, t::Real=1.0)
    α = zeros(n_bath)
    β = fill(t, n_bath - 1)
    H0 = SymTridiagonal(α, β)
    E, T = eigen(H0)

    return Greensfunction(E, abs.(T[:, 1]))
end

"""
    get_hyb_equal(n_bath::Int, t::Real=1.0)

Return non-interacting `Greensfunction`
for given amount of bath site `n_bath`.

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
    return Greensfunction(a, b)
end

function Core.Array(G::Greensfunction{<:T,<:AbstractVector{<:T}}) where {T<:Real}
    result = Matrix(Diagonal([zero(T); G.a]))
    result[1, 2:end] .= G.b
    result[2:end, 1] .= G.b
    return result
end

# PERF: use convolution theorem and FFT for higher performance

"""
    realKK(A::AbstractVector{<:T}, ω::AbstractVector{<:Real}) where {T<:Real}

Calculate the real part of a function given its imaginary part `A`
using Kramers-Kronig relations

```math
\\mathrm{Re}~A(ω) = \\frac{1}{π} 𝒫 ∫_{-∞}^∞ \\frac{A(ω')}{ω' - ω} \\mathrm{d}ω'.
```

See also: [`imagKK`](@ref).
"""
function realKK(A::AbstractVector{<:Real}, ω::AbstractVector{<:Real})
    length(A) == length(ω) || throw(ArgumentError("vector length mismatch"))
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
    imagKK(A::AbstractVector{<:T}, ω::AbstractVector{<:Real}) where {T<:Real}

Calculate the imaginary part of a function given its real part `A`
using Kramers-Kronig relations

```math
\\mathrm{Im}~A(ω) = -\\frac{1}{π} 𝒫 ∫_{-∞}^∞ \\frac{A(ω')}{ω' - ω} \\mathrm{d}ω'.
```

See also: [`realKK`](@ref).
"""
function imagKK(A::AbstractVector{<:Real}, ω::AbstractVector{<:Real})
    return -realKK(A, ω)
end
