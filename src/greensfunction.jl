"""
    Greensfunction(a::A, b::B) where {A<:AbstractVecOrMat,B<:AbstractVecOrMat}

Green's function `G` represented by a sum over poles `G.a` and "residues" `G.b`.

```math
G(z) = ∑_i β_i^† (z𝟏 - α_i)^{-1} β_i
```

with ``z = ω + \\mathrm{i}η ∈ ℂ``.
The coefficients `α` are stored in `G.a` and `β` in `G.b`.
In the most general case `α` and `β` are matrices.
``(z𝟏 - α)^{-1}`` denotes the resolvent.

Can be evaluated at any complex number ``z``.

Currently limited to `α`, `β` being numbers (single orbital case)
which simplifies the calculation to:

```math
G(z) = ∑_i \\frac{|β_i|^2}{z - α_i}
```
"""
struct Greensfunction{A<:AbstractVecOrMat,B<:AbstractVecOrMat}
    a::A
    b::B

    function Greensfunction{A,B}(a, b) where {A,B}
        length(a) == length(b) || throw(ArgumentError("length mismatch"))
        return new{A,B}(a, b)
    end
end

Greensfunction(a::A, b::B) where {A,B} = Greensfunction{A,B}(a, b)

function (G::Greensfunction{<:AbstractVector{<:Number},<:AbstractVector{<:Number}})(
    z::C
) where {C<:Complex}
    result = zero(C)
    for i in eachindex(G.a)
        result += abs2(G.b[i]) / (z - G.a[i])
    end
    return result
end

function (G::Greensfunction{<:AbstractVector{<:Number},<:AbstractVector{<:Number}})(
    Z::AbstractVector{<:Complex}
)
    return map(z -> G(z), Z)
end

# write to a filepath, overwriting contents
function Base.write(
    s::AbstractString,
    G::Greensfunction{<:AbstractVector{<:Number},<:AbstractVector{<:Number}},
)
    h5open(s, "w") do fid
        fid["a"] = G.a
        fid["b"] = G.b
    end
    return nothing
end

# read from a filepath
function Greensfunction{A,B}(
    s::AbstractString
) where {A<:AbstractVector{<:Number},B<:AbstractVector{<:Number}}
    return h5open(s, "r") do fid
        a = read(fid, "a")
        b = read(fid, "b")
        return Greensfunction{A,B}(A(a), B(b))
    end
end

# default Vector{Float64}
function Greensfunction(s::AbstractString)
    V = Vector{Float64}
    return Greensfunction{V,V}(s)
end

"""
    Hybridizationfunction{M,G<:Greensfunction}

Hybridization function `Δ` represented by a sum over poles and "residues".

A Hybridizationfunction describes how an impurity couples to a discretized bath
with positive energies `Δ.pos.a` and coupling terms `Δ.pos.b`;
same for negative energies (`Δ.neg`)
`V0` is the coupling to a bath site with zero energy
If `V0` is a Vector and `Δ.pos.b`, `Δ.neg.b` are matrices,
it describes a multi orbital impurity.
A `Hybridizationfunction` is callable similar to a Greensfunction: `Δ(z)`.

See also [`Greensfunction`](@ref).
"""
struct Hybridizationfunction{M,G<:Greensfunction}
    V0::M
    pos::G
    neg::G
end

function (Δ::Hybridizationfunction)(z::Complex)
    return Δ.pos(z) + Δ.neg(z) + (Δ.V0' * Δ.V0 ./ z)
end

(Δ::Hybridizationfunction)(Z::AbstractVector{<:Complex}) = map(z -> Δ(z), Z)

# write to a filepath, overwriting contents
function Base.write(
    s::AbstractString,
    Δ::Hybridizationfunction{
        <:Number,<:Greensfunction{<:AbstractVector{<:Number},<:AbstractVector{<:Number}}
    },
)
    h5open(s, "w") do fid
        fid["V0"] = Δ.V0
        fid["pos/a"] = Δ.pos.a
        fid["pos/b"] = Δ.pos.b
        fid["neg/a"] = Δ.neg.a
        fid["neg/b"] = Δ.neg.b
    end
    return nothing
end

# read from a filepath
function Hybridizationfunction{M,G}(s::AbstractString) where {M,G<:Greensfunction}
    return h5open(s, "r") do fid
        V0 = read(fid, "V0")
        a_p = read(fid, "pos/a")
        b_p = read(fid, "pos/b")
        a_n = read(fid, "neg/a")
        b_n = read(fid, "neg/b")
        V0 = M(V0)
        g_p = G(a_p, b_p)
        g_n = G(a_n, b_n)
        return Hybridizationfunction{M,G}(V0, g_p, g_n)
    end
end

# default Float64, Vector{Float64}
function Hybridizationfunction(s::AbstractString)
    M = Float64
    V = Vector{Float64}
    G = Greensfunction{V,V}
    return Hybridizationfunction{M,G}(s)
end

"""
    get_hyb(n_bath::Int, t::Real=1.0, ϵ::Real=1E-8)

Return non-interacting `HybridizationFunction`.

### Arguents:
- `n_bath`: number of bath sites
- `t`: hopping amplitude
- `ϵ`: all energies `abs(e) < ϵ` contribute to `V0`.

See also [`Hybridizationfunction`](@ref).
"""
function get_hyb(n_bath::Int, t::Real=1.0, ϵ::Real=1E-8)
    isodd(n_bath) || throw(ArgumentError("n_bath must be odd"))

    α = zeros(n_bath)
    β = t * ones(n_bath - 1)
    H0 = SymTridiagonal(α, β)
    E, T = eigen(H0)

    V0_sqr = zero(Float64)
    ap = Float64[]
    bp = Float64[]
    an = Float64[]
    bn = Float64[]

    for i in eachindex(E)
        if abs(E[i]) < ϵ
            V0_sqr += T[1, i]^2
        elseif E[i] < 0.0
            push!(an, E[i])
            push!(bn, abs(T[1, i]))
        else
            push!(ap, E[i])
            push!(bp, abs(T[1, i]))
        end
    end

    V0 = sqrt(V0_sqr)
    pos = Greensfunction(ap, bp)
    neg = Greensfunction(an, bn)
    return Hybridizationfunction(V0, pos, neg)
end

"""
    get_hyb_equal(n_bath::Int, t::Real=1.0)

Return non-interacting `HybridizationFunction`
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

    # HybridizationFunction
    b = fill(V, n_bath ÷ 2)
    pos = Greensfunction(-reverse(a), b)
    neg = Greensfunction(a, b)
    return Hybridizationfunction(V, pos, neg)
end
