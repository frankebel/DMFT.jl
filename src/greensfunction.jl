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

"""
    Greensfunction(
        A::AbstractVector{AbstractMatrix{<:Real}},
        B::AbstractVector{AbstractMatrix{<:T}},
        E0::Real,
        S_sqrt::AbstractMatrix{<:T},
    ) where {T<:Number}

Diagonalize blocktridiagonal matrix given by `A`, `B` and convert to `Greensfunction`.

`E0` is the ground state energy which is subtracted from each eigenvalue.
`S_sqrt` is the transformation matrix to go back to the original basis.
"""
function Greensfunction(
    A::AbstractVector{<:AbstractMatrix{<:Real}},
    B::AbstractVector{<:AbstractMatrix{<:T}},
    E0::Real,
    S_sqrt::AbstractMatrix{<:T},
) where {T<:Number}
    # NOTE: change once compatibility is set to >= Julia 1.11
    # X = Array(BlockTridiagonal(B, A, map(adjoint, B)))
    X = Array(BlockTridiagonal(B, A, Matrix.(adjoint.(B))))
    E, V = LAPACK.syev!('V', 'U', X)
    poles = E .- E0
    R = S_sqrt * V[1:size(S_sqrt, 1), :]
    return Greensfunction(poles, R)
end

# evaluate at point `z`

function (G::Greensfunction{<:Real,<:AbstractVector{<:Number}})(z::Complex)
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
function (G::Greensfunction{<:Real,<:AbstractMatrix{<:Number}})(z::Complex)
    d = size(G.b, 1)
    res = zeros(ComplexF64, d, d)
    for m in axes(G.b, 2), i in axes(G.b, 1), j in axes(G.b, 1)
        res[i, j] += G.b[i, m] * conj(G.b[j, m]) / (z - G.a[m])
    end
    return res
end

(G::Greensfunction)(Z::AbstractVector{<:Complex}) = map(G, Z)

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
        a = read(fid, "a")
        b = read(fid, "b")
        return Greensfunction{A,B}(Vector{A}(a), B(b))
    end
end

"""
    get_hyb(n_bath::Int, t::Real=1.0)

Return non-interacting `Greensfunction` for hybridization with hopping `t`.
"""
function get_hyb(n_bath::Int, t::Real=1.0)
    isodd(n_bath) || throw(ArgumentError("n_bath must be odd"))

    α = zeros(n_bath)
    β = fill(t, n_bath - 1)
    H0 = SymTridiagonal(α, β)
    E, T = eigen(H0)

    # overwrite 0-peak energy
    E[n_bath ÷ 2 + 1] = 0

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

    a = [-reverse(a); 0; a]
    b = fill(V, n_bath)
    return Greensfunction(a, b)
end

function Core.Array(G::Greensfunction{<:T,<:AbstractVector{<:T}}) where {T<:Real}
    result = Matrix(Diagonal([zero(T); G.a]))
    result[1, 2:end] .= G.b
    result[2:end, 1] .= G.b
    return result
end
