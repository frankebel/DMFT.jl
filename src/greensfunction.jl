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
    return h5open(s, "r") do fid
        V = Vector{Float64}
        a = read(fid, "a")
        b = read(fid, "b")
        return Greensfunction{V,V}(V(a), V(b))
    end
end
