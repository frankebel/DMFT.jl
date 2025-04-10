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

"""
    to_grid_sqr(P::Pole{<:V,<:V}, grid::V) where {V<:AbstractVector{<:Real}}

Create a new [`Pole`](@ref) from `P` on with locations given by `grid`.

A given pole is split locally conserving the zeroth and first moment.
If the pole is lower than the lowest value in the grid or
higher than the highest value, only the zeroth moment is conserved.

Assumes that weights are already squared and keeps them squred.

See also: [`to_grid`](@ref).
"""
function to_grid_sqr(P::Pole{<:V,<:V}, grid::V) where {V<:AbstractVector{<:Real}}
    # check input
    length(P.a) == length(P.b) || throw(DimensionMismatch("length mismatch in P"))
    issorted(grid) || throw(ArgumentError("grid is not sorted"))
    allunique(grid) || throw(ArgumentError("degenerate locations in grid"))

    # new location of poles
    a = copy(grid)
    b = zero(grid)
    # run through each pole and split weight to neighbors
    @inbounds for i in eachindex(P.a)
        pole = P.a[i]
        weight = P.b[i]
        if pole <= first(grid)
            # no pole to the left
            b[begin] += weight
        elseif pole >= last(grid)
            # no pole to the right
            b[end] += weight
        else
            # find next pole with higher location
            i = searchsortedfirst(grid, pole)
            if pole - grid[i - 1] < 10 * eps()
                # previous pole has same location
                b[i - 1] += weight
            elseif grid[i] - pole < 10 * eps()
                # current pole has same location
                b[i] += weight
            else
                # split such that zeroth and first moment is conserved
                alow = grid[i - 1]
                ahigh = grid[i]
                b[i - 1] += (ahigh - pole) / (ahigh - alow) * weight
                b[i] += (pole - alow) / (ahigh - alow) * weight
            end
        end
    end
    return Pole(a, b)
end

"""
    to_grid(P::Pole{<:V,<:V}, grid::V) where {V<:AbstractVector{<:Real}}

Create a new [`Pole`](@ref) from `P` on with locations given by `grid`.

A given pole is split locally conserving the zeroth and first moment.
If the pole is lower than the lowest value in the grid or
higher than the highest value, only the zeroth moment is conserved.

See also: [`to_grid_sqr`](@ref).
"""
function to_grid(P::Pole{<:V,<:V}, grid::V) where {V<:AbstractVector{<:Real}}
    result = copy(P)
    map!(abs2, result.b, result.b) # square weights
    result = to_grid_sqr(result, grid)
    map!(sqrt, result.b, result.b) # undo squaring
    return result
end

"""
    move_negative_weight_to_neighbors!(P::Pole{<:V,<:V}) where {V<:AbstractVector{<:Real}}

Move negative weights of `P` such that the zeroth moment is conserved
and the first moment changes minimally.

Assumes that weights are squared and keeps them in that form.
"""
function move_negative_weight_to_neighbors!(
    P::Pole{<:V,<:V}
) where {V<:AbstractVector{<:Real}}
    a = P.a
    b = P.b

    # check input
    length(a) == length(b) || throw(DimensionMismatch("length mismatch"))
    issorted(a) || issorted(a; rev=true) || throw(ArgumentError("grid is not sorted"))
    allunique(a) || throw(ArgumentError("degenerate locations in grid"))
    sum(b) >= 0 || throw(ArgumentError("total weight is negative"))
    firstindex(a) == firstindex(b) || throw(ArgumentError("input uses different indexing"))

    for i in eachindex(b)
        b[i] >= 0 && continue # no negative weight, go to next
        if i == lastindex(b)
            # find previous positive weight
            for j in Iterators.reverse(firstindex(b):(i - 1))
                iszero(b[j]) && continue
                if b[j] + b[end] >= 0
                    # b[j] can fully compensate b[end]
                    b[j] += b[end]
                    b[end] = 0
                    break
                else
                    # b[j] can't fully compensate b[end]
                    b[end] += b[j]
                    b[j] = 0
                end
            end
        else
            for j in Iterators.reverse(firstindex(b):(i - 1))
                # find a previous pole with positive weight
                iszero(b[j]) && continue
                # calculate fractions how weight should be split
                f_left = (a[i + 1] - a[i]) / (a[i + 1] - a[j])
                f_right = 1 - f_left
                if b[j] + f_left * b[i] >= 0
                    # b[j] can fully compensate b[i]
                    b[j] += f_left * b[i]
                    b[i + 1] += f_right * b[i]
                    b[i] = 0
                else
                    # b[j] can't fully compensate b[i].
                    # Find fraction f ∈ (0, 1) which can be merged such that b[j] gets 0 weight.
                    # b_j + f f_l b_i === 0
                    b[i] += b[j] / f_left
                    b[i + 1] -= f_right / f_left * b[j]
                    b[j] = 0
                end
                if j == firstindex(a)
                    # no pole with positive weight remaining
                    b[i + 1] += b[i]
                    b[i] = 0
                end
            end
            if b[i] <= 0
                # negative weight remaining and no previous weight to compensate
                # move negative weight to next pole
                b[i + 1] += b[i]
                b[i] = 0
            end
        end
    end
    sum(b) >= 0 || throw(ArgumentError("total weight got negative"))
    return P
end

function Core.Array(P::Pole{<:V,<:V}) where {V<:AbstractVector{<:Real}}
    result = Matrix(Diagonal([0; P.a]))
    result[1, 2:end] .= P.b
    result[2:end, 1] .= P.b
    return result
end

Base.copy(P::Pole) = Pole(copy(P.a), copy(P.b))

function Base.sort!(P::Pole{<:Any,<:AbstractVector})
    p = sortperm(P.a)
    P.a[:] = P.a[p]
    P.b[:] = P.b[p]
    return P
end

Base.sort(P::Pole{<:Any,<:AbstractVector}) = sort!(copy(P))

# Subtraction of Poles `result = A - B`.
# After squaring each weight, `B` is put on the same grid as `A`.
# Then, each pole is subtracted `result.b = A.b - B.b.
# If any resulting pole has negative weight,
# it is then shifted around to make all weights non-negative.
function Base.:-(A::Pole{<:V,<:V}, B::Pole{<:V,<:V}) where {V<:AbstractVector{<:Real}}
    # check input
    issorted(A.a) || throw(ArgumentError("A is not sorted"))
    allunique(A.a) || throw(ArgumentError("degenerate energies in A"))
    length(A.a) == length(A.b) || throw(DimensionMismatch("length mismatch in A"))
    length(B.a) == length(B.b) || throw(DimensionMismatch("length mismatch in B"))

    # create copies to keep original unchanged
    # work with squared weights from here on
    result = copy(A)
    map!(abs2, result.b, result.b)
    B = copy(B)
    map!(abs2, B.b, B.b)
    B = to_grid_sqr(B, A.a)
    result.b .-= B.b # subtract poles
    move_negative_weight_to_neighbors!(result)
    map!(sqrt, result.b, result.b) # undo squaring
    return result
end
