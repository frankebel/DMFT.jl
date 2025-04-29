"""
    Pole{A<:AbstractVector{<:Real},B<:AbstractVecOrMat{<:Number}}

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
struct Pole{A<:AbstractVector{<:Real},B<:AbstractVecOrMat{<:Number}}
    a::A
    b::B

    # both are vectors
    function Pole{A,B}(a, b) where {A<:AbstractVector{<:Number},B<:AbstractVector{<:Number}}
        eachindex(a) == eachindex(b) || throw(DimensionMismatch("index mismatch"))
        return new{A,B}(a, b)
    end

    # `b` is a matrix
    function Pole{A,B}(a, b) where {A<:AbstractVector{<:Number},B<:AbstractMatrix{<:Number}}
        eachindex(a) == axes(b, 2) || throw(DimensionMismatch("index mismatch"))
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
    spectral_function_loggauss(P::Pole{<:Any,<:AbstractVector{<:Real}}, ω::Real, b::Real)

Calculate the spectral function ``A(ω) = -1/π \\mathrm{Im}[P(ω)]`` with a lognormal broadening.

Each pole is broadened as in NRG

```math
|b_i|^2 δ(ω - a_i) → |b_i|^2 \\frac{\\mathrm{e}^{-b^2/4}}{\\sqrt{π}|a|b}
\\exp\\left(-\\frac{\\ln^2(ω/a_i)}{b^2}\\right).
```

If there is a pole ``a_i = 0``, it is shifted halfway betweeen its neighbors and
each getting half weight

```math
|b_i|^2 δ(ω) =
  \\frac{|b_i|^2}{2} δ\\left(ω - \\frac{a_{i-1}}{2}\\right)
+ \\frac{|b_i|^2}{2} δ\\left(ω - \\frac{a_{i+1}}{2}\\right).
```
"""
function spectral_function_loggauss(
    P::Pole{<:Any,<:AbstractVector{<:Real}}, ω::Real, b::Real
)
    result = zero(ω)
    iszero(ω) && return result # no weight at ω == 0
    for i in eachindex(P.a)
        if iszero(P.a[i])
            # special case, move half of weight to left/right repectively
            issorted(P.a) || throw(ArgumentError("P is not sorted"))
            location = ω > 0 ? P.a[i + 1] / 2 : P.a[i - 1] / 2
            weight = abs2.(P.b[i]) / 2
        elseif sign(ω) == sign(P.a[i])
            # only contribute weight if on same side of real axis
            weight = abs2.(P.b[i])
            location = P.a[i]
        else
            # frequency has opposite sign compared to pole location
            continue
        end
        prefactor = weight * exp(-b^2 / 4) / (b * abs(location) * sqrt(π))
        result += prefactor * exp(-(log(ω / location) / b)^2)
    end
    return result
end

function spectral_function_loggauss(
    P::Pole{<:Any,<:V}, ω::V, b::Real
) where {V<:AbstractVector{<:Real}}
    return map(w -> spectral_function_loggauss(P, w, b), ω)
end

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

# Pole in continued fraction representation
# P(z) = 1 / (z - a_1 - b_1^2 / (z - a_2 - ⋯))
function _continued_fraction(P::Pole{<:V,<:V}) where {V<:AbstractVector{<:Real}}
    # check input
    Base.require_one_based_indexing(P.a)
    Base.require_one_based_indexing(P.b)
    abs(norm(P.b) - 1) < sqrt(eps()) || throw(ArgumentError("Pole P is not normalized"))

    R = eltype(V)
    N = length(P)
    A = Diagonal(P.a) # Lanczos on this matrix
    # container for Lanzcos
    a = Vector{R}(undef, N)
    b = Vector{R}(undef, N - 1)
    kryl = Matrix{R}(undef, N, N)
    # first Lanczos step
    kryl[:, 1] = P.b  # input vector
    v_new = A * P.b
    a[1] = P.b ⋅ v_new
    v_new .-= a[1] * P.b
    for i in 2:N
        # all other Lanczos steps
        b[i - 1] = norm(v_new)
        rmul!(v_new, inv(b[i - 1]))
        kryl[:, i] = v_new
        mul!(v_new, A, view(kryl, :, i)) # new vector
        a[i] = view(kryl, :, i) ⋅ v_new
        for j in 1:i
            # orthogonalize against all previous states
            v_old = view(kryl, :, j)
            v_new .-= (v_old ⋅ v_new) * v_old
        end
    end
    return a, b
end

"""
    merge_equal_poles!(P::Pole{V,V}, tol::Real=1e-10) where {V<:AbstractVector{<:Real}}

Merge poles which are less than `tol` apart.

If `P.a[i+1] - P.a[i] < tol`, then add up weights and pop index `i+1`.
"""
function merge_equal_poles!(P::Pole{V,V}, tol::Real=1e-10) where {V<:AbstractVector{<:Real}}
    a, b = P.a, P.b
    tol > 0 || throw(ArgumentError("negative tol"))
    issorted(a) || throw(ArgumentError("poles are not sorted"))

    i = firstindex(a)
    while i < lastindex(a)
        if a[i + 1] - a[i] < tol
            # merge
            b[i] = sqrt(abs2(b[i]) + abs2(b[i + 1]))
            popat!(a, i + 1)
            popat!(b, i + 1)
        else
            # increment index
            i += 1
        end
    end
    return P
end

"""
    remove_poles_with_zero_weight!(
        P::Pole{<:Any,<:AbstractVector{<:Number}}, remove_zero::Bool=true
    )

Remove all poles ``|b_i|^2 = 0``.

If `remove_zero`, ``a_i = b_i = 0`` is also removed.

See also: [`remove_poles_with_zero_weight`](@ref).
"""
function remove_poles_with_zero_weight!(
    P::Pole{<:Any,<:AbstractVector{<:Number}}, remove_zero::Bool=true
)
    i = firstindex(P.b)
    while i <= lastindex(P.b)
        if iszero(P.a[i]) && !remove_zero
            # keep pole at zero energy
            i += 1
            continue
        end

        if iszero(P.b[i])
            popat!(P.a, i)
            popat!(P.b, i)
        else
            i += 1
        end
    end
    return P
end

"""
    remove_poles_with_zero_weight(
        P::Pole{<:Any,<:AbstractVector{<:Number}}, remove_zero::Bool=true
    )

Remove all poles ``|b_i|^2 = 0``.

If `remove_zero`, ``a_i = b_i = 0`` is also removed.

See also: [`remove_poles_with_zero_weight!`](@ref).
"""
function remove_poles_with_zero_weight(
    P::Pole{<:Any,<:AbstractVector{<:Number}}, remove_zero::Bool=true
)
    result = copy(P)
    remove_poles_with_zero_weight!(result, remove_zero)
    return result
end

function Core.Array(P::Pole{<:Any,<:V}) where {V<:AbstractVector{<:Real}}
    T = promote_type(eltype(P.a), eltype(P.b))
    result = Matrix{T}(Diagonal([0; P.a]))
    result[1, 2:end] .= P.b
    result[2:end, 1] .= P.b
    return result
end

Base.copy(P::Pole) = Pole(copy(P.a), copy(P.b))

function Base.length(P::Pole)
    length(P.a) == length(P.b) || throw(ArgumentError("length mismatch"))
    return length(P.a)
end

function Base.sort!(P::Pole{<:Any,<:AbstractVector})
    p = sortperm(P.a)
    P.a[:] = P.a[p]
    P.b[:] = P.b[p]
    return P
end

Base.sort(P::Pole{<:Any,<:AbstractVector}) = sort!(copy(P))

# Subtraction of Poles `result = A - B`.
# After squaring each weight, `A` is put on the same grid as `B`.
# Then, each pole is subtracted `result.b = A.b - B.b.
# If any resulting pole has negative weight,
# it is then shifted around to make all weights non-negative.
function Base.:-(A::Pole{<:V,<:V}, B::Pole{<:V,<:V}) where {V<:AbstractVector{<:Real}}
    # check input
    issorted(B.a) || throw(ArgumentError("A is not sorted"))
    allunique(B.a) || throw(ArgumentError("degenerate energies in B"))
    length(A.a) == length(A.b) || throw(DimensionMismatch("length mismatch in A"))
    length(B.a) == length(B.b) || throw(DimensionMismatch("length mismatch in B"))

    # create copies to keep original unchanged
    # work with squared weights from here on
    result = copy(B)
    for i in eachindex(result.b)
        result.b[i] = -abs2(result.b[i])
    end
    A = copy(A)
    map!(abs2, A.b, A.b)
    A = to_grid_sqr(A, B.a)
    result.b .+= A.b # difference of weights
    move_negative_weight_to_neighbors!(result)
    map!(sqrt, result.b, result.b) # undo squaring
    return result
end

"""
    inv(P::Pole{<:V,<:V}) where V<:AbstractVector{<:Real}

Invert `P` → `P_inv` such that

```math
P(z) = ∑_{i=1}^N \\frac{|b_i|^2}{z-a_i}
```

is converted to

```math
P(z)^{1} = \\frac{1}{z - a_0 - \\sum_{i=1}^{N-1} \\frac{|b_i|^2}{z - a_i}} = \\frac{1}{z - a_0 - Q(z)}.
```

Returns `a_0::Real` and `Q::Pole`.

!!! note
    Input `P` must be normalized.
"""
function Base.inv(P::Pole{<:V,<:V}) where {V<:AbstractVector{<:Real}}
    P = remove_poles_with_zero_weight(P)
    a, b = _continued_fraction(P)
    a0 = a[1]
    # take all poles except first and diagonalize
    S = SymTridiagonal(a[2:end], b[2:end])
    a, T = eigen(S)
    b = b[1] * view(T, 1, :)
    map!(abs, b, b) # positive weights easier
    P = Pole(a, b)
    return a0, P
end
