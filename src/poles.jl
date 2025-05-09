"""
    Poles(a::AbstractVector{<:Real}, b::AbstractVecOrMat{<:Number})

Representation of poles on the real axis with locations `a`
and amplitudes (not weights) `b`.

`P = Poles(a, b)` represents the sum

```math
P(ω) = ∑_i \\frac{|b_i|^2}{ω-a_i}
```

if both are `a` and `b` are vectors.
If `b` is a matrix, its ``i``-th column is interpreted as a vector ``\\vec{b_i}`` with

```math
P(ω) = \\sum_i \\frac{\\vec{b}_i\\vec{b}_i^\\dagger}{ω-a_i}.
```
"""
struct Poles{A<:AbstractVector{<:Real},B<:AbstractVecOrMat{<:Number}}
    a::A
    b::B

    # both are vectors
    function Poles{A,B}(
        a, b
    ) where {A<:AbstractVector{<:Number},B<:AbstractVector{<:Number}}
        eachindex(a) == eachindex(b) || throw(DimensionMismatch("index mismatch"))
        return new{A,B}(a, b)
    end

    # `b` is a matrix
    function Poles{A,B}(
        a, b
    ) where {A<:AbstractVector{<:Number},B<:AbstractMatrix{<:Number}}
        eachindex(a) == axes(b, 2) || throw(DimensionMismatch("index mismatch"))
        return new{A,B}(a, b)
    end
end

Poles(a::A, b::B) where {A,B} = Poles{A,B}(a, b)

Poles{A,B}(P::Poles) where {A,B} = Poles(A(P.a), B(P.b))

# getters

"""
    locations(P::Poles)

Return the locations of poles.
"""
locations(P::Poles) = P.a

"""
    amplitudes(P::Poles)

Return the amplitudes of poles.
"""
amplitudes(P::Poles) = P.b

"""
   (P::Poles)(z::Complex)

Evaluate `P` at complex frequency `z`.
"""
function (P::Poles{<:Any,<:AbstractVector})(z::Complex)
    result = zero(z)
    for i in eachindex(P.a)
        result += abs2(P.b[i]) / (z - P.a[i])
    end
    return result
end

function (P::Poles{<:Any,<:AbstractMatrix})(z::Complex)
    d = size(P.b, 1)
    result = zeros(ComplexF64, d, d)
    for i in eachindex(P.a)
        b = view(P.b, :, i)
        result .+= b * b' ./ (z - P.a[i])
    end
    return result
end

"""
   (P::Poles)(Z::AbstractVector{<:Complex})

Evaluate `P` at complex frequencies `Z`.
"""
(P::Poles)(Z::AbstractVector{<:Complex}) = map(P, Z)

"""
   (P::Poles)(ω::R, σ::R) where {R<:Real}

Evaluate `P` at frequency `ω` with Gaussian broadening `σ`.
"""
function (P::Poles{<:Any,<:AbstractVector})(ω::R, σ::R) where {R<:Real}
    real = zero(R)
    imag = zero(R)
    for i in eachindex(P.a)
        real += abs2(P.b[i]) * sqrt(2) / (π * σ) * dawson((ω - P.a[i]) / (sqrt(2) * σ))
        imag += abs2(P.b[i]) * pdf(Normal(P.a[i], σ), ω)
    end
    result = real - im * imag
    return π .* result # not spectral function
end

function (P::Poles{<:Any,<:AbstractMatrix})(ω::R, σ::R) where {R<:Real}
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

"""
   (P::Poles)(ω::AbstractVector{<:R}, σ::R) where {R<:Real}

Evaluate `P` at frequencies `ω` with Gaussian broadening `σ`.
"""
(P::Poles)(ω::AbstractVector{<:R}, σ::R) where {R<:Real} = map(w -> P(w, σ), ω)

"""
    spectral_function_loggauss(P::Poles{<:Any,<:AbstractVector}, ω::Real, b::Real)

Calculate the spectral function ``A(ω) = -1/π \\mathrm{Im}[P(ω)]`` with a lognormal broadening.

Each pole is broadened as in NRG

```math
|b_i|^2 δ(ω - a_i) → |b_i|^2 \\frac{\\mathrm{e}^{-b^2/4}}{\\sqrt{π}|a|b}
\\exp\\left(-\\frac{\\ln^2(ω/a_i)}{b^2}\\right).
```

If there is a pole ``a_i = 0``, it is shifted halfway betweeen its neighbors and
each getting half weight

```math
|b_i|^2 δ(ω) →
  \\frac{|b_i|^2}{2} δ\\left(ω - \\frac{a_{i-1}}{2}\\right)
+ \\frac{|b_i|^2}{2} δ\\left(ω - \\frac{a_{i+1}}{2}\\right).
```
"""
function spectral_function_loggauss(P::Poles{<:Any,<:AbstractVector}, ω::Real, b::Real)
    result = zero(ω)
    iszero(ω) && return result # no weight at ω == 0
    for i in eachindex(P.a)
        if iszero(P.a[i])
            # special case, move half of weight to left/right repectively
            issorted(P) || throw(ArgumentError("P is not sorted"))
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
    P::Poles{<:Any,<:AbstractVector}, ω::V, b::Real
) where {V<:AbstractVector{<:Real}}
    return map(w -> spectral_function_loggauss(P, w, b), ω)
end

"""
    weights(P::Poles)

Return the weight(s) of each pole.
"""
weights(P::Poles{<:Any,<:AbstractVector}) = abs2.(P.b)

function weights(P::Poles{<:Any,<:AbstractMatrix})
    amp = amplitudes(P)
    result = map(i -> view(amp, :, i) * view(amp, :, i)', axes(amp, 2))
    return result
end

"""
    moment(P::Poles, n::Int=0)

Return the `n`-th moment.
"""
function moment(P::Poles, n::Int=0)
    result = sum(i -> i[1]^n * i[2], zip(locations(P), weights(P)))
    return result
end

"""
    moments(P::Poles, ns)

Return the `n`-th moment for each `n` in `ns`.
"""
function moments(P::Poles, ns)
    return map(i -> moment(P, i), ns)
end

function _to_grid_square(P::Poles{<:Any,<:AbstractVector}, grid::AbstractVector{<:Real})
    # check input
    foo = Poles(grid, grid)
    issorted(foo) || throw(ArgumentError("grid is not sorted"))
    allunique(foo) || throw(ArgumentError("grid has degenerate locations"))

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
    return Poles(a, b)
end

"""
    to_grid(P::Poles{<:Any,<:AbstractVector}, grid::AbstractVector{<:Real})

Create a new [`Poles`](@ref) from `P` with locations given by `grid`.

A given pole is split locally conserving the zeroth and first moment.
If a pole is outside of `grid`, only the zeroth moment is conserved.
"""
function to_grid(P::Poles{<:Any,<:AbstractVector}, grid::AbstractVector{<:Real})
    result = copy(P)
    result.b .= abs2.(result.b) # use weights
    result = _to_grid_square(result, grid)
    result.b .= sqrt.(result.b) # return to amplitudes
    return result
end

"""
    merge_negative_weight!(P::Poles{<:Any,<:AbstractVector})

Move negative weights of `P` such that the zeroth moment is conserved
and the first moment changes minimally.

Assumes that `P.b` contains weights and not amplitudes.
"""
function merge_negative_weight!(P::Poles{<:Any,<:AbstractVector})
    a = P.a
    b = P.b

    # check input
    length(a) == length(b) || throw(DimensionMismatch("length mismatch"))
    issorted(P) || issorted(P; rev=true) || throw(ArgumentError("P is not sorted"))
    allunique(P) || throw(ArgumentError("P has degenerate locations"))
    all(isreal, P.b) || throw(ArgumentError("weights must be real"))
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

# Poles in continued fraction representation
# P(z) = 1 / (z - a_1 - b_1^2 / (z - a_2 - ⋯))
function _continued_fraction(P::Poles{<:Any,<:AbstractVector})
    # check input
    Base.require_one_based_indexing(P.a)
    Base.require_one_based_indexing(P.b)
    abs(norm(P.b) - 1) < sqrt(eps()) || throw(ArgumentError("Poles P is not normalized"))

    T = eltype(P.b)
    N = length(P)
    A = Diagonal(P.a) # Lanczos on this matrix
    # container for Lanzcos
    a = Vector{real(T)}(undef, N)
    b = Vector{real(T)}(undef, N - 1)
    kryl = Matrix{T}(undef, N, N)
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
    # look if any coefficient b is small
    value, index = findmin(b)
    @debug "smallest weight b=$(value) at index $(index)/$(lastindex(b))"
    return a, b
end

# weights, not amplitudes are stored
function _merge_degenerate_poles_weights!(P::Poles{<:Any,<:AbstractVector}, tol::Real=1e-10)
    a, b = locations(P), amplitudes(P)
    tol >= 0 || throw(ArgumentError("negative tol"))
    issorted(P) || throw(ArgumentError("P is not sorted"))

    # poles at zero
    idx_zeros = findall(i -> abs(i) <= tol, a)
    if !isempty(idx_zeros)
        i0 = popfirst!(idx_zeros)
        a[i0] = 0
        for i in reverse!(idx_zeros)
            b[i0] += popat!(b, i)
            deleteat!(a, i)
        end
    end

    # positive frequencies
    i = findfirst(>(0), a)
    isnothing(i) && (i = lastindex(a))
    while i < lastindex(a)
        if a[i + 1] - a[i] <= tol
            # merge
            b[i] += popat!(b, i + 1)
            deleteat!(a, i + 1) # keep location closer to zero
        else
            # increment index
            i += 1
        end
    end

    # negative frequencies
    i = findlast(<(0), a)
    isnothing(i) && (i = firstindex(a))
    while i > firstindex(a)
        if a[i] - a[i - 1] <= tol
            # merge
            b[i - 1] += popat!(b, i)
            deleteat!(a, i - 1) # keep location closer to zero
            i -= 1
        else
            # decrement index
            i -= 1
        end
    end

    return P
end

"""
    merge_degenerate_poles!(P::Poles{<:Any,<:AbstractVector}, tol::Real=1e-10)

Merge poles whose locations are less than or equal `tol` apart.
"""
function merge_degenerate_poles!(P::Poles{<:Any,<:AbstractVector}, tol::Real=1e-10)
    P.b .= abs2.(P.b)
    _merge_degenerate_poles_weights!(P, tol)
    P.b .= sqrt.(P.b)
    return P
end

"""
    merge_small_poles!(P::Poles{<:Any,<:AbstractVector}, tol::Real=1e-10)

Merge poles with weight `< tol` to its neighbors.

A given pole is split locally conserving the zeroth and first moment.
"""
function merge_small_poles!(P::Poles{<:Any,<:AbstractVector}, tol::Real=1e-10)
    tol > 0 || throw(ArgumentError("negative tol"))
    issorted(P) || issorted(P; rev=true) || throw(ArgumentError("P is not sorted"))

    map!(abs2, P.b, P.b) # use weights
    i = firstindex(P.a)
    while i <= lastindex(P.a)
        pole = P.a[i]
        weight = P.b[i]

        if weight >= tol
            # enough weight, go to next
            i += 1
            continue
        end

        if pole == first(P.a)
            # add weight to next pole
            P.b[i + 1] += weight
            popfirst!(P.a)
            popfirst!(P.b)
        elseif pole == last(P.a)
            # add weight to previous pole
            P.b[i - 1] += weight
            pop!(P.a)
            pop!(P.b)
        else
            # split weight such that zeroth and first moment is conserved
            alow = P.a[i - 1]
            ahigh = P.a[i + 1]
            P.b[i - 1] += (ahigh - pole) / (ahigh - alow) * weight
            P.b[i + 1] += (pole - alow) / (ahigh - alow) * weight
            popat!(P.a, i)
            popat!(P.b, i)
        end
    end
    map!(sqrt, P.b, P.b) # undo squaring
    return P
end

"""
    remove_poles_with_zero_weight!(
        P::Poles{<:Any,<:AbstractVector}, remove_zero::Bool=true
    )

Remove all poles ``|b_i|^2 = 0``.

If `remove_zero`, ``a_i = b_i = 0`` is also removed.

See also [`remove_poles_with_zero_weight`](@ref).
"""
function remove_poles_with_zero_weight!(
    P::Poles{<:Any,<:AbstractVector}, remove_zero::Bool=true
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
        P::Poles{<:Any,<:AbstractVector}, remove_zero::Bool=true
    )

Remove all poles ``|b_i|^2 = 0``.

If `remove_zero`, ``a_i = b_i = 0`` is also removed.

See also [`remove_poles_with_zero_weight!`](@ref).
"""
function remove_poles_with_zero_weight(
    P::Poles{<:Any,<:AbstractVector}, remove_zero::Bool=true
)
    result = copy(P)
    remove_poles_with_zero_weight!(result, remove_zero)
    return result
end

function Core.Array(P::Poles{<:Any,<:AbstractVector{<:Real}})
    T = promote_type(eltype(P.a), eltype(P.b))
    result = Matrix{T}(Diagonal([0; P.a]))
    result[1, 2:end] .= P.b
    result[2:end, 1] .= P.b
    return result
end

Base.copy(P::Poles) = Poles(copy(P.a), copy(P.b))

function Base.length(P::Poles{<:Any,<:AbstractVector})
    length(P.a) == length(P.b) || throw(ArgumentError("length mismatch"))
    return length(P.a)
end

function Base.length(P::Poles{<:Any,<:AbstractMatrix})
    length(P.a) == size(P.b, 2) || throw(ArgumentError("length mismatch"))
    return length(P.a)
end

function Base.allunique(P::Poles)
    l = locations(P)
    # allunique discrimates between ±zero(Float64)
    return allunique(l) && length(findall(iszero, l)) <= 1
end

Base.issorted(P::Poles, args...; kwargs...) = issorted(locations(P), args...; kwargs...)

function Base.sort!(P::Poles{<:Any,<:AbstractVector})
    p = sortperm(P.a)
    P.a[:] = P.a[p]
    P.b[:] = P.b[p]
    return P
end

Base.sort(P::Poles{<:Any,<:AbstractVector}) = sort!(copy(P))

# Subtraction of Poles `result = A - B`.
# After squaring each weight, `A` is put on the same grid as `B`.
# Then, each pole is subtracted `result.b = A.b - B.b.
# If any resulting pole has negative weight,
# it is then shifted around to make all weights non-negative.
function Base.:-(A::Poles{<:Any,<:V}, B::Poles{<:Any,<:V}) where {V<:AbstractVector{<:Real}}
    # check input
    issorted(B) || throw(ArgumentError("B is not sorted"))
    allunique(B) || throw(ArgumentError("B has degenerate locations"))
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
    A = _to_grid_square(A, B.a)
    result.b .+= A.b # difference of weights
    merge_negative_weight!(result)
    map!(sqrt, result.b, result.b) # undo squaring
    return result
end

"""
    inv(P::Poles{<:Any,<:AbstractVector})

Invert `P` → `P_inv` such that

```math
P(z) = ∑_{i=1}^N \\frac{|b_i|^2}{z-a_i}
```

is converted to

```math
P(z)^{1} = \\frac{1}{z - a_0 - \\sum_{i=1}^{N-1} \\frac{|b_i|^2}{z - a_i}} = \\frac{1}{z - a_0 - Q(z)}.
```

Returns `a_0::Real` and `Q::Poles`.

!!! note
    Input `P` must be normalized.
"""
function Base.inv(P::Poles{<:Any,<:AbstractVector})
    a, b = _continued_fraction(P)
    a0 = a[1]
    # take all poles except first and diagonalize
    S = SymTridiagonal(a[2:end], b[2:end])
    a, T = eigen(S)
    b = b[1] * view(T, 1, :)
    map!(abs, b, b) # positive weights easier
    P = Poles(a, b)
    return a0, P
end
