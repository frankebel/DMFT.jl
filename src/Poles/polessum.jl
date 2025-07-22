"""
    PolesSum{A<:Real,B<:Number} <: AbstractPolesSum

Representation of poles on the real axis with locations ``a_i`` of type `A`
and weights ``w_i`` of type `B`

```math
P(ω) = ∑_i \\frac{w_i}{ω-a_i}.
```

For a block variant see [`PolesSumBlock`](@ref).
"""
struct PolesSum{A<:Real,B<:Number} <: AbstractPolesSum
    loc::Vector{A} # locations of poles
    wgt::Vector{B} # weights of poles

    function PolesSum{A,B}(loc, wgt) where {A,B}
        length(loc) == length(wgt) || throw(DimensionMismatch("length mismatch"))
        return new{A,B}(loc, wgt)
    end
end

"""
    PolesSum(loc::AbstractVector{A}, wgt::AbstractVector{B}) where {A,B}

Create a new instance of [`PolesSum`](@ref) by supplying the locations `loc`
and weights `wgt`.

```jldoctest
julia> loc = collect(0:5);

julia> wgt = collect(5:10);

julia> P = PolesSum(loc, wgt)
6-element PolesSum{Int64, Int64}

julia> locations(P) === loc
true

julia> weights(P) === wgt
true
```
"""
function PolesSum(loc::AbstractVector{A}, wgt::AbstractVector{B}) where {A,B}
    return PolesSum{A,B}(loc, wgt)
end

# convert type
function PolesSum{A,B}(P::PolesSum) where {A,B}
    return PolesSum{A,B}(Vector{A}(locations(P)), Vector{B}(weights(P)))
end

"""
    amplitude(P::PolesSum{<:Any,<:Real}, i::Integer)

Return the amplitude (`sqrt` of weight) of `P` at index `i`.

See also [`amplitudes`](@ref).
"""
amplitude(P::PolesSum{<:Any,<:Real}, i::Integer) = sqrt(weights(P)[i])

"""
    amplitudes(P::PolesSum{<:Any,<:Real})

Return the amplitudes (`sqrt` of weights) of `P`.

See also [`amplitude`](@ref).
"""
amplitudes(P::PolesSum{<:Any,<:Real}) = sqrt.(weights(P))

function evaluate_gaussian(P::PolesSum, ω::Real, σ::Real)
    real = zero(ω)
    imag = zero(ω)
    for i in eachindex(P)
        w = weights(P)[i]
        real += w * sqrt(2) / (π * σ) * dawson((ω - locations(P)[i]) / (sqrt(2) * σ))
        imag += w * pdf(Normal(locations(P)[i], σ), ω)
    end
    result = real - im * imag
    return π * result # not spectral function
end

function evaluate_lorentzian(P::PolesSum, ω::Real, δ::Real)
    result = zero(complex(ω))
    for i in eachindex(P)
        result += DMFT.weights(P)[i] / (ω + im * δ - locations(P)[i])
    end
    return result
end

"""
    flip_spectrum!(P::PolesSum)

Reverse `P` and flip the sign of `locations(P)`.

See also [`flip_spectrum`](@ref).
"""
function flip_spectrum!(P::PolesSum)
    reverse!(P)
    l = locations(P)
    @. l *= -1
    return P
end

"""
    flip_spectrum(P::PolesSum)

Reverse `P` and flip the sign of `locations(P)`.

See also [`flip_spectrum!`](@ref).
"""
flip_spectrum(P::PolesSum) = flip_spectrum!(copy(P))

"""
    merge_degenerate_poles!(P::PolesSum, tol::Real=0)

Merge poles whose locations are `≤ tol` apart.
"""
function merge_degenerate_poles!(P::PolesSum, tol::Real=0)
    # check input
    tol >= 0 || throw(ArgumentError("tol must not be negative"))
    issorted(P) || throw(ArgumentError("P must be sorted"))
    # get information from P
    loc = locations(P)
    wgt = weights(P)
    # pole(s) at [-tol, tol]
    idx_zeros = findall(i -> abs(i) <= tol, loc)
    if !isempty(idx_zeros)
        i0 = popfirst!(idx_zeros)
        loc[i0] = 0
        for i in reverse!(idx_zeros)
            wgt[i0] += popat!(wgt, i)
            deleteat!(loc, i)
        end
    end
    # pole(s) at tol → ∞
    i = findfirst(>(0), loc)
    isnothing(i) && (i = lastindex(loc)) # enforce `i` to be a number
    while i < lastindex(loc)
        if loc[i + 1] - loc[i] <= tol
            # merge
            wgt[i] += popat!(wgt, i + 1)
            deleteat!(loc, i + 1) # keep location closer to zero
        else
            # increment index
            i += 1
        end
    end
    # pole(s) at -tol → -∞
    i = findlast(<(0), loc)
    isnothing(i) && (i = firstindex(loc)) # enforce `i` to be a number
    while i > firstindex(loc)
        if loc[i] - loc[i - 1] <= tol
            # merge
            wgt[i - 1] += popat!(wgt, i)
            deleteat!(loc, i - 1) # keep location closer to zero
            i -= 1
        else
            # decrement index
            i -= 1
        end
    end
    return P
end

"""
    merge_negative_locations_to_zero!(P::PolesSum)

Find all `locations(P) <= 0` and merge them.
"""
function merge_negative_locations_to_zero!(P::PolesSum)
    # check input
    issorted(P) || throw(ArgumentError("P must be sorted"))
    # get information from P
    loc = locations(P)
    wgt = weights(P)
    idx_zeros = findall(<=(0), loc)
    isempty(idx_zeros) && return P
    # add up all weights
    w0 = sum(wgt[idx_zeros])
    i0 = popfirst!(idx_zeros)
    loc[i0] = 0
    wgt[i0] = w0
    # delete degenerate locations
    for i in reverse!(idx_zeros)
        deleteat!(loc, i)
        deleteat!(wgt, i)
    end
    return P
end

"""
    merge_negative_weight!(P::PolesSum)

Move negative weights of `P` such that the zeroth moment is conserved
and the first moment changes minimally.
"""
function merge_negative_weight!(P::PolesSum)
    # check input
    issorted(P) || throw(ArgumentError("P is not sorted"))
    allunique(P) || throw(ArgumentError("P has degenerate locations"))
    moment(P, 0) >= 0 || throw(ArgumentError("total weight is negative"))

    loc = locations(P)
    wgt = weights(P)
    for i in eachindex(P)
        weights(P)[i] >= 0 && continue # no negative weight, go to next
        if i == length(P)
            # find previous positive weight
            for j in Iterators.reverse(1:(i - 1))
                iszero(wgt[j]) && continue
                if wgt[j] + wgt[end] >= 0
                    # wgt[j] can fully compensate wgt[end]
                    wgt[j] += wgt[end]
                    wgt[end] = 0
                    break
                else
                    # wgt[j] can't fully compensate wgt[end]
                    wgt[end] += wgt[j]
                    wgt[j] = 0
                end
            end
        else
            for j in Iterators.reverse(1:(i - 1))
                # find a previous pole with positive weight
                iszero(wgt[j]) && continue
                # calculate fractions how weight should be split
                f_left = (loc[i + 1] - loc[i]) / (loc[i + 1] - loc[j])
                f_right = 1 - f_left
                if wgt[j] + f_left * wgt[i] >= 0
                    # wgt[j] can fully compensate wgt[i]
                    wgt[j] += f_left * wgt[i]
                    wgt[i + 1] += f_right * wgt[i]
                    wgt[i] = 0
                else
                    # wgt[j] can't fully compensate wgt[i].
                    # Find fraction f ∈ (0, 1) which can be merged such that wgt[j] gets 0 weight.
                    # b_j + f f_l b_i === 0
                    wgt[i] += wgt[j] / f_left
                    wgt[i + 1] -= f_right / f_left * wgt[j]
                    wgt[j] = 0
                end
                if j == 1
                    # no pole with positive weight remaining
                    wgt[i + 1] += wgt[i]
                    wgt[i] = 0
                end
            end
            if wgt[i] <= 0
                # negative weight remaining and no previous weight to compensate
                # move negative weight to next pole
                wgt[i + 1] += wgt[i]
                wgt[i] = 0
            end
        end
    end
    moment(P, 0) >= 0 || throw(ArgumentError("total weight got negative"))
    return P
end

"""
    merge_small_poles!(P::PolesSum, tol::Real=1e-10)

Merge poles with weight `< tol` to its neighbors.

A given pole is split locally conserving the zeroth and first moment.
"""
function merge_small_poles!(P::PolesSum, tol::Real=1e-10)
    # check input
    tol >= 0 || throw(ArgumentError("tol must not be negative"))
    issorted(P) || throw(ArgumentError("P must be sorted"))
    # loop over all poles
    i = 1
    while i <= length(P)
        loc = locations(P)[i]
        wgt = weights(P)[i]
        if wgt >= tol
            # enough weight, go to next
            i += 1
            continue
        end
        if i == 1
            # add weight to next pole
            weights(P)[2] += weights(P)[1]
            deleteat!(locations(P), 1)
            deleteat!(weights(P), 1)
        elseif i == length(P)
            # add weight to previous pole
            weights(P)[end - 1] += wgt
            pop!(locations(P))
            pop!(weights(P))
        else
            # split weight such that zeroth and first moment is conserved
            loc_low = locations(P)[i - 1]
            loc_high = locations(P)[i + 1]
            weights(P)[i - 1] += (loc_high - loc) / (loc_high - loc_low) * wgt
            weights(P)[i + 1] += (loc - loc_low) / (loc_high - loc_low) * wgt
            deleteat!(locations(P), i)
            deleteat!(weights(P), i)
        end
    end
    return P
end

function moment(P::PolesSum, n::Int=0)
    foo = map(i -> i[1]^n * i[2], zip(locations(P), weights(P)))
    # sort by abs to guarantee that odd moments are zero for symmetric input
    sort!(foo; by=abs)
    return sum(foo)
end

"""
    remove_zero_weight!(P::PolesSum, remove_zero::Bool=true)

Remove all poles which have zero weight.

If `remove_zero`, the pole at ``a_i = 0`` with zero weight is also removed.

See also [`remove_zero_weight`](@ref).
"""
function remove_zero_weight!(P::PolesSum, remove_zero::Bool=true)
    i = 1
    while i <= length(P)
        if iszero(locations(P)[i]) && !remove_zero
            # keep pole at origin
            i += 1
            continue
        end

        if iszero(weights(P)[i])
            deleteat!(locations(P), i)
            deleteat!(weights(P), i)
        else
            i += 1
        end
    end
    return P
end

"""
    remove_zero_weight(P::PolesSum, remove_zero::Bool=true)

Remove all poles which have zero weight.

If `remove_zero`, the pole at ``a_i = 0`` with zero weight is also removed.

See also [`remove_zero_weight!`](@ref).
"""
function remove_zero_weight(P::PolesSum, remove_zero::Bool=true)
    return remove_zero_weight!(copy(P), remove_zero)
end

"""
    remove_small_poles!(P::PolesSum, tol::Real=1e-10, remove_zero::Bool=true)

Remove poles with weight `<= tol` and rescale remaining poles to conserve zeroth moment.

If `remove_zero`, ``a_i = 0`` with ``b_i ≤ tol`` is also removed.
"""
function remove_small_poles!(P::PolesSum, tol::Real=1e-10, remove_zero::Bool=true)
    # check input
    tol >= 0 || throw(ArgumentError("tol must not be negative"))
    issorted(P) || throw(ArgumentError("P must be sorted"))
    w_old = moment(P, 0) # old weight
    # loop over all poles
    i = 1
    while i <= length(P)
        if iszero(locations(P)[i]) && !remove_zero
            # keep pole at origin
            i += 1
        elseif weights(P)[i] <= tol
            deleteat!(locations(P), i)
            deleteat!(weights(P), i)
        else
            i += 1
        end
    end
    # rescale to conserve zeroth moment
    w_new = moment(P, 0) # new weight
    factor = w_old / w_new
    weights(P) .*= factor
    return P
end

"""
    spectral_function_loggaussian(P::PolesSum, ω, b::Real)

Calculate the spectral function ``A(ω) = -1/π \\mathrm{Im}[P(ω)]`` with a
lognormal broadening.

Each pole is broadened as in NRG

```math
b_i δ(ω - a_i) → b_i \\frac{\\mathrm{e}^{-b^2/4}}{\\sqrt{π}|a|b}
\\exp\\left(-\\frac{\\ln^2(ω/a_i)}{b^2}\\right).
```

If there is a pole ``a_i = 0``, it is shifted halfway betweeen its neighbors and
each getting half weight

```math
b_i δ(ω) →
  \\frac{b_i}{2} δ\\left(ω - \\frac{a_{i-1}}{2}\\right)
+ \\frac{b_i}{2} δ\\left(ω - \\frac{a_{i+1}}{2}\\right).
```
"""
function spectral_function_loggaussian(P::PolesSum, ω::Real, b::Real)
    result = zero(ω)
    iszero(ω) && return result # no weight at ω == 0
    for i in eachindex(P)
        if iszero(locations(P)[i])
            # special case, move half of weight to left/right repectively
            issorted(P) || throw(ArgumentError("P is not sorted"))
            loc = ω > 0 ? locations(P)[i + 1] / 2 : locations(P)[i - 1] / 2
            w = weights(P)[i] / 2
        elseif sign(ω) == sign(locations(P)[i])
            # only contribute weight if on same side of real axis
            w = weights(P)[i]
            loc = locations(P)[i]
        else
            # frequency has opposite sign compared to pole location
            continue
        end
        prefactor = w * exp(-b^2 / 4) / (b * abs(loc) * sqrt(π))
        result += prefactor * exp(-(log(ω / loc) / b)^2)
    end
    return result
end

function spectral_function_loggaussian(P::PolesSum, ω::Vector{<:Real}, b::Real)
    # map for each point in given grid
    return map(i -> spectral_function_loggaussian(P, i, b), ω)
end

"""
    to_grid(P::PolesSum, grid::Vector{<:Real})

Create a new [`PolesSum`](@ref) from `P` with locations given by `grid`.

A given pole is split locally conserving the zeroth and first moment.
If a pole is outside of `grid`, only the zeroth moment is conserved.
"""
function to_grid(P::PolesSum, grid::Vector{<:Real})
    # check input
    moment(P, 0) > 0 || throw(ArgumentError("P needs to have positive total weight"))
    issorted(grid) || throw(ArgumentError("grid is not sorted"))
    allunique(grid) || throw(ArgumentError("grid has degenerate locations"))

    # new location and weights
    weights_new = zero(grid)

    # run through each existing pole and split weight to new locations
    @inbounds for i in eachindex(P)
        loc = locations(P)[i]
        w = weights(P)[i]
        if loc <= first(grid)
            # no pole to the left
            weights_new[begin] += w
        elseif loc >= last(grid)
            # no pole to the right
            weights_new[end] += w
        else
            # find next pole with higher location
            i = searchsortedfirst(grid, loc)
            if loc - grid[i - 1] < 10 * eps()
                # previous pole has same location
                weights_new[i - 1] += w
            elseif grid[i] - loc < 10 * eps()
                # current pole has same location
                weights_new[i] += w
            else
                # split such that zeroth and first moment is conserved
                loc_low = grid[i - 1]
                loc_high = grid[i]
                weights_new[i - 1] += (loc_high - loc) / (loc_high - loc_low) * w
                weights_new[i] += (loc - loc_low) / (loc_high - loc_low) * w
            end
        end
    end
    return PolesSum(copy(grid), weights_new)
end

weight(P::PolesSum, i::Integer) = weights(P)[i]

weights(P::PolesSum) = P.wgt

function Core.Array(P::PolesSum)
    T = eltype(P)
    result = Matrix{T}(Diagonal([0; locations(P)]))
    result[1, 2:end] .= amplitudes(P)
    result[2:end, 1] .= amplitudes(P)
    return result
end

function Base.:+(A::PolesSum, B::PolesSum)
    result = PolesSum([locations(A); locations(B)], [weights(A); weights(B)])
    sort!(result)
    merge_degenerate_poles!(result, 0)
    return result
end

function Base.:-(A::PolesSum, B::PolesSum)
    result = PolesSum([locations(A); locations(B)], [weights(A); -weights(B)])
    sort!(result)
    merge_degenerate_poles!(result, 0)
    return result
end

function Base.copy(P::PolesSum)
    return PolesSum(copy(locations(P)), copy(weights(P)))
end

Base.eltype(::Type{<:PolesSum{A,B}}) where {A,B} = promote_type(A, B)

function Base.reverse!(P::PolesSum)
    reverse!(locations(P))
    reverse!(weights(P))
    return P
end

# create a better show?
Base.show(io::IO, P::PolesSum) = print(io, length(P), "-element ", summary(P))

function Base.sort!(P::PolesSum)
    p = sortperm(locations(P))
    P.loc[:] = P.loc[p]
    P.wgt[:] = P.wgt[p]
    return P
end
