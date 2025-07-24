"""
    correlator(
        H::CIOperator, E0::Real, ψ0::CI, O::Operator, n_kryl::Int
    ) where {CI<:CIWavefunction}

Calculate the correlator

```math
C(ω) = \\left⟨ ψ_0 O^† \\frac{1}{ω - H} O ψ_0 \\right⟩.
```
"""
function correlator(
    H::CIOperator, ψ0::CI, O::Operator, n_kryl::Int
) where {CI<:CIWavefunction}
    v = O * ψ0
    scale = norm(v)
    rmul!(v, inv(scale))
    locations, amplitudes = lanczos(H, v, n_kryl)

    # NOTE: break Lanczos early if amplitude is small
    value, index = findmin(amplitudes)
    @debug "smallest amplitude b=$(value) at index $(index)/$(lastindex(amplitudes))"

    # create block tridiagonal pole representation which is then diagonalized
    PCF = PolesContinuedFraction(locations, amplitudes, scale)
    return PolesSum(PCF)
end

"""
    correlator(
        H::CIOperator, ψ0::CI, O::AbstractVector{<:Operator}, n_kryl::Int
    ) where {CI<:CIWavefunction}

Calculate the block correlator

```math
C(ω) = \\left⟨ ψ_0 O^† \\frac{1}{ω - H} O ψ_0 \\right⟩.
```
"""
function correlator(
    H::CIOperator, ψ0::CI, O::AbstractVector{<:Operator}, n_kryl::Int
) where {CI<:CIWavefunction}
    V = Matrix{CI}(undef, 1, length(O)) # 1×n matrix
    for i in eachindex(V)
        V[i] = O[i] * ψ0
    end
    W, scale = _orthonormalize_SVD(V)
    locations, amplitudes = block_lanczos(H, W, n_kryl)

    # create block tridiagonal pole representation which is then diagonalized
    Pt = PolesContinuedFractionBlock(locations, amplitudes, scale)
    return PolesSumBlock(Pt)
end

"""
    correlator_plus(
        H::CIOperator, E0::Real, ψ0::CI, O::Operator, n_kryl::Int
    ) where {CI<:CIWavefunction}

Calculate the positive spectrum of the correlator.

```math
C^+(ω) = \\left⟨ ψ_0 O^† \\frac{1}{ω - H + E_0} O ψ_0 \\right⟩
```

See also [`correlator_minus`](@ref).
"""
function correlator_plus(
    H::CIOperator, E0::Real, ψ0::CI, O::Operator, n_kryl::Int
) where {CI<:CIWavefunction}
    C = correlator(H, ψ0, O, n_kryl)
    locations(C) .-= E0

    # poles at negative locatiions (never happens on exact arithmetic)
    idx_neg = findall(<(0), locations(C))
    if !isempty(idx_neg)
        n_neg = length(idx_neg)
        weight_neg = sum(weights(C)[idx_neg])
        @warn "C+ has negative specral weight $(weight_neg) on $(n_neg) pole(s)"
    end

    return C
end

"""
    correlator_plus(
        H::CIOperator, E0::Real, ψ0::CI, O::AbstractVector{<:Operator}, n_kryl::Int
    ) where {CI<:CIWavefunction}

Calculate the positive spectrum of the block correlator.

```math
C^+(ω) = \\left⟨ ψ_0 O^† \\frac{1}{ω - H + E_0} O ψ_0 \\right⟩
```

See also [`correlator_minus`](@ref).
"""
function correlator_plus(
    H::CIOperator, E0::Real, ψ0::CI, O::AbstractVector{<:Operator}, n_kryl::Int
) where {CI<:CIWavefunction}
    C = correlator(H, ψ0, O, n_kryl)
    locations(C) .-= E0

    # poles at negative energies (never happens on exact arithmetic)
    idx_neg = findall(<(0), locations(C))
    if !isempty(idx_neg)
        n_neg = length(idx_neg)
        @warn "C+ has $(n_neg) negative pole location(s)"
    end

    return C
end

"""
    correlator_minus(
        H::CIOperator, E0::Real, ψ0::CI, O::Operator, n_kryl::Int
    ) where {CI<:CIWavefunction}

Calculate the negative spectrum of the correlator.

```math
C^-(ω) = \\left⟨ ψ_0 O^† \\frac{1}{ω + H - E_0} O ψ_0 \\right⟩
```

See also [`correlator_plus`](@ref).
"""
function correlator_minus(
    H::CIOperator, E0::Real, ψ0::CI, O::Operator, n_kryl::Int
) where {CI<:CIWavefunction}
    C = correlator(H, ψ0, O, n_kryl)
    locations(C) .-= E0

    map!(-, locations(C), locations(C)) # flip sign of eigenvalues
    reverse!(C) # order form lowest to highest

    # poles at positive energies (never happens on exact arithmetic)
    idx_pos = findall(>(0), locations(C))
    if !isempty(idx_pos)
        n_pos = length(idx_pos)
        weight_pos = sum(weights(C)[idx_pos])
        @warn "C- has positive specral weight $(weight_pos) on $(n_pos) pole(s)"
    end

    return C
end

"""
    correlator_minus(
        H::CIOperator, E0::Real, ψ0::CI, O::AbstractVector{<:Operator}, n_kryl::Int
    ) where {CI<:CIWavefunction}

Calculate the negative spectrum of the block correlator.

```math
C^+(ω) = \\left⟨ ψ_0 O^† \\frac{1}{ω + H - E_0} O ψ_0 \\right⟩
```

See also [`correlator_minus`](@ref).
"""
function correlator_minus(
    H::CIOperator, E0::Real, ψ0::CI, O::AbstractVector{<:Operator}, n_kryl::Int
) where {CI<:CIWavefunction}
    C = correlator(H, ψ0, O, n_kryl)
    locations(C) .-= E0

    map!(-, locations(C), locations(C)) # flip sign of eigenvalues
    reverse!(C) # order form lowest to highest

    # poles at positive energies (never happens on exact arithmetic)
    idx_pos = findall(>(0), locations(C))
    if !isempty(idx_pos)
        n_pos = length(idx_pos)
        @warn "C- has $(n_pos) positive pole location(s)"
    end

    return C
end
