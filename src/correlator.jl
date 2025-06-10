"""
    correlator(
        H::CIOperator, E0::Real, ψ0::CI, O::Operator, n_kryl::Int; minus::Bool=true
    ) where {CI<:CIWavefunction}

If `minus` calculate the correlator

```math
C(ω) = \\left⟨ ψ_0 O^† \\frac{1}{ω - H} O ψ_0 \\right⟩,
```

otherwise

```math
C(ω) = \\left⟨ ψ_0 O^† \\frac{1}{ω + H} O ψ_0 \\right⟩.
```
"""
function correlator(
    H::CIOperator, ψ0::CI, O::Operator, n_kryl::Int; minus::Bool=true
) where {CI<:CIWavefunction}
    v = O * ψ0
    b0 = norm(v)
    rmul!(v, inv(b0))
    a, b = lanczos(H, v, n_kryl)

    # look if any coefficient in `b` is small
    value, index = findmin(b)
    @debug "smallest weight b=$(value) at index $(index)/$(lastindex(b))"

    # diagonalize tridiagonal matrix given by `a`, `b`
    S = SymTridiagonal(a, b)
    E, T = eigen(S)
    R = b0 * T[1, :]
    @. R = abs(R) # sign does not matter, positive is easier

    if !minus
        @. E = -E
        # reverse to list locations from smallest to largest
        reverse!(E)
        reverse!(R)
    end

    return Poles(E, R)
end

"""
    correlator(
        H::CIOperator, ψ0::CI, O::AbstractVector{<:Operator}, n_kryl::Int; minus::Bool=true
    ) where {CI<:CIWavefunction}

If `minus` calculate the block correlator

```math
C(ω) = \\left⟨ ψ_0 O^† \\frac{1}{ω - H} O ψ_0 \\right⟩,
```

otherwise

```math
C(ω) = \\left⟨ ψ_0 O^† \\frac{1}{ω + H} O ψ_0 \\right⟩.
```
"""
function correlator(
    H::CIOperator, ψ0::CI, O::AbstractVector{<:Operator}, n_kryl::Int; minus::Bool=true
) where {CI<:CIWavefunction}
    V = Matrix{CI}(undef, 1, length(O)) # 1×n matrix
    for i in eachindex(V)
        V[i] = O[i] * ψ0
    end
    W, S_sqrt = orthogonalize_states(V)
    A, B = block_lanczos(H, W, n_kryl)

    # diagonalize blocktridiagonal matrix given by `A`, `B` and convert to `Poles`
    n1 = length(A)
    n2 = length(O)
    n = n1 * n2
    X = zeros(eltype(eltype(A)), n, n)
    for i in 1:(n1 - 1)
        i1 = 1 + (i - 1) * n2
        i2 = i * n2
        X[i1:i2, (i1 + n2):(i2 + n2)] = B[i] # upper diagonal, don't need adjoint
        X[i1:i2, i1:i2] = A[i] # main diagonal
        X[(i1 + n2):(i2 + n2), i1:i2] = B[i] # lower diagonal
    end
    X[(end - n2 + 1):end, (end - n2 + 1):end] = A[end] # last element
    E, _ = LAPACK.syev!('V', 'U', X)
    R = S_sqrt * X[1:size(S_sqrt, 1), :]

    if !minus
        @. E = -E
        # reverse to list locations from smallest to largest
        reverse!(E)
        reverse!(R; dims=2)
    end

    return Poles(E, R)
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
    shift_spectrum!(C, E0)

    # poles at negative energies (never happens on exact arithmetic)
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
    shift_spectrum!(C, E0)

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
    C = correlator(H, ψ0, O, n_kryl; minus=false)

    shift_spectrum!(C, -E0)

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
    C = correlator(H, ψ0, O, n_kryl; minus=false)

    shift_spectrum!(C, -E0)

    # poles at positive energies (never happens on exact arithmetic)
    idx_pos = findall(>(0), locations(C))
    if !isempty(idx_pos)
        n_pos = length(idx_pos)
        @warn "C- has $(n_pos) positive pole location(s)"
    end

    return C
end
