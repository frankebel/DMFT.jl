"""
    solve_impurity(
        Δ::Pole{V,V},
        H_int::Operator,
        ϵ_imp::Real,
        n_v_bit::Int,
        n_c_bit::Int,
        e::Int,
        n_kryl_gs::Int,
        n_kryl::Int,
        O::AbstractVector{<:Operator},
    ) where {V<:AbstractVector{<:Real}}

Solve the Anderson impurity problem.

`O` contains operators which should act on the ground state ``|ψ_0⟩``.
The block Lanczos algorithm is then used to obtain the spectrum.
"""
function solve_impurity(
    Δ::Pole{V,V},
    H_int::Operator,
    ϵ_imp::Real,
    n_v_bit::Int,
    n_c_bit::Int,
    e::Int,
    n_kryl_gs::Int,
    n_kryl::Int,
    O::AbstractVector{<:Operator},
) where {V<:AbstractVector{<:Real}}
    # initialize system
    H, E0, ψ0 = init_system(Δ, H_int, ϵ_imp, n_v_bit, n_c_bit, e, n_kryl_gs)

    # same thread
    G_plus = _pos(H, E0, ψ0, O, n_kryl)
    G_minus = _neg(H, E0, ψ0, O, n_kryl)

    # Hartree self-energy
    # Σ_H = ⟨{[d_α, H_int], d_α^†}⟩
    op = O[2]' * O[1] + O[1] * O[2]'
    Σ_H = real(dot(ψ0, op * ψ0))

    return G_plus, G_minus, Σ_H
end

# positive frequencies, scalar
function _pos(
    H::CIOperator, E0::Real, ψ0::CI, O::Operator, n_kryl::Int
) where {CI<:CIWavefunction}
    v = O * ψ0
    b0 = norm(v)
    rmul!(v, inv(b0))
    a, b = lanczos(H, v, n_kryl)
    return _pole(a, b, E0, b0)
end

# positive frequencies, block
function _pos(
    H::CIOperator, E0::Real, ψ0::CI, O::AbstractVector{<:Operator}, n_kryl::Int
) where {CI<:CIWavefunction}
    V = Matrix{CI}(undef, 1, length(O)) # 1×n matrix
    for i in eachindex(V)
        V[i] = O[i] * ψ0
    end
    W, S_sqrt = orthogonalize_states(V)
    A, B = block_lanczos(H, W, n_kryl)
    return _pole(A, B, E0, S_sqrt)
end

# negative frequencies, scalar
function _neg(
    H::CIOperator, E0::Real, ψ0::CI, O::Operator, n_kryl::Int
) where {CI<:CIWavefunction}
    v = O * ψ0
    b0 = norm(v)
    rmul!(v, inv(b0))
    a, b = lanczos(H, v, n_kryl)
    return _pole(-a, -b, -E0, b0)
end

# negative frequencies
function _neg(
    H::CIOperator, E0::Real, ψ0::CI, O::AbstractVector{<:Operator}, n_kryl::Int
) where {CI<:CIWavefunction}
    V = Matrix{CI}(undef, 1, length(O)) # 1×n matrix
    for i in eachindex(V)
        V[i] = O[i]' * ψ0
    end
    W, S_sqrt = orthogonalize_states(V)
    A, B = block_lanczos(H, W, n_kryl)
    return _pole(-A, B, -E0, S_sqrt)
end

# diagonalize tridiagonal matrix given by `a`, `b` and convert to `Pole`
function _pole(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}, E0::Real, b0::Real)
    S = SymTridiagonal(a, b)
    E, T = eigen(S)
    E .-= E0
    R = b0 * T[1, :]
    map!(abs, R, R) # sign does not matter, positive is easier
    return Pole(E, R)
end

# diagonalize blocktridiagonal matrix given by `A`, `B` and convert to `Pole`
function _pole(
    A::AbstractVector{<:AbstractMatrix{<:Real}},
    B::AbstractVector{<:AbstractMatrix{<:T}},
    E0::Real,
    S_sqrt::AbstractMatrix{<:T},
) where {T<:Number}
    n1 = length(A)
    n2 = size(S_sqrt, 1)
    n = n1 * n2
    X = zeros(T, n, n)
    for i in 1:(n1 - 1)
        i1 = 1 + (i - 1) * n2
        i2 = i * n2
        X[i1:i2, (i1 + n2):(i2 + n2)] = B[i] # upper diagonal, don't need adjoint
        X[i1:i2, i1:i2] = A[i] # main diagonal
        X[(i1 + n2):(i2 + n2), i1:i2] = B[i] # lower diagonal
    end
    X[(end - n2 + 1):end, (end - n2 + 1):end] = A[end] # last element
    E, _ = LAPACK.syev!('V', 'U', X)
    E .-= E0
    R = S_sqrt * X[1:size(S_sqrt, 1), :]
    return Pole(E, R)
end
