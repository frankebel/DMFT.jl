"""
    solve_impurity(
        Δ::Greensfunction{<:Real,<:AbstractVector{<:Real}},
        H_int::Operator,
        ϵ_imp::T,
        n_v_bit::Int,
        n_c_bit::Int,
        e::Int,
        n_kryl_gs::Int,
        n_kryl::Int,
        O::AbstractVector{<:Operator},
    ) where {T<:Real}

Solve the Anderson impurity problem.

`O` contains operators which should act on the ground state ``|ψ_0⟩``.
The block Lanczos algorithm is then used to obtain the spectrum.
"""
function solve_impurity(
    Δ::Greensfunction{<:T,<:AbstractVector{<:T}},
    H_int::Operator,
    ϵ_imp::T,
    n_v_bit::Int,
    n_c_bit::Int,
    e::Int,
    n_kryl_gs::Int,
    n_kryl::Int,
    O::AbstractVector{<:Operator},
) where {T<:Real}
    # initialize system
    H, E0, ψ0 = init_system(Δ, H_int, ϵ_imp, n_v_bit, n_c_bit, e, n_kryl_gs)

    # different threads
    thr1 = Threads.@spawn _pos(
        deepcopy(H), copy(E0), deepcopy(ψ0), deepcopy(O), copy(n_kryl)
    )
    thr2 = Threads.@spawn _neg(
        deepcopy(H), copy(E0), deepcopy(ψ0), deepcopy(O), copy(n_kryl)
    )
    G_plus::Greensfunction{T,Matrix{T}} = fetch(thr1)
    G_minus::Greensfunction{T,Matrix{T}} = fetch(thr2)

    # # same thread
    # G_plus = _pos(H, E0, ψ0, O, n_kryl)
    # G_minus = _neg(H, E0, ψ0, O, n_kryl)

    # Hartree self-energy
    # Σ_H = ⟨{[d_α, H_int], d_α^†}⟩
    op = O[2]' * O[1] + O[1] * O[2]'
    Σ_H = real(dot(ψ0, op * ψ0))

    return G_plus, G_minus, Σ_H
end

# positive frequencies
function _pos(
    H::CIOperator, E0::Real, ψ0::CI, O::AbstractVector{<:Operator}, n_kryl::Int
) where {CI<:CIWavefunction}
    V = Matrix{CI}(undef, 1, length(O)) # 1×n matrix
    for i in eachindex(V)
        V[i] = O[i] * ψ0
    end
    W, S_sqrt = orthogonalize_states(V)
    A, B = block_lanczos(H, W, n_kryl)
    return _greensfunction(A, B, E0, S_sqrt)
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
    return _greensfunction(-A, B, -E0, S_sqrt)
end

# Diagonalize blocktridiagonal matrix given by `A`, `B` and convert to `Greensfunction`.
function _greensfunction(
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
    return Greensfunction(E, R)
end
