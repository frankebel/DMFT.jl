"""
    block_lanczos(
        H::CIOperator, W::AbstractMatrix{<:CWF}, n_kryl::Int
    ) where {CWF<:CIWavefunction}


Block Lanczos algorithm for given operator `H`, states in `W` and `n_kryl` Krylov cycles.

Returns main diagonal `A` and subdiagonal `B`.
"""
function block_lanczos(
    H::CIOperator, Q1::AbstractMatrix{<:CWF}, N::Integer
) where {CWF<:CIWavefunction}
    foo, q = size(Q1)
    isone(foo) || throw(ArgumentError("input matrix must have 1 row"))
    A = Vector{Matrix{Float64}}(undef, N)
    B = Vector{Matrix{Float64}}(undef, N - 1)
    N >= 1 || throw(ArgumentError("N must be >= 1"))
    T = scalartype(CWF)
    Q_curr = deepcopy(Q1)
    Q_new = similar(Q1)
    Q_old = similar(Q1)
    # containers to reduce allocations
    Q_int = zero(Q1) # int ≙ intermediate
    V1 = Vector{T}(undef, q)
    M1 = Matrix{T}(undef, q, q)

    # first step
    mul!(Q_new, H, Q_curr)
    @inbounds A[1] = Matrix{T}(undef, q, q)
    mul!(A[1], Q_curr', Q_new)
    hermitianpart!(A[1]) # enforce due to finite precision
    copyto!(M1, A[1])
    rmul!(M1, -1)
    mul!(Q_new, Q_curr, M1, true, true)

    # successive steps
    for j in 2:N
        # orthonormalize Q_new
        zerovector!(Q_int)
        @inline B[j - 1] = Matrix{T}(undef, q, q)
        _orthonormalize_SVD!(V1, M1, B[j - 1], Q_int, Q_new)
        # TODO: Stop early if norm is small.
        Q_new, Q_int = Q_int, Q_new
        # cycle for new step
        Q_curr, Q_old, Q_new = Q_new, Q_curr, Q_old
        # Q_new = H Q
        zerovector!(Q_new)
        mul!(Q_new, H, Q_curr)
        # Q_new -= Q_old B^†_{j-1}
        adjoint!(M1, B[j - 1])
        rmul!(M1, -1)
        mul!(Q_new, Q_old, M1, true, true)
        # A_j = Q^† Q_new
        @inline A[j] = Matrix{T}(undef, q, q)
        mul!(A[j], Q_curr', Q_new)
        hermitianpart!(A[j]) # enforce due to finite precision
        # Q_new -= Q A_j
        copyto!(M1, A[j])
        rmul!(M1, -1)
        mul!(Q_new, Q_curr, M1, true, true)
    end
    return A, B
end
