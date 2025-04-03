"""
    block_lanczos(
        H::CIOperator, W::AbstractMatrix{<:CWF}, n_kryl::Int
    ) where {CWF<:CIWavefunction}


Block Lanczos algorithm for given operator `H`, states in `W` and `n_kryl` Krylov cycles.

Returns main diagonal `A` and subdiagonal `B`.
"""
function block_lanczos(
    H::CIOperator, W::AbstractMatrix{<:CWF}, n_kryl::Int
) where {CWF<:CIWavefunction}
    A = Vector{Matrix{Float64}}(undef, n_kryl)
    B = Vector{Matrix{Float64}}(undef, n_kryl - 1)
    for i in eachindex(A)
        @inbounds A[i] = Matrix{Float64}(undef, 2, 2)
    end
    for i in eachindex(B)
        @inbounds B[i] = Matrix{Float64}(undef, 2, 2)
    end
    V = deepcopy(W)
    V_new = zero(W)
    V_old = zero(W)
    # dummy containers to reduce allocations
    SVD = zero(W)
    M1 = Matrix{Float64}(undef, length(W), length(W))
    M2 = similar(M1)
    M3 = similar(M1)
    Adj = Vector{CWF}(undef, length(W))

    # first step
    mul!(V_new, H, V)
    mul!(A[1], V', V_new)
    copyto!(M1, A[1])
    rmul!(M1, -1)
    mul!(V_new, V, M1, true, true)

    # successive steps
    for j in 2:n_kryl
        # orthonormalize V_new
        zerovector!(SVD)
        _svd_orthogonalize!(B[j - 1], V_new, SVD, Adj, M1, M2, M3)
        V_new, SVD = SVD, V_new
        # cycle for new step
        V, V_old, V_new = V_new, V, V_old
        zerovector!(V_new)
        mul!(V_new, H, V)
        adjoint!(Adj, V)
        mul!(A[j], Adj, V_new) # A = V' V_new
        copyto!(M1, A[j])
        rmul!(M1, -1)
        mul!(V_new, V, M1, true, true) # V_new -= V*A
        adjoint!(M1, B[j - 1])
        rmul!(M1, -1)
        mul!(V_new, V_old, M1, true, true) # V_new -= V*B'
    end
    # hermitize matrices
    map(hermitianpart!, A)
    map(hermitianpart!, B)
    return A, B
end

# Explanation available under ch. 3.2 of Martin's thesis.
# https://archiv.ub.uni-heidelberg.de/volltextserver/29305/2/Thesis_V5.pdf
function _svd_orthogonalize!(
    B::AbstractMatrix{<:T},
    ψ::AbstractMatrix{<:C},
    SVD::AbstractMatrix{<:C},
    Adj::AbstractVector{<:C},
    M1::AbstractMatrix{<:T},
    M2::AbstractMatrix{<:T},
    M3::AbstractMatrix{<:T},
) where {C<:CIWavefunction,T<:Number}
    tol = 1E-6 # potential issue with small/negative eigenvalues
    n = size(M2, 1)
    adjoint!(Adj, ψ)
    mul!(M1, Adj, ψ) # overlap matrix
    S, _ = LAPACK.syev!('V', 'U', M1) # How to avoid allocations in diagonalization?
    # orthonormalized states in SVD
    rmul!(M2, false) # S_sqrt_inv
    for i in 1:n
        M2[i, i] = S[i] > tol ? 1 / sqrt(S[i]) : zero(T)
    end
    adjoint!(B, M1) # V'
    mul!(M3, M2, B) # S_sqrt_inv V'
    mul!(M2, M1, M3) # V S_sqrt_inv V'
    mul!(SVD, ψ, M2) # ψ V S_sqrt_inv V'
    # B matrix
    rmul!(M2, false) # M2 = S_sqrt
    for i in 1:n
        M2[i, i] = S[i] > tol ? sqrt(S[i]) : zero(T)
    end
    adjoint!(B, M1) # V'
    mul!(M3, M2, B) # S_sqrt V'
    mul!(B, M1, M3) # V S_sqrt V'
    hermitianpart!(B) # potential numerical inexactness
    return nothing
end
