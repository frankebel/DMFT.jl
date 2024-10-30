"""
    block_lanczos(H::CIOperator, W::AbstractMatrix{<:CIWavefunction}, n_kryl::Int)

Block Lanczos algorithm for given operator `H`, states in `W` and `n_kryl` Krylov cycles.

Returns main diagonal `A` and subdiagonal `B`.
"""
function block_lanczos(H::CIOperator, W::AbstractMatrix{<:CIWavefunction}, n_kryl::Int)
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
    SVD = zero(W) # dummy container for SVD orthonormalization

    # first step
    mul!(V_new, H, V)
    mul!(A[1], V', V_new)
    mul!(V_new, V, -A[1])

    # successive steps
    for j in 2:n_kryl
        zerovector!(SVD)
        _svd_orthogonalize!(V_new, SVD, B[j - 1])
        V_new, SVD = SVD, V_new
        foo = similar(A[1])
        mul!(foo, V_new', V_new)
        # Cycle 3 vectors.
        V, V_old, V_new = V_new, V, V_old
        zerovector!(V_new)
        mul!(V_new, H, V)
        mul!(A[j], V', V_new)
        mul!(V_new, V, -A[j])
        mul!(V_new, V_old, -B[j - 1]')
    end
    return A, B
end

# Explanation available under ch. 3.2 of Martin's thesis.
# https://archiv.ub.uni-heidelberg.de/volltextserver/29305/2/Thesis_V5.pdf
function _svd_orthogonalize!(
    ψ::AbstractMatrix{<:C}, SVD::AbstractMatrix{<:C}, B::AbstractMatrix{<:T}
) where {C<:CIWavefunction,T<:Number}
    # Possible issues with negative eigenvalues. Therefore use tolerance `tol`.
    tol = 1E-6
    S = similar(B)
    mul!(S, ψ', ψ) # write in dummy location
    D, V = LAPACK.syev!('V', 'U', S)
    # SVD = ψ*V*D_sqrt_inv*V^†
    # B = V*S_sqrt*V^†
    mul!(SVD, ψ, V * Diagonal(map(d -> d > tol ? 1 / sqrt(d) : zero(T), D)) * V')
    foo = V * Diagonal(map(d -> d > tol ? sqrt(d) : zero(T), D)) * V'
    B[:] = 0.5 * (foo + foo') # hermitize due to potential numerical inexactness
    return ψ, SVD, B
end
