# Methods for orthogonalization and orthonormalization of states.

"""
    _orthonormalize_SVD(V::AbstractMatrix)

Löwdin orthonormalization (Singular value decomposition SVD) for given states in `V`.

Calulate overlap matrix ``S = V^† V`` and diagonalize

```math
\\begin{aligned}
S        &= T Λ T^† \\\\
S^{1/2}  &= T Λ^{1/2} T^† \\\\
S^{-1/2} &= T Λ^{-1/2} T^†.
\\end{aligned}
```

Returns ``W = V S^{-1/2}`` and ``S^{1/2}``.
"""
function _orthonormalize_SVD(V::AbstractMatrix{<:T}) where {T<:Number}
    S = hermitianpart!(V' * V)
    F = eigen!(S)
    tol = sqrt(eps()) * maximum(F.values)
    S_sqrt =
        F.vectors * Diagonal(map(i -> i >= tol ? sqrt(i) : zero(i), F.values)) * F.vectors'
    S_sqrt_inv =
        F.vectors *
        Diagonal(map(i -> i >= tol ? 1 / sqrt(i) : zero(i), F.values)) *
        F.vectors'
    hermitianpart!(S_sqrt)
    hermitianpart!(S_sqrt_inv)
    W = V * S_sqrt_inv
    return W, S_sqrt
end
function _orthonormalize_SVD(V::AbstractMatrix{<:C}) where {C<:CIWavefunction}
    size(V, 1) == 1 || throw(ArgumentError("input matrix must have 1 row"))
    n = size(V, 2)
    T = scalartype(C)
    S_old = Matrix{T}(undef, n, n)
    mul!(S_old, V', V)
    S = T <: Real ? Symmetric(S_old) : Hermitian(S_old)
    F = eigen(S)
    tol = maximum(F.values) * sqrt(eps())
    S_sqrt =
        F.vectors * Diagonal(map(i -> i >= tol ? sqrt(i) : zero(i), F.values)) * F.vectors'
    S_sqrt_inv =
        F.vectors *
        Diagonal(map(i -> i >= tol ? 1 / sqrt(i) : zero(i), F.values)) *
        F.vectors'
    hermitianpart!(S_sqrt)
    hermitianpart!(S_sqrt_inv)
    W = zero(V)
    mul!(W, V, S_sqrt_inv)
    return W, S_sqrt
end

"""
    orthonormalize_GramSchmidt!(V::AbstractMatrix{<:Number})

Orthonormalize given states (columns) using Gram-Schmidt.
"""
function _orthonormalize_GramSchmidt!(V::AbstractMatrix{<:Number})
    tol = 1000 * eps()
    for i in axes(V, 2)
        v = view(V, :, i) # state
        if norm(v)^2 < tol
            v .= 0 # set state to zero
            continue
        end
        for j in 1:(i - 1)
            # orthogonalize against previous state
            vj = view(V, :, j)
            a = vj ⋅ v
            axpy!(-a, vj, v) # v -= a*vj
        end
        if norm(v)^2 < tol
            v .= 0 # set state to zero
            continue
        end
        normalize!(v)
    end
    return V
end
