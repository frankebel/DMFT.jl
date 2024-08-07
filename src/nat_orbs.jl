"""
    to_natural_orbitals(H::AbstractMatrix, ϵ::Real=1E-8)

Transforms a single particle Hamiltonian `H` to natural orbital basis.

`H[1,1]` is the onsite energy of the impurity.
States with energies `E ∈ (-ϵ, ϵ)` are considered degenerate.
"""
function to_natural_orbitals(H::AbstractMatrix{<:Real}, ϵ::Real=1E-8)
    ishermitian(H) || throw(ArgumentError("`H` not hermitian"))
    E, T = LAPACK.syev!('V', 'U', copy(H))
    n_lower = count(x -> x <= -ϵ, E)
    n_zero = count(x -> abs(x) < ϵ, E)
    isodd(n_zero) && @warn "odd number of energies equal zero"
    n_occ = n_lower + (n_zero ÷ 2) # half of states around zero occupied
    if n_zero > 1
        # symmetrize degenerate states around zero
        @info "degenerate zero-energy"
        p = T'[n_occ:(n_occ + 1), 1]
        p ./= norm(p)
        R = inv([p[1] p[2]; p[2] -p[1]]) * [1 / sqrt(2), 1 / sqrt(2)]
        t1 = R[1] * T[:, n_occ] + R[2] * T[:, n_occ + 1]
        t2 = R[1] * T[:, n_occ + 1] - R[2] * T[:, n_occ]
        T[:, n_occ] .= t1[:]
        T[:, n_occ + 1] .= t2[:]
    end

    base_occ = T[:, 1:n_occ] # valence states
    base_emp = T[:, (n_occ + 1):end] # conduction states

    P = base_occ * base_occ' # projector on valence
    Q = I - P # projector on conduction
    w = P[:, 1] # impurity projected on valence
    α = norm(w)
    w ./= α
    u = Q[:, 1] # impurity projected on conduction
    β = norm(u)
    u ./= β

    # orthogonalize the remaining valence wrt impurity
    for v in eachcol(base_occ)
        v .-= dot(v, w) .* w
    end
    # orthogonalize the remaining conduction wrt impurity
    for v in eachcol(base_emp)
        v .-= dot(v, u) .* u
    end

    # Löwdin on valence states
    base_occ[:, 1] .= w
    B = @view base_occ[:, 2:end]
    S = B' * B
    E, V = LAPACK.syev!('V', 'U', S)
    R = V * Diagonal(map(x -> 1 / sqrt(x), E)) * V'
    B .= B * R
    h = base_occ' * H * base_occ

    a_occ, b_occ, _ = sytrd!('L', h)

    # Löwdin on conduction states
    base_emp[:, 1] .= u
    B = @view base_emp[:, 2:end]
    S = B' * B
    E, V = LAPACK.syev!('V', 'U', S)
    R = V * Diagonal(map(x -> 1 / sqrt(x), E)) * V'
    B .= B * R
    h = base_emp' * H * base_emp

    a_emp, b_emp, _ = sytrd!('L', h)

    push!(b_occ, 0.0)
    a = vcat(a_occ, a_emp)
    b = vcat(b_occ, b_emp)
    H_tri = SymTridiagonal(a, b)
    n = length(a)
    v1 = zeros(n)
    v1[1] = α
    v1[n_occ + 1] = β
    v2 = zeros(n)
    v2[1] = β
    v2[n_occ + 1] = -α
    T = Matrix(Diagonal(ones(n)))
    T[:, 1] = v1
    T[:, n_occ + 1] = v2
    H_trafo = T' * H_tri * T
    H_trafo = 0.5 * (H_trafo' + H_trafo) # hermitize
    return H_trafo, n_occ
end
