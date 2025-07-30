# Methods related to `Fermions.Wavefunctions` module.

"""
    Wavefunction_singlet(
        D::Type, L_v::Integer, L_c::Integer, V_v::Integer, V_c::Integer
    )

Initialize a `Wavefunction` with
singlet in impurity and mirror site
and all valence sites filled.
"""
function Wavefunction_singlet(
    D::Type, L_v::Integer, L_c::Integer, V_v::Integer, V_c::Integer
)
    K, V = keytype(D), valtype(D)
    s1 = slater_start(K, 0b0110, L_v, L_c, V_v, V_c)
    s2 = slater_start(K, 0b1001, L_v, L_c, V_v, V_c)
    return Wavefunction(Dict(s1 => V(1 / sqrt(2)), s2 => V(1 / sqrt(2))))
end

"""
    CIWavefunction_singlet(
        D::Type, L_v::Integer, L_c::Integer, V_v::Integer, V_c::Integer, excitation::Integer
    )

Initialize a `CIWavefunction` with
singlet in impurity and mirror site
and all valence sites filled.
"""
function CIWavefunction_singlet(
    D::Type, L_v::Integer, L_c::Integer, V_v::Integer, V_c::Integer, excitation::Integer
)
    K, V = keytype(D), valtype(D)
    s1 = slater_start(K, 0b0110, L_v, L_c, 0, 0)
    s2 = slater_start(K, 0b1001, L_v, L_c, 0, 0)
    v_dim = sum(i -> binomial(2 * (V_v + V_c), i), 0:excitation)
    vec = zeros(V, v_dim)
    vec[1] = 1 / sqrt(2)
    result = CIWavefunction(
        Dict(s1 => copy(vec), s2 => copy(vec)), 2 + L_v + L_c, V_v, V_c, excitation
    )
    return result
end

"""
    ground_state!(
        H::CIOperator,
        ψ_start::CIWavefunction,
        n_kryl::Integer,
        n_max_restart::Integer,
        variance::Real,
    )

Shift ``H → H - E_0`` in-place and return ``E_0``, ``|ψ_0⟩``.

Get approximate ground state and energy using steps of `n_kryl` Krylov cycles
and at most `n_max_restart` restarts.
Calculation is stopped early if `⟨H^2⟩ <= variance`.
"""
function ground_state!(
    H::CIOperator,
    ψ_start::CIWavefunction,
    n_kryl::Integer,
    n_max_restart::Integer,
    variance::Real,
)
    # check input
    isapprox(norm(ψ_start), 1; atol=10 * eps()) ||
        throw(ArgumentError("ψ_start is not normalized"))
    n_kryl >= 1 || throw(ArgumentError("n_kryl must be at least 1"))
    n_max_restart >= 1 || throw(ArgumentError("n_max_restart must be at least 1"))
    variance >= 0 || throw(ArgumentError("variance must be >= 0"))

    # initial guess
    ψ0 = deepcopy(ψ_start)
    E0 = dot(ψ0, H, ψ0)
    shift_spectrum!(H, E0)

    for _ in 1:n_max_restart
        α, β, states = lanczos_krylov(H, ψ0, n_kryl)
        F = eigen!(SymTridiagonal(α, β))
        # new state is linear combination
        ψ0_new = zero(ψ_start)
        @inbounds for i in 1:n_kryl
            axpy!(F.vectors[i, 1], states[i], ψ0_new) # ψ0_new += c_i * ψ_i
        end
        normalize!(ψ0_new) # possible orthogonality loss in Lanczos
        ψ0 = ψ0_new
        E0 = dot(ψ0, H, ψ0)
        shift_spectrum!(H, E0)

        # calculate variance
        foo = H * ψ0
        foo ⋅ foo < variance && break # variance is below input
    end

    # find constant term for E0
    for t in H.opbit.terms
        if iszero(t.mask) &&
            iszero(t.left) &&
            iszero(t.right) &&
            iszero(t.change) &&
            iszero(t.signmask)
            E0 = -t.value
            break
        end
    end

    return E0, ψ0
end
