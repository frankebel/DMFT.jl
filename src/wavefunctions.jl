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
    ground_state(H::Operator, ψ_start::Wavefunction, n_kryl::Int)
    ground_state(H::CIOperator, ψ_start::CIWavefunction, n_kryl::Int)

Get approximate ground state and energy using `n_kryl` Krylov cycles.
"""
function ground_state(H::Operator, ψ_start::Wavefunction, n_kryl::Int)
    ψ0, E0 = Fermions.ground_state(H, ψ_start; n_kryl=n_kryl, rtol=5e-8, verbose=false)
    return E0, ψ0
end

function ground_state(H::CIOperator, ψ_start::CIWavefunction, n_kryl::Int)
    α, β, states = lanczos_krylov(H, ψ_start, n_kryl)
    E, T = LAPACK.stev!('V', α, β)
    E0 = E[1]
    ψ0 = zero(ψ_start)
    @inbounds for i in eachindex(E)
        axpy!(T[i, 1], states[i], ψ0)
    end
    normalize!(ψ0) # possible orthogonality loss in Lanczos
    Hψ = H * ψ0
    H_avg = dot(ψ0, Hψ)
    H_sqr = dot(Hψ, Hψ)
    var_rel = H_sqr / H_avg^2 - 1
    # compare eigenvalue with expectation value
    rdiff = H_avg / E0 - 1
    abs(rdiff) < 1e-14 || @warn "discrepancy eigenvalue to eigenstate"
    @info "ground state" E0 var_rel length(ψ0)
    return E0, ψ0
end
