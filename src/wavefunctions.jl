# Methods related to `Fermions.Wavefunctions` module.

"""
    starting_Wavefunction(
        D::Type, n_v_bit::Int, n_c_bit::Int, n_v_vector::Int, n_c_vector::Int
    )

Return 2-determinant-state `Wavefunction` with
opposite filling in impurity, mirror bath site.
"""
function starting_Wavefunction(
    D::Type, n_v_bit::Int, n_c_bit::Int, n_v_vector::Int, n_c_vector::Int
)
    K, V = keytype(D), valtype(D)
    s1 = slater_start(K, 0b0110, n_v_bit, n_c_bit, n_v_vector, n_c_vector)
    s2 = slater_start(K, 0b1001, n_v_bit, n_c_bit, n_v_vector, n_c_vector)
    result = Wavefunction(Dict(s1 => one(V), s2 => one(V)))
    normalize!(result)
    return result
end

"""
    starting_CIWavefunction(
        D::Type,
        n_v_bit::Int,
        n_c_bit::Int,
        n_v_vector::Int,
        n_c_vector::Int,
        excitation::Int,
    )

Return 2-determinant-state `CIWavefunction` with
opposite filling in impurity, mirror bath site.

E.g. `D = Dict{UInt64,Float64}` for bit component `UInt64`
and vector component `Vector{Float64}`.
"""
function starting_CIWavefunction(
    D::Type, n_v_bit::Int, n_c_bit::Int, n_v_vector::Int, n_c_vector::Int, excitation::Int
)
    K, V = keytype(D), valtype(D)
    s1 = slater_start(K, 0b0110, n_v_bit, n_c_bit, 0, 0)
    s2 = slater_start(K, 0b1001, n_v_bit, n_c_bit, 0, 0)
    v_dim = sum(i -> binomial(2 * (n_v_vector + n_c_vector), i), 0:excitation)
    v = zeros(V, v_dim)
    v[1] = one(V)
    result = CIWavefunction(
        Dict(s1 => copy(v), s2 => copy(v)),
        2 + n_v_bit + n_c_bit,
        n_v_vector,
        n_c_vector,
        excitation,
    )
    normalize!(result)
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
