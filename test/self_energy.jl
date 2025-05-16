using DMFT
using Fermions
using LinearAlgebra
using Test

@testset "self-energy" begin
    n_bath = 31
    U = 4.0
    μ = U / 2
    ϵ_imp = -μ
    n_v_bit = 1
    n_c_bit = 1
    e = 2
    n_kryl = 100
    n_kryl_gs = 20
    step_size = 0.02
    W = collect(-10:step_size:10)
    δ = 0.08
    # do not change parameters below
    n_sites = 1 + n_bath
    Z = W .+ im * δ

    Δ0 = hybridization_function_bethe_simple(n_bath)
    # Operators for positive frequencies. Negative ones are calculated by adjoint.
    fs = FockSpace(Orbitals(2 + n_v_bit + n_c_bit), FermionicSpin(1//2))
    c = annihilators(fs)
    n = occupations(fs)
    H_int = U * n[1, 1//2] * n[1, -1//2]
    d_dag = c[1, -1//2]' # d_↓^†
    q_dag = H_int * d_dag - d_dag * H_int  # q_↓^† = [H_int, d^†]
    O = [d_dag, q_dag]

    # impurity solver
    H, E0, ψ0 = init_system(Δ0, H_int, ϵ_imp, n_v_bit, n_c_bit, e, n_kryl_gs)
    C_plus = correlator_plus(H, E0, ψ0, O, n_kryl)
    C_minus = correlator_minus(H, E0, ψ0, map(adjoint, O), n_kryl)
    O_H = O[1]' * O[2] + O[2] * O[1]'
    Σ_H = dot(ψ0, O_H, ψ0)

    @testset "Poles" begin
        # impurity Green's fuction
        G_plus = Poles(copy(locations(C_plus)), abs.(amplitudes(C_plus)[1, :]))
        remove_poles_with_zero_weight!(G_plus)
        merge_degenerate_poles!(G_plus)
        merge_small_poles!(G_plus)
        G_minus = flip_spectrum(G_plus)
        G_imp = Poles(
            [locations(G_minus); locations(G_plus)],
            [amplitudes(G_minus); amplitudes(G_plus)],
        )

        Σ_H, Σ = self_energy_poles(-μ, Δ0, G_imp)
        @test Σ_H ≈ U / 2 atol = 100 * eps() # half-filling
        @test DMFT.moment(Σ, 0) ≈ U^2 / 4 atol = 1e-5 # bad agreement
    end # Poles

    @testset "correlator" begin
        # self-energies
        Σ_IFG = self_energy_IFG(C_plus, C_minus, Z, Σ_H)
        Σ_IFG_gauss = self_energy_IFG_gauss(C_plus, C_minus, W, δ, Σ_H)
        Σ_FG = self_energy_FG(C_plus, C_minus, Z)
        @test Σ_IFG != Σ_IFG_gauss
        @test Σ_IFG != Σ_FG
        @test norm(Σ_IFG - Σ_FG) * step_size < 0.0004 # they should be somewhat similar
        @test iszero(imag(first(Σ_IFG_gauss))) # exponential decay results in zero
        @test minimum(imag(Σ_IFG_gauss)) < minimum(imag(Σ_IFG)) # Gauss is steeper
    end # correlator
end # self-energy
