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

    Δ0 = hybridization_function_bethe_simple(n_bath)
    # Operators for positive frequencies. Negative ones are calculated by adjoint.
    fs = FockSpace(Orbitals(2 + n_v_bit + n_c_bit), FermionicSpin(1//2))
    c = annihilators(fs)
    n = occupations(fs)
    H_int = U * n[1, 1//2] * n[1, -1//2]
    d_dag = c[1, -1//2]' # d_↓^†
    q_dag = H_int * d_dag - d_dag * H_int  # q_↓^† = [H_int, d^†]
    O = [q_dag, d_dag]

    # impurity solver
    H, E0, ψ0 = init_system(Δ0, H_int, ϵ_imp, n_v_bit, n_c_bit, e, n_kryl_gs)
    C_plus = correlator_plus(H, E0, ψ0, O, n_kryl)
    C_minus = correlator_minus(H, E0, ψ0, map(adjoint, O), n_kryl)
    C = transpose(C_minus) + C_plus
    O_H = O[1]' * O[2] + O[2] * O[1]'
    Σ_H = dot(ψ0, O_H, ψ0)

    @testset "Poles" begin
        # impurity Green's fuction
        G_plus = PolesSum(copy(locations(C_plus)), map(i -> i[2, 2], weights(C_plus)))
        remove_zero_weight!(G_plus)
        merge_degenerate_poles!(G_plus)
        merge_small_poles!(G_plus)
        G_minus = flip_spectrum(G_plus)
        G_imp = G_minus + G_plus

        Σ_H, Σ = self_energy_dyson(-μ, Δ0, G_imp, -5:0.02:5)
        @test Σ_H ≈ U / 2 atol = 100 * eps() # half-filling
        @test DMFT.moment(Σ, 0) ≈ U^2 / 4 atol = 1e-5 # bad agreement
        @test !any(iszero, locations(Σ)) # no pole at 0 for metal
    end # Poles

    @testset "correlator" begin
        # TODO: continue here
        # self-energies
        Σ_IFG_lorentz = self_energy_IFG_lorentzian(Σ_H, C, W, δ)
        Σ_IFG_gauss = self_energy_IFG_gaussian(Σ_H, C, W, δ)
        Σ_FG_lorentz = self_energy_FG_lorentzian(C, W, δ)
        @test Σ_IFG_lorentz != Σ_IFG_gauss
        @test Σ_IFG_lorentz != Σ_FG_lorentz
        @test norm(Σ_IFG_lorentz - Σ_FG_lorentz) * step_size < 0.0004 # they should be somewhat similar
        @test iszero(imag(first(Σ_IFG_gauss))) # exponential decay results in zero
        @test minimum(imag(Σ_IFG_gauss)) < minimum(imag(Σ_IFG_lorentz)) # Gauss is steeper
    end # correlator
end # self-energy
