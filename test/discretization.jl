using DMFT
using Fermions
using LinearAlgebra
using Test

@testset "discretization" begin
    @testset "Poles" begin
        # parameters
        n_bath = 101
        U = 2.0
        μ = U / 2
        ϵ_imp = -μ
        n_v_bit = 1
        n_c_bit = 1
        e = 1
        n_kryl = 100
        n_kryl_gs = 20

        Δ0 = hybridization_function_bethe_simple(n_bath)

        # Operators for positive frequencies. Negative ones are calculated by adjoint.
        fs = FockSpace(Orbitals(2 + n_v_bit + n_c_bit), FermionicSpin(1//2))
        c = annihilators(fs)
        n = occupations(fs)
        H_int = U * n[1, 1//2] * n[1, -1//2]
        O = c[1, -1//2]' # d_↓^†

        # initialize system
        H, E0, ψ0 = init_system(Δ0, H_int, ϵ_imp, n_v_bit, n_c_bit, e, n_kryl_gs)

        # impurity Green's functions
        G_plus = g_plus(H, E0, ψ0, O, n_kryl)
        G_minus = g_minus(H, E0, ψ0, O, n_kryl)
        G_imp = Poles([G_minus.a; G_plus.a], [G_minus.b; G_plus.b])
        sort!(G_imp)

        # self-energy
        Σ_H, Σ = self_energy_poles(ϵ_imp, Δ0, G_imp)
        merge_small_poles!(Σ)

        # new hypridization function
        Δ = update_hybridization_function(Δ0, μ, Σ_H, Σ)
        merge_degenerate_poles!(Δ)
        merge_small_poles!(Δ)
        Δ_new = discretize_similar_weight(Δ, sqrt(eps()), n_bath)

        @test length(Δ_new) == 101
        @test iszero(locations(Δ_new)[51])
        @test weights(Δ_new)[51] ≈ 0.002083461320853373 atol = 1e-10
        @test DMFT.moment(Δ_new, 0) ≈ 0.25 atol = 10 * eps()
        @test DMFT.moment(Δ_new, 1) ≈ 0.0 atol = 2000 * eps()
    end # Poles
end # discretization
