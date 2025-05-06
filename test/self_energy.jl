using DMFT
using Fermions
using LinearAlgebra
using Test

@testset "self-energy" begin
    n_bath = 31
    U = 4.0
    μ = U / 2
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
    H_int = U * c[1, 1//2]' * c[1, 1//2] * c[1, -1//2]' * c[1, -1//2]
    A = c[1, -1//2]' # f_↓^†
    B = A' * H_int - H_int * A' # [f_↓, H_int]
    B = B' # need to apply to ket
    O = [A, B]

    # impurity solver
    G_plus, G_minus, Σ_H = solve_impurity(
        Δ0, H_int, -μ, n_v_bit, n_c_bit, e, n_kryl_gs, n_kryl, O
    )

    @testset "Poles" begin
        G_imp = Poles([G_minus.a; G_plus.a], [G_minus.b[1, :]; G_plus.b[1, :]])
        sort!(G_imp)
        Σ_H, Σ = self_energy_poles(-μ, Δ0, G_imp)
        @test abs(Σ_H - U / 2) < 100 * eps() # half-filling
        @test Σ.a == Δ0.a
        @test norm(sum(abs2.(Σ.b)) - U^2 / 4) < 1000 * sqrt(eps())
    end # Poles

    @testset "correlator" begin
        # self-energies
        Σ_IFG = self_energy_IFG(G_plus, G_minus, Z, Σ_H)
        Σ_IFG_gauss = self_energy_IFG_gauss(G_plus, G_minus, W, δ, Σ_H)
        Σ_FG = self_energy_FG(G_plus, G_minus, Z)
        @test Σ_IFG != Σ_IFG_gauss
        @test Σ_IFG != Σ_FG
        @test norm(Σ_IFG - Σ_FG) * step_size < 0.0004 # they should be somewhat similar
        @test iszero(imag(first(Σ_IFG_gauss))) # exponential decay results in zero
        @test minimum(imag(Σ_IFG_gauss)) < minimum(imag(Σ_IFG)) # Gauss is steeper
    end # correlator
end # self-energy
