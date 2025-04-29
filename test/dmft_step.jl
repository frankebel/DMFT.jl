using DMFT
using Fermions
using Fermions.Wavefunctions
using LinearAlgebra
using Test

@testset "DMFT step" begin
    n_bath = 301
    U = 4.0
    μ = U / 2
    ϵ_imp = -U / 2
    n_v_bit = 1
    n_c_bit = 1
    e = 1
    n_kryl = 100
    n_kryl_gs = 20
    W = collect(-10:0.0002:10)
    δ = 0.08
    grid_poles = collect(range(-5, 5; length=n_bath))
    # do not change parameters below
    n_sites = 1 + n_bath
    Z = W .+ im * δ

    Δ0 = hybridization_function_bethe_grid(grid_poles)
    Δ = copy(Δ0)

    # Operators for positive frequencies. Negative ones are calculated by adjoint.
    fs = FockSpace(Orbitals(2 + n_v_bit + n_c_bit), FermionicSpin(1//2))
    c = annihilators(fs)
    n = occupations(fs)
    H_int = U * n[1, 1//2] * n[1, -1//2]
    A = c[1, -1//2]' # d_↓^†
    B = A' * H_int - H_int * A' # [f_↓, H_int]
    B = B' # need to apply to ket
    O = [A, B]
    Ogs = [1.0 * n[1, -1//2], 1.0 * n[1, 1//2], 1.0 * n[1, 1//2] * n[1, -1//2]]

    @testset "Lanczos" begin
        G_imp, Σ_H, Σ, Δ, E0, expectation_values = dmft_step(
            Δ0, Δ, H_int, μ, ϵ_imp, n_v_bit, n_c_bit, e, O[1], Ogs, n_kryl, n_kryl_gs
        )
        # impurity Green's function
        @test G_imp.a == grid_poles
        weights = abs2.(G_imp.b)
        @test sum(weights) ≈ 1 atol = 100 * eps()
        @test norm(weights - reverse(weights)) < 10 * eps()
        # self-energy
        @test Σ_H ≈ U / 2 atol = 100 * eps()
        @test Σ.a == grid_poles
        weights = abs2.(Σ.b)
        @test sum(weights) ≈ U^2 / 4 atol = 2e-4
        @test norm(weights - reverse(weights)) < 1200 * eps() # ERROR: foo
        # new hybridization
        @test Δ.a == grid_poles
        weights = abs2.(Δ.b)
        @test sum(weights) ≈ 0.25 atol = 100 * eps()
        @test norm(weights - reverse(weights)) < 100 * eps()
        # GS energy
        @test E0 ≈ -33.15707032876605 atol = sqrt(eps())
        # expectation values
        @test expectation_values[1] ≈ 0.5 atol = 100 * eps()
        @test expectation_values[2] ≈ 0.5 atol = 100 * eps()
        @test expectation_values[3] ≈ 0.04355604136992631 atol = 100 * eps()
    end # Lanczos

    @testset "block Lanczos" begin
        # self-energy
        G_plus, G_minus, Δ_new, Δ_grid = dmft_step(
            Δ0, Δ0, H_int, μ, ϵ_imp, Z, n_v_bit, n_c_bit, e, O, n_kryl, n_kryl_gs, n_bath, δ
        )
        @test typeof(Δ_new) === Pole{Vector{Float64},Vector{Float64}}
        @test typeof(Δ_grid) === Vector{ComplexF64}
        @test length(Δ_new.a) === n_bath
        @test all(b -> isapprox(b, 1 / sqrt(n_bath) / 2; rtol=3E-3), Δ_new.b)
        # small weight loss due to truncated interval
        @test 0.24 <= sum(abs2.(Δ_new.b)) <= 0.25
        # PHS
        @test abs(sum(abs2.(Δ_new.b) .* Δ_new.a)) < 200 * eps()

        # Gaussian broadening
        G_plus2, G_minus2, Δ_new2, Δ_grid2 = dmft_step_gauss(
            Δ0, Δ0, H_int, μ, -μ, W, n_v_bit, n_c_bit, e, O, n_kryl, n_kryl_gs, n_bath, δ
        )
        @test typeof(Δ_new2) === Pole{Vector{Float64},Vector{Float64}}
        @test typeof(Δ_grid2) === Vector{ComplexF64}
        @test length(Δ_new2.a) === n_bath
        weights_without_zero = [Δ_new2.b[1:150]; Δ_new2.b[152:end]]
        @test all(b -> isapprox(b, 1 / sqrt(n_bath) / 2; rtol=3E-3), weights_without_zero)
        @test Δ_new2.b[n_bath ÷ 2 + 1] ≈ 1 / sqrt(n_bath) / 2 rtol = 3e-2
        # small weight loss due to truncated interval
        @test 0.24999 <= sum(abs2.(Δ_new2.b)) <= 0.25
        # PHS
        @test abs(sum(abs2.(Δ_new2.b) .* Δ_new2.a)) < 200 * eps()

        # test equality
        @test G_plus.a == G_plus2.a
        @test G_minus.b == G_minus2.b
        # they are not the same
        @test Δ_new.a != Δ_new2.a
        @test Δ_new.b != Δ_new2.b
        @test Δ_grid != Δ_grid2

        # Discretization for Gaussian returned wrong number of poles.
        w = collect(-10:0.0002:10)
        g = similar(w)
        @. g = exp(-w^2)
        wrong_length = 0
        for i in 3:2:1001
            # test different number of poles
            dis = equal_weight_discretization(g, w, 0.04, i)
            length(dis.a) == i || (wrong_length += 1)
        end
        @test wrong_length == 0

        # FG
        Σ_FG = self_energy_FG(G_plus, G_minus, Z)
        Δ_grid3 = update_hybridization_function(Δ0, μ, Z, Σ_FG)
        @test Δ_grid != Δ_grid3
    end # block Lanczos
end # DMFT step
