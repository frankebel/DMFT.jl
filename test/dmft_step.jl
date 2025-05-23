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
    d_dag = c[1, -1//2]' # d_↓^†
    q_dag = H_int * d_dag - d_dag * H_int  # q_↓^† = [H_int, d^†]
    O = [d_dag, q_dag]

    @testset "block Lanczos" begin
        # self-energy
        G_plus, G_minus, Δ_new, Δ_grid = dmft_step(
            Δ0, Δ0, H_int, μ, ϵ_imp, Z, n_v_bit, n_c_bit, e, O, n_kryl, n_kryl_gs, n_bath, δ
        )
        @test typeof(Δ_new) === Poles{Vector{Float64},Vector{Float64}}
        @test typeof(Δ_grid) === Vector{ComplexF64}
        @test length(Δ_new.a) === n_bath
        @test all(b -> isapprox(b, 1 / sqrt(n_bath) / 2; rtol=3E-3), Δ_new.b)
        # small weight loss due to truncated interval
        @test 0.24 <= DMFT.moment(Δ_new, 0) <= 0.25
        @test DMFT.moment(Δ_new, 1) < 200 * eps() # PHS

        # Gaussian broadening
        G_plus2, G_minus2, Δ_new2, Δ_grid2 = dmft_step_gauss(
            Δ0, Δ0, H_int, μ, -μ, W, n_v_bit, n_c_bit, e, O, n_kryl, n_kryl_gs, n_bath, δ
        )
        @test typeof(Δ_new2) === Poles{Vector{Float64},Vector{Float64}}
        @test typeof(Δ_grid2) === Vector{ComplexF64}
        @test length(Δ_new2) === n_bath
        amplitudes_without_zero = copy(amplitudes(Δ_new2))
        popat!(amplitudes_without_zero, cld(n_bath, 2))
        @test all(
            b -> isapprox(b, 1 / sqrt(n_bath) / 2; rtol=3E-3), amplitudes_without_zero
        )
        @test Δ_new2.b[cld(n_bath, 2)] ≈ 1 / sqrt(n_bath) / 2 rtol = 3e-2
        # small weight loss due to truncated interval
        @test 0.24999 <= DMFT.moment(Δ_new2, 0) <= 0.25
        @test DMFT.moment(Δ_new2, 1) < 200 * eps() # PHS

        # test equality
        @test locations(G_plus) == locations(G_plus2)
        @test amplitudes(G_minus) == amplitudes(G_minus2)
        # they are not the same
        @test locations(Δ_new) != locations(Δ_new2)
        @test amplitudes(Δ_new) != amplitudes(Δ_new2)
        @test Δ_grid != Δ_grid2

        # Discretization for Gaussian returned wrong number of poles.
        w = collect(-10:0.0002:10)
        g = similar(w)
        @. g = exp(-w^2)
        wrong_length = 0
        for i in 3:2:1001
            # test different number of poles
            dis = equal_weight_discretization(g, w, 0.04, i)
            length(dis) == i || (wrong_length += 1)
        end
        @test iszero(wrong_length)

        # FG
        Σ_FG = self_energy_FG(G_plus, G_minus, Z)
        Δ_grid3 = Δ0(Z .+ μ - Σ_FG)
        @test Δ_grid != Δ_grid3
    end # block Lanczos
end # DMFT step
