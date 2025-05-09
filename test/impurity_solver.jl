using DMFT
using Fermions
using LinearAlgebra
using Test

@testset "impurity_solver" begin
    # parameters
    n_bath = 31
    U = 4.0
    μ = U / 2
    ϵ_imp = -μ
    n_v_bit = 1
    n_c_bit = 1
    e = 2
    n_kryl = 50
    n_kryl_gs = 20
    w = collect(-10:0.0002:10)
    η = 0.08
    # do not change parameters below
    n_sites = 1 + n_bath
    Z = w .+ im * η

    Δ0 = hybridization_function_bethe_simple(n_bath)
    # Operators for positive frequencies. Negative ones are calculated by adjoint.
    fs = FockSpace(Orbitals(2 + n_v_bit + n_c_bit), FermionicSpin(1//2))
    c = annihilators(fs)
    H_int = U * c[1, 1//2]' * c[1, 1//2] * c[1, -1//2]' * c[1, -1//2]
    A = c[1, -1//2]' # f_↓^†
    B = A' * H_int - H_int * A' # [f_↓, H_int]
    B = B' # need to apply to ket
    O = [A, B]

    H, E0, ψ0 = init_system(Δ0, H_int, ϵ_imp, n_v_bit, n_c_bit, e, n_kryl_gs)

    @testset "Lanczos" begin
        V = Vector{Float64}
        # G+
        G_plus = g_plus(H, E0, ψ0, A, n_kryl)
        @test typeof(G_plus) === Poles{V,V}
        @test length(G_plus) === 50
        @test issorted(G_plus)
        @test all(>=(0), G_plus.a)
        @test sum(abs2.(G_plus.b)) ≈ 0.5 atol = 100 * eps()
        # G-
        G_minus = g_minus(H, E0, ψ0, A', n_kryl)
        @test typeof(G_minus) === Poles{V,V}
        @test length(G_minus) === 50
        @test issorted(G_minus)
        @test all(<=(0), G_minus.a)
        @test sum(abs2.(G_minus.b)) ≈ 0.5 atol = 100 * eps()
        # symmetry: first moment must be zero
        m1_pos = sum(G_plus.a .* abs2.(G_plus.b))
        m1_neg = sum(G_minus.a .* abs2.(G_minus.b))
        @test m1_pos + m1_neg < 100 * eps()
    end # Lanczos

    @testset "block Lanczos" begin
        G_plus, G_minus, Σ_H = solve_impurity(
            Δ0, H_int, -μ, n_v_bit, n_c_bit, e, n_kryl_gs, n_kryl, O
        )

        @test issorted(G_plus)
        @test length(G_plus.a) == length(O) * n_kryl
        @test size(G_plus.b) == (length(O), length(O) * n_kryl)

        @test issorted(G_minus)
        @test length(G_minus) == length(O) * n_kryl
        @test size(G_plus.b) == (length(O), length(O) * n_kryl)

        # both should have half weight
        bp = G_plus.b[1, :]
        bm = G_minus.b[1, :]
        @test sum(abs2.(bp)) ≈ 0.5 rtol = 10 * eps()
        @test sum(abs2.(bm)) ≈ 0.5 rtol = 10 * eps()

        # compare absolute moments of impurity Green's function
        m_pos = [sum(abs2.(bp) .* G_plus.a .^ i) for i in 0:10]
        m_neg = [sum(abs2.(bm) .* G_minus.a .^ i) for i in 0:10]
        ratio = m_pos ./ m_neg
        @test all(r -> isapprox(r, 1; rtol=100 * eps()), ratio[1:2:end])
        @test all(r -> isapprox(r, -1; rtol=100 * eps()), ratio[2:2:end])

        # evaluation
        gp = G_plus(Z)
        gm = G_minus(Z)
        G = map(g -> g[1, 1], gm) .+ map(g -> g[1, 1], gp)
        F = map(g -> g[1, 2], gm) .+ map(g -> g[2, 1], gp)
        Σ = F ./ G
        @test all(i -> i <= 0, imag(Σ))
        # Hartree term Re[Σ(ω=0^+)] = U/2
        @test Σ_H ≈ U / 2 rtol = 20 * eps()
        @test real(Σ[cld(length(w), 2)]) ≈ U / 2 rtol = 200 * eps()
    end # block Lanczos
end # impurity_solver
