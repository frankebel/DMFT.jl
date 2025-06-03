using DMFT
using Fermions
using LinearAlgebra
using Test

@testset "correlator" begin
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
    n = occupations(fs)
    H_int = U * n[1, 1//2] * n[1, -1//2]
    d_dag = c[1, -1//2]' # d_↓^†
    q_dag = H_int * d_dag - d_dag * H_int  # q_↓^† = [H_int, d^†]
    O = [d_dag, q_dag]

    H, E0, ψ0 = init_system(Δ0, H_int, ϵ_imp, n_v_bit, n_c_bit, e, n_kryl_gs)

    @testset "Lanczos" begin
        V = Vector{Float64}
        # G+
        G_plus = correlator_plus(H, E0, ψ0, d_dag, n_kryl)
        @test typeof(G_plus) === Poles{V,V}
        @test length(G_plus) === 50
        @test issorted(G_plus)
        @test all(>=(0), locations(G_plus))
        @test all(>=(0), amplitudes(G_plus))
        @test DMFT.moment(G_plus, 0) ≈ 0.5 atol = 100 * eps()
        # G-
        G_minus = correlator_minus(H, E0, ψ0, d_dag', n_kryl)
        @test typeof(G_minus) === Poles{V,V}
        @test length(G_minus) === 50
        @test issorted(G_minus)
        @test all(<=(0), locations(G_minus))
        @test all(>=(0), amplitudes(G_minus))
        @test DMFT.moment(G_minus, 0) ≈ 0.5 atol = 100 * eps()
        # symmetry: first moment must be zero
        @test DMFT.moment(G_plus, 1) + DMFT.moment(G_minus, 1) ≈ 0 atol = 100 * eps()
    end # Lanczos

    @testset "block Lanczos" begin
        # C+
        C_plus = correlator_plus(H, E0, ψ0, O, n_kryl)
        @test typeof(C_plus) === Poles{Vector{Float64},Matrix{Float64}}
        @test issorted(C_plus)
        @test length(locations(C_plus)) == length(O) * n_kryl
        @test size(amplitudes(C_plus)) == (2, 2 * n_kryl)
        @test all(>=(0), locations(C_plus))
        # C-
        C_minus = correlator_minus(H, E0, ψ0, map(adjoint, O), n_kryl)
        @test typeof(C_minus) === Poles{Vector{Float64},Matrix{Float64}}
        @test issorted(C_minus)
        @test length(C_minus) == length(O) * n_kryl
        @test size(amplitudes(C_minus)) == (2, 2 * n_kryl)

        # G±
        G_plus = Poles(copy(locations(C_plus)), amplitudes(C_plus)[1, :])
        G_minus = Poles(copy(locations(C_minus)), amplitudes(C_minus)[1, :])

        # half-filling
        @test DMFT.moment(G_plus, 0) ≈ 0.5 rtol = 20 * eps()
        @test DMFT.moment(G_minus, 0) ≈ 0.5 rtol = 20 * eps()

        # compare absolute moments of impurity Green's function
        m_pos = moments(G_plus, 0:10)
        m_neg = moments(G_minus, 0:10)
        ratio = m_pos ./ m_neg
        @test all(r -> isapprox(r, 1; atol=500 * eps()), ratio[1:2:end])
        @test all(r -> isapprox(r, -1; atol=500 * eps()), ratio[2:2:end])

        # Hartree term
        O_H = O[1]' * O[2] + O[2] * O[1]'
        Σ_H = dot(ψ0, O_H, ψ0)
        @test Σ_H ≈ U / 2 rtol = 20 * eps()

        # evaluation
        cp = C_plus(Z)
        cm = C_minus(Z)
        G = map(c -> c[1, 1], cm) .+ map(c -> c[1, 1], cp)
        F = map(c -> c[1, 2], cm) .+ map(c -> c[2, 1], cp)
        Σ = F ./ G
        @test all(i -> i <= 0, imag(Σ))
        @test real(Σ[cld(length(w), 2)]) ≈ U / 2 rtol = 500 * eps()

        @testset "_flip_sign!" begin
            # `@.` allocates, custom function does not
            V = map(i -> rand(2, 2), 1:20)
            DMFT._flip_sign!(V)
            @. V = -V
            @test iszero(@allocated DMFT._flip_sign!(V))
            @test !iszero(@allocated @. V = -V)
        end # _flip_sign!
    end # block Lanczos
end # correlator
