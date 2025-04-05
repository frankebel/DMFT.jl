using DMFT
using Distributions
using Fermions
using Fermions.Lanczos
using Fermions.Wavefunctions
using LinearAlgebra
using Test

@testset "util" begin
    @testset "get_CI_parameters" begin
        @test get_CI_parameters(10, 5, 1, 1) == (4, 3, 3)
        @test get_CI_parameters(10, 6, 1, 1) == (4, 4, 2)
        @test get_CI_parameters(10, 4, 1, 1) == (4, 2, 4)
        @test get_CI_parameters(10, 5, 2, 1) == (5, 2, 3)
        @test get_CI_parameters(10, 5, 1, 2) == (5, 3, 2)
        @test get_CI_parameters(10, 6, 2, 1) == (5, 3, 2)
        @test get_CI_parameters(10, 4, 1, 2) == (5, 2, 3)
        # non-sensical input values still work
        @test get_CI_parameters(10, 10, 1, 1) == (4, 8, -2)
        @test get_CI_parameters(10, 0, 1, 1) == (4, -2, 8)
    end # get_CI_parameters

    @testset "init_system" begin
        # parameters
        n_bath = 31
        U = 4.0
        μ = U / 2
        n_v_bit = 1
        n_c_bit = 1
        e = 2
        n_kryl = 15
        E0_target = -21.527949603162277 # target ground state energy
        Δ = hybridization_function_bethe_simple(n_bath)
        fs = FockSpace(Orbitals(2 + n_v_bit + n_c_bit), FermionicSpin(1//2))
        n = occupations(fs)
        H_int = U * n[1, -1//2] * n[1, 1//2]
        H, E0, ψ0 = init_system(Δ, H_int, -μ, n_v_bit, n_c_bit, e, n_kryl)
        Hψ = H * ψ0
        H_avg = dot(ψ0, Hψ)
        H_sqr = dot(Hψ, Hψ)
        var_rel = H_sqr / H_avg^2 - 1
        @test H isa CIOperator
        @test abs(E0 / E0_target - 1) < 1E-4
        @test abs(H_avg / E0 - 1) < 1E-14
        @test var_rel < 4E-8
    end # init system

    @testset "orthogonalize_states" begin
        v1 = CIWavefunction(Dict(zero(UInt8) => rand(5), one(UInt8) => rand(5)), 4, 1, 1, 1)
        v2 = CIWavefunction(Dict(zero(UInt8) => rand(5)), 4, 1, 1, 1)
        V = [v1 v2]
        W, S_sqrt = orthogonalize_states(V)
        # W^† W = 𝟙
        foo = Matrix{Float64}(undef, 2, 2)
        mul!(foo, W', W)
        @test norm(foo - I) < 8 * eps()
        # V = W S^{1/2}
        bar = zero(V)
        mul!(bar, W, S_sqrt)
        for i in eachindex(V)
            @test norm(bar[i] - V[i]) < 8 * eps()
        end
    end # orthogonalize_states

    @testset "η_gaussian" begin
        η_0 = 0.01
        η_∞ = 0.04
        w = collect(-10:0.08:10)
        η1 = η_gaussian(η_0, η_∞, 1.0, w)
        @test typeof(η1) === typeof(w)
        @test length(η1) === length(w)
        @test norm(η1 - reverse(η1)) === 0.0 # symmetric
        @test η1[126] ≈ η_0 rtol = 5 * eps() # w = 0
        @test η1[1] == η_∞
        # smaller broadening
        η2 = η_gaussian(η_0, η_∞, 0.5, w)
        @test all(η2 .>= η1)
    end # η_gaussian

    @testset "Kondo temperature" begin
        @test temperature_kondo(0.3, -0.1, 0.1) == 0.04297872341114842
        @test temperature_kondo(0.2, -0.1, 0.015) == 0.00020610334475146955
    end # Kondo temperature

    @testset "find chemical potential" begin
        # Create a symmetric uniform density on `n_tot` poles.
        n_tot = 100 # number of poles ≙ filling for μ = ∞
        h = Diagonal(range(-1, 1, n_tot)) # uniform density in [-1, 1]
        Hk = [h]
        Z = collect(-10:0.01:10) .+ 0.05im
        Σ = [zero(h) for _ in eachindex(Z)]
        G = greens_function_local(Z, 0, Hk)
        # test half-filling
        μ, filling = find_chemical_potential(Z, Hk, Σ, n_tot / 2)
        @test μ ≈ 0 atol = 2e-3
        @test filling ≈ n_tot / 2 atol = 2e-2
    end # find chemical potential
end # util
