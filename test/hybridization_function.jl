using DMFT
using LinearAlgebra
using Test

@testset "hybridization" begin
    @testset "Bethe lattice" begin
        V = Vector{Float64}

        @testset "analytic" begin
            # test different number types
            @test hybridization_function_bethe_analytic(-2) == -0.1339745962155614
            @test hybridization_function_bethe_analytic(-true) == -1//2
            @test hybridization_function_bethe_analytic(-1//2) ==
                -0.25 - 0.4330127018922193im
            @test hybridization_function_bethe_analytic(false) == -0.5im
            @test hybridization_function_bethe_analytic(0.1im) == -0.4524937810560445im
            @test hybridization_function_bethe_analytic(0.5) == 0.25 - 0.4330127018922193im
            @test hybridization_function_bethe_analytic(0x1) == 0.5
            @test hybridization_function_bethe_analytic(2.0) == 0.1339745962155614
            # # vary half-bandwidth
            @test hybridization_function_bethe_analytic(-1, 2) ==
                -0.5 - 0.8660254037844386im
            @test hybridization_function_bethe_analytic(0, 10) == -5im
            # Vector{Complex}
            Δ = hybridization_function_bethe_analytic([2.0 + 0.1im, 3.0 + 0.5im])
            @test typeof(Δ) === Vector{ComplexF64}
            @test length(Δ) === 2
            @test Δ[1] == hybridization_function_bethe_analytic(2.0 + 0.1im)
            @test Δ[2] == hybridization_function_bethe_analytic(3.0 + 0.5im)
        end # analytic

        @testset "simple" begin
            # 101 poles
            Δ = hybridization_function_bethe_simple(101)
            @test typeof(Δ) == Pole{V,V}
            @test length(Δ.a) === length(Δ.b) === 101
            @test sum(abs2.(Δ.b)) ≈ 0.25 rtol = 10 * eps()
            @test Δ.a[51] ≈ 0 atol = 10 * eps()
            @test norm(Δ.a + reverse(Δ.a)) < 50 * eps()
            @test norm(abs2.(Δ.b) - reverse(abs2.(Δ.b))) < 600 * eps()
            # 100 poles
            Δ = hybridization_function_bethe_simple(100)
            @test typeof(Δ) === Pole{V,V}
            @test length(Δ.a) === length(Δ.b) === 100
            @test sum(abs2.(Δ.b)) ≈ 0.25 rtol = 10 * eps()
            @test norm(Δ.a + reverse(Δ.a)) < 100 * eps()
            @test norm(abs2.(Δ.b) - reverse(abs2.(Δ.b))) < 600 * eps()
            # 101 poles, D = 2
            Δ = greens_function_bethe_simple(101, 2)
            @test sum(abs2.(Δ.b)) ≈ 1.0 rtol = 10 * eps()
        end # simple

        @testset "equal weight" begin
            @test_throws DomainError hybridization_function_bethe_equal_weight(2)
            # D = 1
            Δ = hybridization_function_bethe_equal_weight(101)
            @test typeof(Δ) === Pole{V,V}
            @test length(Δ.a) === length(Δ.b) === 101
            @test all(i -> i === 1 / sqrt(101) / 2, Δ.b)
            @test norm(Δ.a + reverse(Δ.a)) === 0.0
            @test issorted(Δ.a)
            # D = 4
            Δ = hybridization_function_bethe_equal_weight(101, 4)
            @test all(i -> i === 2 / sqrt(101), Δ.b)
        end # equal weight
    end # Bethe lattice
end # hybridization
