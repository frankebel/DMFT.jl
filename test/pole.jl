using DMFT
using Distributions
using LinearAlgebra
using Test

@testset "Pole" begin
    V = Vector{Float64}
    M = Matrix{Float64}
    @testset "constructor" begin
        a = sort(rand(10))
        bv = rand(10) # vector
        bm = rand(2, 10) # matrix

        # inner constructor
        P = Pole{V,V}(a, bv)
        @test P.a === a
        @test P.b === bv
        P = Pole{V,M}(a, bm)
        @test P.a === a
        @test P.b === bm

        # outer constructor
        P = Pole(a, bv)
        @test typeof(P) === Pole{V,V}
        @test_throws TypeError Pole(rand(ComplexF64, 10), bv)
        @test_throws DimensionMismatch Pole(a, rand(9))
        @test_throws DimensionMismatch Pole(a, rand(2, 9))

        # conversion of type
        a = collect(1:10)
        b = collect(11:20)
        P = Pole(a, b)
        P_new = Pole{Vector{Float64},Vector{Int}}(P)
        @test typeof(P_new) === Pole{Vector{Float64},Vector{Int}}
        @test P_new.a == P.a
        @test P_new.b == P.b

        @testset "blockdiagonal" begin
            A = [rand(2, 2) for _ in 1:5]
            B = [rand(2, 2) for _ in 1:4]
            E0 = rand(Float64)
            S_sqrt = rand(2, 2)
            P = DMFT._pole(A, B, E0, S_sqrt)
            @test typeof(P.a) == Vector{Float64}
            @test length(P.a) == 10
            @test typeof(P.b) == Matrix{Float64}
            @test size(P.b) == (2, 10)
        end # blockdiagonal
    end # constructor

    @testset "evaluation" begin
        @testset "Lorentzian" begin
            a = collect(1.0:10)
            # single point
            z = 10 + im
            # b::Vector
            b = collect(0.1:0.1:1)
            P = Pole(a, b)
            foo = P(z)
            @test typeof(P(z)) === ComplexF64
            @test abs(P(z) - sum(i -> (0.1 * i)^2 / (10 + im - i), 1:10)) < eps()
            # b::Matrix
            b = reshape(collect(0.1:0.1:2), (2, 10))
            P = Pole(a, b)
            foo = P(z)
            @test typeof(P(z)) === Matrix{ComplexF64}
            @test norm(
                P(z) - [
                    3.419109056634164-5.796080126589452im 3.669440321902302-6.127487634859227im
                    3.669440321902302-6.127487634859227im 3.9413975570979876-6.478614061451364im
                ],
            ) < eps()

            # grid
            z = rand(ComplexF64, 2)
            @test P(z) == [P(z[1]), P(z[2])]
        end # Lorentzian

        @testset "Gaussian" begin
            ω = 0.15
            σ = 0.04
            # b::Matrix
            a = [0.1, 0.2]
            b = reshape(collect(0.1:0.1:0.4), (2, 2))
            P = Pole(a, b)
            @test norm(
                P(ω, σ) - [
                    -1.5277637226549838-1.4345225621076145im -1.90970465331873-2.00833158695066im
                    -1.90970465331873-2.00833158695066im -2.2916455839824765-2.8690451242152295im
                ],
            ) < 10 * eps()

            # semicircular DOS
            G = greens_function_bethe_simple(3001)
            ω = collect(-3:0.01:3)
            σ = 0.01
            # constant broadening
            h = G(ω, σ)
            ex = π .* pdf.(Semicircle(1), ω) # exact solution
            @test norm(ex + imag(h)) < 0.2
            @test maximum(abs.(ex + imag(h))) < 0.12
            @test findmin(imag(h))[2] == cld(length(ω), 2) # symmetric
        end # Gaussian
    end # evaluate

    @testset "Core" begin
        @testset "Array" begin
            a = collect(1:5)
            b = collect(6:10)
            Δ = Pole(a, b)
            m = Array(Δ)
            @test typeof(m) === Matrix{Int}
            @test m == [
                0 6 7 8 9 10
                6 1 0 0 0 0
                7 0 2 0 0 0
                8 0 0 3 0 0
                9 0 0 0 4 0
                10 0 0 0 0 5
            ]
        end # Array
    end # Core

    @testset "Base" begin
        @testset "copy" begin
            a = collect(1:5)
            b = collect(6:10)
            A = Pole(a, b)
            B = copy(A)
            @test typeof(B) === typeof(A)
            @test B.a == A.a
            @test B.b == A.b
            # mutate values
            B.a[begin] = 100
            @test A.a == collect(1:5)
            @test B.a == [100, 2, 3, 4, 5]
            B.b[end] = 100
            @test A.b == collect(6:10)
            @test B.b == [6, 7, 8, 9, 100]
        end # copy
    end # Base
end # Pole
