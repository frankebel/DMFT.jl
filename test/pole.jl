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

    @testset "custom functions" begin
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

        @testset "to_grid_sqr" begin
            # all poles within grid, middle pole centerd
            A = Pole([0.1, 0.2, 0.3], [5.0, -10.0, 1.0])
            grid = [0.1, 0.3]
            B = to_grid_sqr(A, grid)
            @test B.a == [0.1, 0.3]
            @test norm(B.b - [0.0, -4.0]) < 10 * eps()
            # all poles within grid, middle pole not centered
            A = Pole([0.1, 0.25, 0.3], [5.0, -10.0, 1.0])
            grid = [0.1, 0.3]
            B = to_grid_sqr(A, grid)
            @test B.a == [0.1, 0.3]
            @test norm(B.b - [2.5, -6.5]) < 10 * eps()
            # pole outside grid
            A = Pole([0.0, 1.0], [5.0, -10.0])
            grid = [0.1, 0.3]
            B = to_grid_sqr(A, grid)
            @test B.a == [0.1, 0.3]
            @test B.b == [5.0, -10.0]
            # poles very close to grid
            A = Pole([4e-16, 0.9999999999999998], [4.0, 5.0])
            grid = [0.0, 1.0]
            B = to_grid_sqr(A, grid)
            @test B.a == [0.0, 1.0]
            @test B.b == [4.0, 5.0]
        end  # to_grid_sqr

        @testset "to_grid" begin
            # all poles within grid, middle pole centerd
            A = Pole([0.1, 0.2, 0.3], [5.0, -10.0, 1.0])
            grid = [0.1, 0.3]
            B = to_grid(A, grid)
            @test B.a == [0.1, 0.3]
            @test norm(B.b - [sqrt(75), sqrt(51)]) < 10 * eps()
            # all poles within grid, middle pole not centered
            A = Pole([0.1, 0.25, 0.3], [5.0, -10.0, 1.0])
            grid = [0.1, 0.3]
            B = to_grid(A, grid)
            @test B.a == [0.1, 0.3]
            @test norm(B.b - [sqrt(50), sqrt(76)]) < 10 * eps()
            # pole outside grid
            A = Pole([0.0, 1.0], [5.0, -10.0])
            grid = [0.1, 0.3]
            B = to_grid(A, grid)
            @test B.a == [0.1, 0.3]
            @test B.b == [5.0, 10.0]
            # poles very close to grid
            A = Pole([4e-16, 0.9999999999999998], [4.0, 5.0])
            grid = [0.0, 1.0]
            B = to_grid(A, grid)
            @test B.a == [0.0, 1.0]
            @test B.b == [4.0, 5.0]
        end  # to_grid

        @testset "move negative weight to neighbors" begin
            # equidistant grid
            a = [-0.5, 0.5, 1.5]
            b = [1.5, -0.5, 5.0]
            foo = move_negative_weight_to_neighbors!(Pole(a, b))
            @test foo.a === a
            @test foo.b === b
            @test a == [-0.5, 0.5, 1.5]
            @test b == [1.25, 0.0, 4.75]
            # not equidistant grid
            a = [-0.5, 0.0, 1.5]
            b = [1.5, -0.5, 5.0]
            move_negative_weight_to_neighbors!(Pole(a, b))
            @test a == [-0.5, 0.0, 1.5]
            @test b == [1.125, 0.0, 4.875]
            # first pole negative
            a = [0.0, 1.0, 5.0]
            b = [-1.0, 0.5, 2.25]
            move_negative_weight_to_neighbors!(Pole(a, b))
            @test a == [0.0, 1.0, 5.0]
            @test b == [0.0, 0.0, 1.75]
            # last pole negative
            a = [0.0, 1.0, 5.0]
            b = [2.25, 0.5, -1.0]
            move_negative_weight_to_neighbors!(Pole(a, b))
            @test a == [0.0, 1.0, 5.0]
            @test b == [1.75, 0.0, 0.0]
            # weight exactly cancel
            a = [0.0, 1.0, 5.0]
            b = [-1.0, 0.5, 0.5]
            move_negative_weight_to_neighbors!(Pole(a, b))
            @test a == [0.0, 1.0, 5.0]
            @test b == [0.0, 0.0, 0.0]
            # symmetric case
            a = [-2.0, -0.5, 0.0, 0.5, 2.0]
            b = [5.0, -2.0, 1.0, -2.0, 5.0]
            move_negative_weight_to_neighbors!(Pole(a, b))
            @test a == [-2.0, -0.5, 0.0, 0.5, 2.0]
            @test norm(b - [3.5, 0.0, 0.0, 0.0, 3.5]) < 10 * eps()
            # previous pole would get negative weight
            a = [-1.0, -0.5, 0.0, 1.5]
            b = [2.0, 1.5, -2.5, 5.0]
            move_negative_weight_to_neighbors!(Pole(a, b))
            @test a == [-1.0, -0.5, 0.0, 1.5]
            @test b == [1.7, 0.0, 0.0, 4.3]
        end # move negative weight to neighbors
    end # custom functions

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

        @testset "sort!" begin
            a = [2, 1]
            b = [3, 4]
            A = Pole(a, b)
            B = sort!(A)
            @test B === A
            @test A.a == [1, 2]
            @test A.b == [4, 3]
        end # sort!

        @testset "sort" begin
            a = [2, 1]
            b = [3, 4]
            A = Pole(a, b)
            B = sort(A)
            @test B !== A
            @test A.a == [2, 1]
            @test A.b == [3, 4]
            @test B.a == [1, 2]
            @test B.b == [4, 3]
        end # sort

        @testset "-" begin
            # same pole locations, result has no negative weight
            a = [-1.0, 0.0, 5.0]
            Ab = [5.0, 6.0, 7.0]
            Bb = [2.5, 3.0, 4.8]
            A = Pole(a, Ab)
            B = Pole(a, Bb)
            C = A - B
            # original must be untouched
            @test A.a == [-1.0, 0.0, 5.0]
            @test A.b == [5.0, 6.0, 7.0]
            @test B.a == [-1.0, 0.0, 5.0]
            @test B.b == [2.5, 3.0, 4.8]
            # new Pole
            @test C.a == [-1.0, 0.0, 5.0]
            @test C.b == [sqrt(18.75), sqrt(27), sqrt(25.96)]

            # same pole locations, 1 negative weight
            a = [-1.0, 0.0, 3.0]
            Ab = [5.0, 2.0, 7.0]
            Bb = [2.5, 3.0, 4.8]
            A = Pole(a, Ab)
            B = Pole(a, Bb)
            C = A - B
            # original must be untouched
            @test A.a == [-1.0, 0.0, 3.0]
            @test A.b == [5.0, 2.0, 7.0]
            @test B.a == [-1.0, 0.0, 3.0]
            @test B.b == [2.5, 3.0, 4.8]
            # new Pole
            @test C.a == [-1.0, 0.0, 3.0]
            @test norm(C.b - [sqrt(15), 0, sqrt(24.71)]) < 10 * eps()

            # different pole locations, result has no negative weight
            a = [-1.0, 0.0, 3.0]
            Ab = [5.0, 6.0, 7.0]
            Bb = [2.5, 3.0, 4.8]
            A = Pole(a, Ab)
            B = Pole([-1.0, 1.0, 3.0], Bb)
            C = A - B
            # original must be untouched
            @test A.a == [-1.0, 0.0, 3.0]
            @test A.b == [5.0, 6.0, 7.0]
            @test B.a == [-1.0, 1.0, 3.0]
            @test B.b == [2.5, 3.0, 4.8]
            # new Pole
            @test C.a == [-1.0, 0.0, 3.0]
            @test norm(C.b - [sqrt(18.75), sqrt(30), sqrt(22.96)]) < 10 * eps()

            # different pole locations, middle gets zero weight
            a = [-1.0, 0.0, 3.0]
            Ab = [5.0, 6.0, 7.0]
            Bb = [2.5, sqrt(54), 4.8]
            A = Pole(a, Ab)
            B = Pole([-1.0, 1.0, 3.0], Bb)
            C = A - B
            # original must be untouched
            @test A.a == [-1.0, 0.0, 3.0]
            @test A.b == [5.0, 6.0, 7.0]
            @test B.a == [-1.0, 1.0, 3.0]
            @test B.b == [2.5, sqrt(54), 4.8]
            # new Pole
            @test C.a == [-1.0, 0.0, 3.0]
            @test norm(C.b - [sqrt(18.75), 0, sqrt(7.96)]) < 10 * eps()

            # different pole locations, middle gets negative weight
            a = [-1.0, 0.0, 3.0]
            Ab = [5.0, 6.0, 7.0]
            Bb = [2.5, 8.0, 4.8]
            A = Pole(a, Ab)
            B = Pole([-1.0, 1.0, 3.0], Bb)
            C = A - B
            # original must be untouched
            @test A.a == [-1.0, 0.0, 3.0]
            @test A.b == [5.0, 6.0, 7.0]
            @test B.a == [-1.0, 1.0, 3.0]
            @test B.b == [2.5, 8.0, 4.8]
            # new Pole
            @test C.a == [-1.0, 0.0, 3.0]
            @test norm(C.b - [sqrt(13.75), 0, sqrt(2.96)]) < 10 * eps()
        end # -
    end # Base
end # Pole
