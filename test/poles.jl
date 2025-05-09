using DMFT
using Distributions
using LinearAlgebra
using Test

@testset "Poles" begin
    V = Vector{Float64}
    M = Matrix{Float64}
    @testset "constructor" begin
        a = sort(rand(10))
        bv = rand(10) # vector
        bm = rand(2, 10) # matrix

        # inner constructor
        P = Poles{V,V}(a, bv)
        @test P.a === a
        @test P.b === bv
        P = Poles{V,M}(a, bm)
        @test P.a === a
        @test P.b === bm

        # outer constructor
        P = Poles(a, bv)
        @test typeof(P) === Poles{V,V}
        @test_throws TypeError Poles(rand(ComplexF64, 10), bv)
        @test_throws DimensionMismatch Poles(a, rand(9))
        @test_throws DimensionMismatch Poles(a, rand(2, 9))

        # conversion of type
        a = collect(1:10)
        b = collect(11:20)
        P = Poles(a, b)
        P_new = Poles{Vector{Float64},Vector{Int}}(P)
        @test typeof(P_new) === Poles{Vector{Float64},Vector{Int}}
        @test P_new.a == P.a
        @test P_new.b == P.b

        @testset "blockdiagonal" begin
            A = [rand(2, 2) for _ in 1:5]
            B = [rand(2, 2) for _ in 1:4]
            E0 = rand(Float64)
            S_sqrt = rand(2, 2)
            P = DMFT._poles(A, B, E0, S_sqrt)
            @test typeof(P.a) == Vector{Float64}
            @test length(P.a) == 10
            @test typeof(P.b) == Matrix{Float64}
            @test size(P.b) == (2, 10)
        end # blockdiagonal
    end # constructor

    @testset "getters" begin
        a = sort(rand(10))
        bv = rand(10) # vector
        bm = rand(2, 10) # matrix
        Pv = Poles(a, bv)
        Pm = Poles(a, bm)

        @testset "locations" begin
            @test locations(Pv) === Pv.a
            @test locations(Pm) === Pv.a
        end # locations

        @testset "amplitudes" begin
            @test amplitudes(Pv) === Pv.b
            @test amplitudes(Pm) === Pm.b
        end # amplitudes
    end # getters

    @testset "custom functions" begin
        @testset "evaluation" begin
            @testset "Lorentzian" begin
                a = collect(1.0:10)
                # single point
                z = 10 + im
                # b::Vector
                b = collect(0.1:0.1:1)
                P = Poles(a, b)
                foo = P(z)
                @test typeof(P(z)) === ComplexF64
                @test abs(P(z) - sum(i -> (0.1 * i)^2 / (10 + im - i), 1:10)) < eps()
                # b::Matrix
                b = reshape(collect(0.1:0.1:2), (2, 10))
                P = Poles(a, b)
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
                P = Poles(a, b)
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

            @testset "loggauss" begin
                Λ = 1.2
                N = 150
                a = grid_log(1, Λ, N)
                grid = [-reverse(a); 0; a]
                G = greens_function_bethe_grid(grid)
                W = collect(-2:0.001:2)
                popat!(W, findfirst(iszero, W)) # exclude ω == 0
                A = spectral_function_loggauss(G, W, 0.2)
                A .*= π
                @test all(>=(0), A) # positive semidefinite
                @test norm(A - reverse(A)) * 0.001 < 10 * eps() # symmetry
                @test abs(A[2000] - 2) < 0.01 # Luttinger pinning
                @test first(A) < 1e-6 # decay for ω → ±∞
                @test last(A) < 1e-6 # decay for ω → ±∞
            end # loggauss
        end # evaluate

        @testset "weights" begin
            # vector
            P = Poles([-1.0, 0.0, 0.5], [0.25, 1.5, 2.5])
            @test weights(P) == [0.0625, 2.25, 6.25]
            # matrix
            P = Poles([-1.0, 0.0, 0.5], [0.25 1.5 2.5; 1.0 2.0 0.75])
            @test weights(P) ==
                [[0.0625 0.25; 0.25 1.0], [2.25 3.0; 3.0 4.0], [6.25 1.875; 1.875 0.5625]]
        end # weights

        @testset "moment" begin
            # vector
            P = Poles([-0.5, 0.0, 0.5], [0.25, 1.5, 0.25])
            @test DMFT.moment(P) == 2.375
            @test iszero(DMFT.moment(P, 1))
            @test DMFT.moment(P, 2) == 0.03125
            @test iszero(DMFT.moment(P, 101))
            # matrix
            P = Poles([-0.5, 0.0, 0.5], [0.25 1.5 0.25; 0.5 0.75 2.5])
            @test DMFT.moment(P) == [2.375 1.875; 1.875 7.0625]
            @test DMFT.moment(P, 1) == [0.0 0.25; 0.25 3.0]
        end # moment

        @testset "moments" begin
            # vector
            P = Poles([-0.5, 0.0, 0.5], [0.25, 1.5, 0.25])
            @test moments(P, 0:1) == [2.375, 0]
            @test all(iszero, moments(P, 1:2:11))
            # matrix
            P = Poles([-0.5, 0.0, 0.5], [0.25 1.5 0.25; 0.5 0.75 2.5])
            @test moments(P, 0:1) == [[2.375 1.875; 1.875 7.0625], [0.0 0.25; 0.25 3.0]]
        end # moments

        @testset "to_grid" begin
            # all poles within grid, middle pole centerd
            A = Poles([0.1, 0.2, 0.3], [5.0, -10.0, 1.0])
            grid = [0.1, 0.3]
            B = to_grid(A, grid)
            @test B.a == [0.1, 0.3]
            @test norm(B.b - [sqrt(75), sqrt(51)]) < 10 * eps()
            # all poles within grid, middle pole not centered
            A = Poles([0.1, 0.25, 0.3], [5.0, -10.0, 1.0])
            grid = [0.1, 0.3]
            B = to_grid(A, grid)
            @test B.a == [0.1, 0.3]
            @test norm(B.b - [sqrt(50), sqrt(76)]) < 10 * eps()
            # pole outside grid
            A = Poles([0.0, 1.0], [5.0, -10.0])
            grid = [0.1, 0.3]
            B = to_grid(A, grid)
            @test B.a == [0.1, 0.3]
            @test B.b == [5.0, 10.0]
            # poles very close to grid
            A = Poles([4e-16, 0.9999999999999998], [4.0, 5.0])
            grid = [0.0, 1.0]
            B = to_grid(A, grid)
            @test B.a == [0.0, 1.0]
            @test B.b == [4.0, 5.0]
        end  # to_grid

        @testset "merge negative weight" begin
            # equidistant grid
            a = [-0.5, 0.5, 1.5]
            b = [1.5, -0.5, 5.0]
            foo = merge_negative_weight!(Poles(a, b))
            @test foo.a === a
            @test foo.b === b
            @test a == [-0.5, 0.5, 1.5]
            @test b == [1.25, 0.0, 4.75]
            # not equidistant grid
            a = [-0.5, 0.0, 1.5]
            b = [1.5, -0.5, 5.0]
            merge_negative_weight!(Poles(a, b))
            @test a == [-0.5, 0.0, 1.5]
            @test b == [1.125, 0.0, 4.875]
            # first pole negative
            a = [0.0, 1.0, 5.0]
            b = [-1.0, 0.5, 2.25]
            merge_negative_weight!(Poles(a, b))
            @test a == [0.0, 1.0, 5.0]
            @test b == [0.0, 0.0, 1.75]
            # last pole negative
            a = [0.0, 1.0, 5.0]
            b = [2.25, 0.5, -1.0]
            merge_negative_weight!(Poles(a, b))
            @test a == [0.0, 1.0, 5.0]
            @test b == [1.75, 0.0, 0.0]
            # weight exactly cancel
            a = [0.0, 1.0, 5.0]
            b = [-1.0, 0.5, 0.5]
            merge_negative_weight!(Poles(a, b))
            @test a == [0.0, 1.0, 5.0]
            @test b == [0.0, 0.0, 0.0]
            # symmetric case
            a = [-2.0, -0.5, 0.0, 0.5, 2.0]
            b = [5.0, -2.0, 1.0, -2.0, 5.0]
            merge_negative_weight!(Poles(a, b))
            @test a == [-2.0, -0.5, 0.0, 0.5, 2.0]
            @test norm(b - [3.5, 0.0, 0.0, 0.0, 3.5]) < 10 * eps()
            # previous pole would get negative weight
            a = [-1.0, -0.5, 0.0, 1.5]
            b = [2.0, 1.5, -2.5, 5.0]
            merge_negative_weight!(Poles(a, b))
            @test a == [-1.0, -0.5, 0.0, 1.5]
            @test b == [1.7, 0.0, 0.0, 4.3]
        end # merge negative weight

        @testset "continued fraction" begin
            function evaluate_cont_frac(a, b, z::Complex)
                result = zero(z)
                for (a, b) in zip(reverse(a[2:end]), reverse(b))
                    result = abs2(b) / (z - a - result)
                end
                result = 1 / (z - a[1] - result)
                return result
            end

            G = greens_function_bethe_simple(101)
            a, b = DMFT._continued_fraction(G)
            @test length(a) === 101
            @test length(b) === 100
            # don't have intuition who poles should look, but test below pass
            @test all(i -> isapprox(i, 0.5; atol=1000 * eps()), b)
            @test all(i -> i < 1000 * eps(), @view a[2:end])
            # evaluate
            z = 0.1im
            @test norm(G(z) - evaluate_cont_frac(a, b, z)) < 10 * eps()
            z = 1.2 + 0.1im
            @test norm(G(z) - evaluate_cont_frac(a, b, z)) < 10 * eps()
            z1 = evaluate_cont_frac(a, b, -0.8 + 0.1im)
            z2 = evaluate_cont_frac(a, b, 0.8 + 0.1im)
            @test real(z1) ≈ -real(z2) rtol = 2000 * eps()
            @test imag(z1) ≈ imag(z2) rtol = 7000 * eps()
        end # continued fraction

        @testset "merge degenerate poles!" begin
            P = Poles([0.2, 0.3, 0.6], [0.25, 0.75, 1.5])
            # manual tolerance
            P1 = copy(P)
            @test merge_degenerate_poles!(P1, 0.11) === P1
            @test P1.a == [0.2, 0.6]
            @test P1.b == [sqrt(0.625), 1.5]
            # default tolerance too small
            P1 = copy(P)
            @test merge_degenerate_poles!(P1) === P1
            @test P1.a == [0.2, 0.3, 0.6]
            @test P1.b == [0.25, 0.75, 1.5]
            # default tolerance
            P1 = copy(P)
            P1.a[2] = 0.5999999999999
            @test merge_degenerate_poles!(P1) === P1
            @test P1.a == [0.2, 0.5999999999999]
            @test P1.b == [0.25, sqrt(2.8125)]
            # negative poles
            P = Poles([-0.6, -0.3, -0.2], [1.5, 0.75, 0.25])
            @test merge_degenerate_poles!(P, 0.11) === P
            @test P.a == [-0.6, -0.2]
            @test P.b == [1.5, sqrt(0.625)]
            # poles around zero
            P = Poles([-0.03, -0.01, 0.01, 0.05], [0.25, 0.75, 1.5, 2.5])
            @test merge_degenerate_poles!(P, 0.04) === P
            @test P.a == [0.0, 0.05]
            @test P.b == [sqrt(2.875), 2.5]
            # poles at exactly same location
            P = Poles([-0.5, -0.5, 0.0, 0.05], [0.25, 0.75, 1.5, 2.5])
            @test merge_degenerate_poles!(P, 0) === P
            @test P.a == [-0.5, 0.0, 0.05]
            @test P.b == [sqrt(0.625), 1.5, 2.5]
        end # merge degenerate poles!

        @testset "merge negative locations to zero!" begin
            P = Poles([-0.1, -0.0, 0.0, 0.2], [0.5, 0.75, 2.5, 1.5])
            @test merge_negative_locations_to_zero!(P) === P
            @test P.a == [0.0, 0.2]
            @test P.b == [sqrt(7.0625), 1.5]
            # degeneracy at zero
            P = Poles([-0.0, -0.0, 0.0, 0.2], [0.5, 0.75, 2.5, 1.5])
            @test merge_negative_locations_to_zero!(P) === P
            @test P.a == [0.0, 0.2]
            @test P.b == [sqrt(7.0625), 1.5]
            # no negative location
            P = Poles([0.1, 0.1, 0.5, 1.0], [0.5, 0.75, 2.5, 1.5])
            @test merge_negative_locations_to_zero!(P) === P
            @test P.a == [0.1, 0.1, 0.5, 1.0]
            @test P.b == [0.5, 0.75, 2.5, 1.5]
        end # merge negative locations to zero!

        @testset "merge small poles!" begin
            # first index
            a = [-1.0, 0.0, 1.5]
            b = [0.5, -2.5, 5.0]
            P = Poles(a, b)
            @test merge_small_poles!(P, 1.0) === P
            @test P.a == [0.0, 1.5]
            @test P.b == [sqrt(6.5), 5.0]
            # last index
            a = [-1.0, 0.0, 1.5]
            b = [5.0, -2.5, 0.5]
            P = Poles(a, b)
            @test merge_small_poles!(P, 1.0) === P
            @test P.a == [-1.0, 0.0]
            @test P.b == [5.0, sqrt(6.5)]
            # middle index
            a = [-1.0, 0.0, 1.5]
            b = [5.0, 0.5, -2.5]
            P = Poles(a, b)
            @test merge_small_poles!(P, 1.0) === P
            @test P.a == [-1.0, 1.5]
            @test P.b == [sqrt(25.15), sqrt(6.35)]
        end # merge small poles!

        @testset "remove poles with zero weight!" begin
            a = collect(1:6)
            b = [0, 7, 0, 9, 0, -0.0]
            P = Poles(a, b)
            @test remove_poles_with_zero_weight!(P, true) === P
            @test P.a == [2, 4]
            @test P.b == [7, 9]
            # pole at a=0
            a = [-1, 0, 1]
            b = [2, 0, 0]
            P = Poles(copy(a), copy(b))
            @test remove_poles_with_zero_weight!(P) === P
            @test P.a == [-1]
            @test P.b == [2]
            P = Poles(copy(a), copy(b))
            @test remove_poles_with_zero_weight!(P, false) === P
            @test P.a == [-1, 0]
            @test P.b == [2, 0]
        end # remove poles with zero weight!

        @testset "remove poles with zero weight" begin
            a = collect(1:6)
            b = [0, 7, 0, 9, 0, -0.0]
            P = Poles(a, b)
            P_new = remove_poles_with_zero_weight(P)
            @test typeof(P_new) === Poles{Vector{Int},Vector{Float64}}
            @test P_new.a == [2, 4]
            @test P_new.b == [7, 9]
            @test P.a == collect(1:6)
            @test P.b == [0, 7, 0, 9, 0, 0]
        end # remove poles with zero weight
    end # custom functions

    @testset "Core" begin
        @testset "Array" begin
            a = collect(1:5)
            b = collect(6:10)
            P = Poles(a, b)
            m = Array(P)
            @test typeof(m) === Matrix{Int}
            @test m == [
                0 6 7 8 9 10
                6 1 0 0 0 0
                7 0 2 0 0 0
                8 0 0 3 0 0
                9 0 0 0 4 0
                10 0 0 0 0 5
            ]
            # poles with zero weight
            a = collect(1:5)
            b = [6, 7, 0, 9, 0]
            P = Poles(a, b)
            m = Array(P)
            @test m == [
                0 6 7 0 9 0
                6 1 0 0 0 0
                7 0 2 0 0 0
                0 0 0 3 0 0
                9 0 0 0 4 0
                0 0 0 0 0 5
            ]
            # correct promotion
            a = collect(1:2)
            b = [1.1, 5.5]
            P = Poles(a, b)
            m = Array(P)
            @test m == [
                0 1.1 5.5
                1.1 1 0
                5.5 0 2
            ]
        end # Array
    end # Core

    @testset "Base" begin
        @testset "copy" begin
            a = collect(1:5)
            b = collect(6:10)
            A = Poles(a, b)
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

        @testset "length" begin
            P = Poles(rand(10), rand(10))
            @test length(P) === 10
            append!(P.a, 0)
            @test_throws ArgumentError length(P)
            # b::Matrix
            P = Poles(rand(10), rand(2, 10))
            @test length(P) === 10
            append!(P.a, 0)
            @test_throws ArgumentError length(P)
        end # length

        @testset "allunique" begin
            @test allunique(Poles([0.1, 0.0, -0.5], rand(3)))
            @test !allunique(Poles([0.1, 0.0, 0.1], rand(3)))
            @test !allunique(Poles([-0.1, 0.0, -0.1], rand(3)))
            @test !allunique(Poles([0.0, 0.0, -0.1], rand(3)))
            @test !allunique(Poles([-0.0, 0.0, -0.1], rand(3)))
        end # allunique

        @testset "issorted" begin
            @test issorted(Poles([-0.3, 0.0, 0.1], rand(3)))
            @test issorted(Poles([-0.0, 0.0, 0.1], rand(3)))
            @test issorted(Poles([0.0, 0.0, 0.1], rand(3)))
            @test !issorted(Poles([0.0, -0.0, 0.1], rand(3)))
            @test !issorted(Poles([0.0, 0.2, 0.1], rand(3)))
            @test issorted(Poles([0.2, 0.1, 0.0], rand(3)); rev=true)
        end # issorted

        @testset "sort!" begin
            a = [2, 1]
            b = [3, 4]
            A = Poles(a, b)
            B = sort!(A)
            @test B === A
            @test A.a == [1, 2]
            @test A.b == [4, 3]
        end # sort!

        @testset "sort" begin
            a = [2, 1]
            b = [3, 4]
            A = Poles(a, b)
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
            A = Poles(a, Ab)
            B = Poles(a, Bb)
            C = A - B
            # original must be untouched
            @test A.a == [-1.0, 0.0, 5.0]
            @test A.b == [5.0, 6.0, 7.0]
            @test B.a == [-1.0, 0.0, 5.0]
            @test B.b == [2.5, 3.0, 4.8]
            # new Poles
            @test C.a == [-1.0, 0.0, 5.0]
            @test C.b == [sqrt(18.75), sqrt(27), sqrt(25.96)]

            # same pole locations, 1 negative weight
            a = [-1.0, 0.0, 3.0]
            Ab = [5.0, 2.0, 7.0]
            Bb = [2.5, 3.0, 4.8]
            A = Poles(a, Ab)
            B = Poles(a, Bb)
            C = A - B
            # original must be untouched
            @test A.a == [-1.0, 0.0, 3.0]
            @test A.b == [5.0, 2.0, 7.0]
            @test B.a == [-1.0, 0.0, 3.0]
            @test B.b == [2.5, 3.0, 4.8]
            # new Poles
            @test C.a == [-1.0, 0.0, 3.0]
            @test norm(C.b - [sqrt(15), 0, sqrt(24.71)]) < 10 * eps()

            # different pole locations, result has no negative weight
            a = [-1.0, 0.0, 3.0]
            Ab = [5.0, 6.0, 7.0]
            Bb = [2.5, 3.0, 4.8]
            A = Poles([-1.0, 1.0, 3.0], Ab)
            B = Poles(a, Bb)
            C = A - B
            # original must be untouched
            @test A.a == [-1.0, 1.0, 3.0]
            @test A.b == [5.0, 6.0, 7.0]
            @test B.a == [-1.0, 0.0, 3.0]
            @test B.b == [2.5, 3.0, 4.8]
            # new Poles
            @test C.a == [-1.0, 0.0, 3.0]
            @test norm(C.b - [sqrt(18.75), sqrt(15), sqrt(37.96)]) < 10 * eps()

            # different pole locations, middle gets exactly zero weight
            a = [-1.0, 0.0, 4.0]
            Ab = [5.0, 6.0, 7.0]
            Bb = [2.5, sqrt(27), 4.8]
            A = Poles([-1.0, 1.0, 4.0], Ab)
            B = Poles(a, Bb)
            C = A - B
            # original must be untouched
            @test A.a == [-1.0, 1.0, 4.0]
            @test A.b == [5.0, 6.0, 7.0]
            @test B.a == [-1.0, 0.0, 4.0]
            @test B.b == [2.5, sqrt(27), 4.8]
            # new Poles
            @test C.a == [-1.0, 0.0, 4.0]
            @test norm(C.b - [sqrt(18.75), 0, sqrt(34.96)]) < 10 * eps()

            # different pole locations, middle gets negative weight
            a = [-1.0, 0.0, 4.0]
            Ab = [5.0, 6.0, 7.0]
            Bb = [2.5, 6.0, 4.8]
            A = Poles([-1.0, 1.0, 4.0], Ab)
            B = Poles(a, Bb)
            C = A - B
            # original must be untouched
            @test A.a == [-1.0, 1.0, 4.0]
            @test A.b == [5.0, 6.0, 7.0]
            @test B.a == [-1.0, 0.0, 4.0]
            @test B.b == [2.5, 6.0, 4.8]
            # new Poles
            @test C.a == [-1.0, 0.0, 4.0]
            @test norm(C.b - [sqrt(11.55), 0, sqrt(33.16)]) < 10 * eps()
        end # -

        @testset "inv" begin
            grid = collect(range(-1, 1; length=101))
            G = greens_function_bethe_grid(grid)
            a0, P = inv(G)
            @test length(P.a) === length(P.b) === 100 # originally 101 poles
            @test all(>=(0), P.b)
            # poles are symmetric
            @test abs(a0) < 10 * eps()
            @test norm(P.a + reverse(P.a)) < 100 * eps()
            @test norm(P.b - reverse(P.b)) < 100 * eps()
            @test sum(abs2.(P.b)) ≈ 0.25 atol = 1e-4 # total weight
            # evaluate
            z = 0.1im
            @test norm(G(z) - 1 / (z - a0 - P(z))) < 30 * eps()
            z = 1.2 + 0.1im
            @test norm(G(z) - 1 / (z - a0 - P(z))) < 10 * eps()
            z1 = 1 / (-0.8 + 0.1im - a0 - P(-0.8 + 0.1im))
            z2 = 1 / (0.8 + 0.1im - a0 - P(0.8 + 0.1im))
            @test real(z1) ≈ -real(z2) rtol = 2000 * eps()
            @test imag(z1) ≈ imag(z2) rtol = 7000 * eps()
        end # inv
    end # Base
end # Poles
