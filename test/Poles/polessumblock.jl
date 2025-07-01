using DMFT
using LinearAlgebra
using Test

@testset "PolesSumBlock" begin
    @testset "constructor" begin
        loc = collect(0:5)
        amp = reshape(collect(-5:6), (2, 6))

        # inner constructor
        P = PolesSumBlock{Int,Int}(loc, amp)
        @test P.loc === loc
        @test P.amp === amp
        # length mismatch
        @test_throws DimensionMismatch PolesSumBlock(rand(10), rand(10, 4))

        # outer constructor
        P = PolesSumBlock(loc, amp)
        @test P.loc === loc
        @test P.amp === amp

        # conversion of type
        P = PolesSumBlock(loc, amp)
        P_new = PolesSumBlock{UInt,Float64}(P)
        @test typeof(P_new) === PolesSumBlock{UInt,Float64}
        @test P_new.loc == loc
        @test P_new.amp == amp
    end # constructor

    @testset "custom functions" begin
        @testset "amplitudes" begin
            loc = collect(0:5)
            amp = reshape(collect(-5:6), (2, 6))
            P = PolesSumBlock(loc, amp)
            @test amplitudes(P) === amp
        end # amplitudes

        @testset "evaluate_gaussian" begin
            ω = 0.15
            σ = 0.04
            loc = [0.1, 0.2]
            amp = reshape(collect(0.1:0.1:0.4), (2, 2))
            P = PolesSumBlock(loc, amp)
            @test norm(
                evaluate_gaussian(P, ω, σ) - [
                    -1.5277637226549838-1.4345225621076145im -1.90970465331873-2.00833158695066im
                    -1.90970465331873-2.00833158695066im -2.2916455839824765-2.8690451242152295im
                ],
            ) < 10 * eps()
        end # evaluate_gaussian

        @testset "evaluate_lorentzian" begin
            ω = 10.0
            δ = 1.0
            loc = collect(1.0:10)
            amp = reshape(collect(0.1:0.1:2), (2, 10))
            # single point
            P = PolesSumBlock(loc, amp)
            @test norm(
                evaluate_lorentzian(P, ω, δ) - [
                    3.419109056634164-5.796080126589452im 3.669440321902302-6.127487634859227im
                    3.669440321902302-6.127487634859227im 3.9413975570979876-6.478614061451364im
                ],
            ) < eps()
            # grid
            ω = rand(2)
            @test evaluate_lorentzian(P, ω, δ) ==
                [evaluate_lorentzian(P, ω[1], δ), evaluate_lorentzian(P, ω[2], δ)]
        end # evaluate_lorentzian

        @testset "moment" begin
            P = PolesSumBlock([-0.5, 0.0, 0.5], [0.25 1.5 0.25; 0.5 0.75 2.5])
            @test DMFT.moment(P) == [2.375 1.875; 1.875 7.0625]
            @test DMFT.moment(P, 1) == [0.0 0.25; 0.25 3.0]
        end # moment

        @testset "moments" begin
            P = PolesSumBlock([-0.5, 0.0, 0.5], [0.25 1.5 0.25; 0.5 0.75 2.5])
            @test moments(P, 0:1) == [[2.375 1.875; 1.875 7.0625], [0.0 0.25; 0.25 3.0]]
        end # moments

        @testset "weight" begin
            P = PolesSumBlock([-1.0, 0.0, 0.5], [0.25 1.5 2.5; 1.0 2.0 0.75])
            @test_throws BoundsError weight(P, 0)
            @test weight(P, 1) == [0.0625 0.25; 0.25 1.0]
            @test weight(P, 2) == [2.25 3.0; 3.0 4.0]
            @test weight(P, 3) == [6.25 1.875; 1.875 0.5625]
            @test_throws BoundsError weight(P, 4)
        end # weight

        @testset "weights" begin
            P = PolesSumBlock([-1.0, 0.0, 0.5], [0.25 1.5 2.5; 1.0 2.0 0.75])
            @test weights(P) ==
                [[0.0625 0.25; 0.25 1.0], [2.25 3.0; 3.0 4.0], [6.25 1.875; 1.875 0.5625]]
        end # weights
    end # custom functions

    @testset "Base" begin
        @testset "length" begin
            P = PolesSumBlock(rand(10), rand(4, 10))
            @test length(P) === 10
        end
        @testset "size" begin
            P = PolesSumBlock(rand(10), rand(4, 10))
            @test size(P) == (4, 10)
            @test size(P, 1) === 4
            @test size(P, 2) === 10
        end # size
    end # base
end # PolesSumBlock
