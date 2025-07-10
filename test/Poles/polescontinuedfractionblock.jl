using DMFT
using LinearAlgebra
using Test

@testset "PolesContinuedFractionBlock" begin
    @testset "constructor" begin
        loc = [[1 2; 2 3], [4 5; 5 6]]
        amp = [[7 8; 8 9]]
        scl = [10 11; 11 12]

        # inner contstructor
        P = PolesContinuedFractionBlock{Int,Int}(loc, amp, scl)
        @test P.loc === loc
        @test P.amp === amp
        @test P.scl === scl
        # wrong input
        # length mismatch
        @test_throws ArgumentError PolesContinuedFractionBlock{Int,Int}(loc, loc, scl)
        # matrices not hermitian
        @test_throws ArgumentError PolesContinuedFractionBlock{Int,Int}(
            [[1 2; 3 4], [4 5; 5 6]], amp, scl
        )
        # matrices have wrong size
        @test_throws DimensionMismatch PolesContinuedFractionBlock{Int,Int}(
            [[1;;], [4 5; 5 6]], amp, scl
        )
        @test_throws DimensionMismatch PolesContinuedFractionBlock{Int,Int}(
            loc, [[1;;]], scl
        )
        @test_throws DimensionMismatch PolesContinuedFractionBlock{Int,Int}(loc, amp, [1;;])

        # outer constructor
        P = PolesContinuedFractionBlock(loc, amp, scl)
        @test P.loc === loc
        @test P.amp === amp
        @test P.scl === scl
        # # default scale
        P = PolesContinuedFractionBlock(loc, amp)
        @test P.loc === loc
        @test P.amp === amp
        @test P.scl == [1 0; 0 1]

        # conversion of type
        P = PolesContinuedFractionBlock(loc, amp, scl)
        P_new = PolesContinuedFractionBlock{UInt,Float64}(P)
        @test typeof(P_new) === PolesContinuedFractionBlock{UInt,Float64}
        @test P_new.loc == loc
        @test P_new.amp == amp
        @test P_new.scl == scl
    end # constructor

    @testset "custom functions" begin
        @testset "amplitudes" begin
            loc = [[1 2; 2 3], [4 5; 5 6]]
            amp = [[7 8; 8 9]]
            P = PolesContinuedFractionBlock(loc, amp)
            @test amplitudes(P) === amp
        end # amplitudes

        @testset "evaluate_lorentzian" begin
            loc = [[1 2; 2 3], [4 5; 5 6]]
            amp = [[7 8; 8 9]]
            scl = [10 11; 11 12]
            P = PolesContinuedFractionBlock(loc, amp, scl)
            # single point
            @test norm(
                evaluate_lorentzian(P, 10, 1) - [
                    0.10758121432383735-0.8486851180203552im 0.1192067097465451-0.9291288301545801im
                    0.11920670974654493-0.92912883015458im 0.132513408476524-1.0172414419361508im
                ],
            ) < 10 * eps()
            # grid
            @test evaluate_lorentzian(P, [0.1, 0.3], 0.5) ==
                [evaluate_lorentzian(P, 0.1, 0.5), evaluate_lorentzian(P, 0.3, 0.5)]
            # non-hermitian entries
            P = PolesContinuedFractionBlock(
                [[1 0; 0 1], [1 0; 0 1]], [[1 0; 1 1]], [1 1; 0 1]
            )
            @test norm(
                evaluate_lorentzian(P, 1.7, 0.02) - [
                    1.4582589320316797-0.3997975176329503im -1.4331010284152463+0.22488790309674483im
                    -1.4331010284152463+0.22488790309674483im 0.025157903616432886-0.1749096145362053im
                ],
            ) < 10 * eps()
        end # evaluate_lorentzian

        @testset "locations" begin
            loc = [[1 2; 2 3], [4 5; 5 6]]
            amp = [[7 8; 8 9]]
            P = PolesContinuedFractionBlock(loc, amp)
            @test locations(P) === loc
        end # locations

        @testset "scale" begin
            loc = [[1 2; 2 3], [4 5; 5 6]]
            amp = [[7 8; 8 9]]
            scl = [10 11; 11 12]
            P = PolesContinuedFractionBlock(loc, amp, scl)
            @test DMFT.scale(P) === scl
        end # scale
    end # custom functions

    @testset "Core" begin
        @testset "Array" begin
            loc = [[1 2; 2 3], [4 5; 5 6], [7 8; 8 9]]
            amp = [[10 11; 11 12], [13 14; 14 15]]
            scl = [16 17; 17 18]
            P = PolesContinuedFractionBlock(loc, amp, scl)
            @test Array(P) == [
                1 2 10 11 0 0
                2 3 11 12 0 0
                10 11 4 5 13 14
                11 12 5 6 14 15
                0 0 13 14 7 8
                0 0 14 15 8 9
            ]
            P = PolesContinuedFractionBlock(loc, amp)
            @test Array(P) == [
                1 2 10 11 0 0
                2 3 11 12 0 0
                10 11 4 5 13 14
                11 12 5 6 14 15
                0 0 13 14 7 8
                0 0 14 15 8 9
            ]
        end # Array
    end # Core

    @testset "Base" begin
        @testset "eltype" begin
            loc = Matrix{Float64}[[1 2; 2 3], [4 5; 5 6]]
            amp = [[7 8; 8 9]]
            P = PolesContinuedFractionBlock(loc, amp)
            @test eltype(P) === Float64
        end # eltype

        @testset "length" begin
            loc = [[1 2; 2 3], [4 5; 5 6]]
            amp = [[7 8; 8 9]]
            P = PolesContinuedFractionBlock(loc, amp)
            @test length(P) == 2
        end # length

        @testset "size" begin
            loc = [[1 2; 2 3], [4 5; 5 6]]
            amp = [[7 8; 8 9]]
            P = PolesContinuedFractionBlock(loc, amp)
            @test size(P) == (2, 2)
            @test size(P, 1) === 2
            @test size(P, 2) === 2
        end # size
    end # Base
end # PolesContinuedFractionBlock
