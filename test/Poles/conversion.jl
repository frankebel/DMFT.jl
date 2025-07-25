using DMFT
using LinearAlgebra
using Test

@testset "conversion" begin
    @testset "scalar" begin
        P = PolesContinuedFraction(5:10, 0.1:0.1:0.5, 0.25)
        PS = PolesSum(P)
        PCF = PolesContinuedFraction(PS)
        @test norm(locations(P) - locations(PCF)) < 50 * eps()
        @test norm(amplitudes(P) - amplitudes(PCF)) < 50 * eps()
        @test DMFT.scale(P) ≈ DMFT.scale(PCF) atol = 10 * eps()
        # early stopping
        PS = PolesSum([0, 1], [1, 0]) # second location has no weight
        PCF = PolesContinuedFraction(PS)
        @test length(PCF) == 1
        @test locations(PCF) == [0.0]
        @test amplitudes(PCF) == Float64[]
        @test DMFT.scale(PCF) === 1.0
    end # scalar

    @testset "block" begin
        P = PolesSumBlock([1, 2], [[1 0; 0 1], [0 0; 0 1]])
        PCF = PolesContinuedFractionBlock(P)
        @test DMFT.scale(PCF) == Diagonal([1, sqrt(2)])
        @test norm(Array(PCF) - [1 0 0 0; 0 1.5 0 0.5; 0 0 0 0; 0 0.5 0 1.5]) < 10 * eps()
        PS = PolesSumBlock(PCF)
        merge_degenerate_poles!(PS, 5 * eps())
        @test norm(locations(PS) - [1, 2]) < 10 * eps()
        @test norm(weights(PS)[1] - [1 0; 0 1]) < 10 * eps()
        @test norm(weights(PS)[2] - [0 0; 0 1]) < 10 * eps()
    end # block

    @testset "block → scalar" begin
        P = PolesSumBlock([0, 1], [[2 3im; -3im 4], [5 -6im; 6im 7]])
        foo = PolesSum(P, 1, 1)
        @test locations(foo) !== locations(P)
        @test locations(foo) == [0, 1]
        @test weights(foo) == [2, 5]
        foo = PolesSum(P, 1, 2)
        @test locations(foo) == [0, 1]
        @test weights(foo) == [3im, -6im]
        foo = PolesSum(P, 2, 1)
        @test locations(foo) == [0, 1]
        @test weights(foo) == [-3im, 6im]
        foo = PolesSum(P, 2, 2)
        @test locations(foo) == [0, 1]
        @test weights(foo) == [4, 7]
    end # block → scalar
end # conversion
