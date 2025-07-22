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

    @testset "matrix" begin
        P = PolesSumBlock([1, 2], [[1 0; 0 1], [0 0; 0 1]])
        PCF = PolesContinuedFractionBlock(P)
        @test DMFT.scale(PCF) == Diagonal([1, sqrt(2)])
        @test norm(Array(PCF) - [1 0 0 0; 0 1.5 0 0.5; 0 0 0 0; 0 0.5 0 1.5]) < 10 * eps()
        PS = PolesSumBlock(PCF)
        merge_degenerate_poles!(PS, 2 * eps())
        @test norm(locations(PS) - [1, 2]) < 10 * eps()
        @test norm(weights(PS)[1] - [1 0; 0 1]) < 10 * eps()
        @test norm(weights(PS)[2] - [0 0; 0 1]) < 10 * eps()
    end # matrix
end # conversion
