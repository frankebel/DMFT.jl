using DMFT
using Test

@testset "grid" begin
    @testset "logarithmic grid" begin
        @test_throws ArgumentError grid_log(1, 1.0, 10)
        @test_throws ArgumentError grid_log(1, 2.0, 0)
        @test_throws ArgumentError grid_log(-16.0, 2.0, 1)
        @test grid_log(1, 2, 10) == [
            2^(-9), 2^(-8), 2^(-7), 2^(-6), 2^(-5), 2^(-4), 2^(-3), 2^(-2), 2^(-1), 2^(-0)
        ]
        @test grid_log(16.0, 2, 4) == [2.0, 4.0, 8.0, 16.0]
        @test grid_log(100, 500, 1) == [100.0]
    end # logarithmic grid
end # grid
