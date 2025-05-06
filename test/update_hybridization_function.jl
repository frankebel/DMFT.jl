using DMFT
using LinearAlgebra
using Test

@testset "update hybridization function" begin
    @testset "pole" begin
        Δ0 = Poles([1.0, 5.0], [0.1, 0.2])
        μ = Σ_H = 5.0 # cancel in PHS, take any value
        Σ = Poles([2.0, 4.0, 7.0], [0.5, 3.0, 0.05])
        Δ = update_hybridization_function(Δ0, μ, Σ_H, Σ)

        @test Δ0.a == [1.0, 5.0] # original must be unchanged
        @test Δ0.b == [0.1, 0.2] # original must be unchanged
        @test typeof(Δ) === Poles{Vector{Float64},Vector{Float64}}
        @test Δ.a == [1.0, 5.0]
        @test norm(Δ.b - [0.14886850687536038, 0.16684773795500135]) < 10 * eps()

        # exact value must not matter
        Δ1 = update_hybridization_function(Δ0, 0, 0, Σ)
        @test Δ.b == Δ1.b

        # PHS in → PHS out
        Δ0 = Poles([-0.5, 0.0, 0.5], [0.25, 2.0, 0.25])
        Σ = Poles([-2.125, 2.125], [0.7, 0.7])
        Δ = update_hybridization_function(Δ0, μ, Σ_H, Σ)
        @test norm(Δ.b - reverse(Δ.b)) < 200 * eps()

        # Σ = 0
        Δ0 = Poles([1.0, 5.0], [0.1, 0.2])
        μ = Σ_H = 0.0
        Σ = Poles([0.0], [0.0])
        Δ = update_hybridization_function(Δ0, μ, Σ_H, Σ)
        @test Δ.a == Δ0.a
    end # pole
end # update hybridization function
