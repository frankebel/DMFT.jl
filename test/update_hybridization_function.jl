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
        @test typeof(Δ) === Poles{Vector{Float64},Vector{Float64}}
        @test length(Δ) == 8
        @test norm(
            locations(Δ) - [
                -0.9165752683383968,
                1.3033051770501383,
                2.0442740368422125,
                2.128903756501482,
                5.871454262575046,
                6.997629805986772,
                7.000846968921157,
                7.570161260461611,
            ],
        ) < 100 * eps()
        @test norm(
            amplitudes(Δ) - [
                0.08446264864682886,
                0.12054472900528726,
                0.008740652284334396,
                0.04635348846837134,
                0.05279009575030346,
                0.009459420434504227,
                0.0016934496982472338,
                0.1524166715976232,
            ],
        ) < 100 * eps()

        # exact value must not matter
        Δ = update_hybridization_function(Δ0, 0.0, 0.0, Σ)
        @test norm(
            locations(Δ) - [
                -0.9165752683383968,
                1.3033051770501383,
                2.0442740368422125,
                2.128903756501482,
                5.871454262575046,
                6.997629805986772,
                7.000846968921157,
                7.570161260461611,
            ],
        ) < 100 * eps()
        @test norm(
            amplitudes(Δ) - [
                0.08446264864682886,
                0.12054472900528726,
                0.008740652284334396,
                0.04635348846837134,
                0.05279009575030346,
                0.009459420434504227,
                0.0016934496982472338,
                0.1524166715976232,
            ],
        ) < 100 * eps()

        # PHS in → PHS out
        Δ0 = Poles([-0.5, 0.0, 0.5], [0.25, 2.0, 0.25])
        Σ = Poles([-2.125, 2.125], [0.7, 0.7])
        Δ = update_hybridization_function(Δ0, μ, Σ_H, Σ)
        @test DMFT.moment(Δ, 1) < 100 * eps()

        # Σ = 0
        Δ0 = Poles([1.0, 5.0], [-0.1, 0.2])
        μ = Σ_H = 0.0
        Σ = Poles([0.0], [0.0])
        Δ = update_hybridization_function(Δ0, μ, Σ_H, Σ)
        @test locations(Δ) == [1.0, 5.0]
        @test amplitudes(Δ) == [0.1, 0.2] # positive amplitudes
    end # pole
end # update hybridization function
