using DMFT
using Fermions
using Test

@testset "mask" begin
    @testset "mask_fe" begin
        f, e = mask_fe(UInt64, 1, 1, 1)
        @test f === UInt64(0b010_010)
        @test e === UInt64(0b100_100)
        f, e = mask_fe(UInt64, 4, 8, 9)
        @test f === UInt64(0b000000000111111110000_000000000111111110000)
        @test e === UInt64(0b111111111000000000000_111111111000000000000)
        f, e = mask_fe(BigMask{2,UInt64}, 10, 11, 12)
        @test f === BigMask((
            UInt64(0b0000000000111111111110000000000_000000000000111111111110000000000),
            UInt64(0b00),
        ))
        @test e === BigMask((
            UInt64(0b1111111111000000000000000000000_111111111111000000000000000000000),
            UInt64(0b11),
        ))

        # wrong arguments
        @test_throws ArgumentError mask_fe(UInt64, 0, 0, 33)
        @test_throws ArgumentError mask_fe(UInt64, -1, 1, 1)
        @test_throws ArgumentError mask_fe(UInt64, 1, -1, 1)
        @test_throws ArgumentError mask_fe(UInt64, 1, 1, -1)

        @test_throws MethodError mask_fe(Int64, 1, 1, 1)
    end # mask_fe

    @testset "get_excitation" begin
        f, e = mask_fe(UInt64, 4, 5, 6)
        @test get_excitation(UInt64(0b000000_11111_0000_000000_11111_0000), f, e) == 0
        # # flip all bits
        @test get_excitation(UInt64(0b111111_00000_0000_111111_00000_0000), f, e) == 22
        # # flip some bits
        @test get_excitation(UInt64(0b000000_11111_1111_000000_11111_1111), f, e) == 0
        @test get_excitation(UInt64(0b000000_11111_0000_000000_11110_0000), f, e) == 1
        @test get_excitation(UInt64(0b000000_11111_0000_000001_11111_0000), f, e) == 1
        @test get_excitation(UInt64(0b000000_11110_0000_000000_11111_0000), f, e) == 1
        @test get_excitation(UInt64(0b000001_11111_0000_000000_11111_0000), f, e) == 1
        @test get_excitation(UInt64(0b000000_11111_0000_000000_11100_0000), f, e) == 2
        @test get_excitation(UInt64(0b000000_11111_0000_000011_11111_0000), f, e) == 2
        @test get_excitation(UInt64(0b000000_11100_0000_000000_11111_0000), f, e) == 2
        @test get_excitation(UInt64(0b000011_11111_0000_000000_11111_0000), f, e) == 2
        @test get_excitation(UInt64(0b000000_11111_0000_000001_11110_0000), f, e) == 2
        @test get_excitation(UInt64(0b000001_11110_0000_000000_11111_0000), f, e) == 2
        @test get_excitation(UInt64(0b000000_11110_0000_000000_11110_0000), f, e) == 2
        @test get_excitation(UInt64(0b000001_11111_0000_000001_11111_0000), f, e) == 2
        @test get_excitation(UInt64(0b000001_11111_0000_000000_11110_0000), f, e) == 2
        @test get_excitation(UInt64(0b000000_11111_0000_000000_11000_0000), f, e) == 3
        @test get_excitation(UInt64(0b000000_11111_0000_000111_11111_0000), f, e) == 3
        @test get_excitation(UInt64(0b000000_11000_0000_000000_11111_0000), f, e) == 3
        @test get_excitation(UInt64(0b000111_11111_0000_000000_11111_0000), f, e) == 3
    end # get_excitation

    @testset "excitation!" begin
        f, e = mask_fe(UInt64, 4, 5, 6)
        d = Dict{UInt64,Vector{Int}}(
            0b000000_11111_0000_000000_11111_0000 => rand(Int, 2),
            0b111111_00000_0000_111111_00000_0000 => rand(Int, 2),
            0b000000_11111_1111_000000_11111_1111 => rand(Int, 2),
            0b000000_11111_0000_000000_11110_0000 => rand(Int, 2),
            0b000000_11111_0000_000001_11111_0000 => rand(Int, 2),
            0b000000_11110_0000_000000_11111_0000 => rand(Int, 2),
            0b000001_11111_0000_000000_11111_0000 => rand(Int, 2),
            0b000000_11111_0000_000000_11100_0000 => rand(Int, 2),
            0b000000_11111_0000_000011_11111_0000 => rand(Int, 2),
            0b000000_11100_0000_000000_11111_0000 => rand(Int, 2),
            0b000011_11111_0000_000000_11111_0000 => rand(Int, 2),
            0b000000_11111_0000_000001_11110_0000 => rand(Int, 2),
            0b000001_11110_0000_000000_11111_0000 => rand(Int, 2),
            0b000000_11110_0000_000000_11110_0000 => rand(Int, 2),
            0b000001_11111_0000_000001_11111_0000 => rand(Int, 2),
            0b000001_11111_0000_000000_11110_0000 => rand(Int, 2),
            0b000000_11111_0000_000000_11000_0000 => rand(Int, 2),
            0b000000_11111_0000_000111_11111_0000 => rand(Int, 2),
            0b000000_11000_0000_000000_11111_0000 => rand(Int, 2),
            0b000111_11111_0000_000000_11111_0000 => rand(Int, 2),
        )

        ψ = Wavefunction(copy(d))
        DMFT.excitation!(ψ, f, e, 0)
        @test Set(keys(ψ)) == Set([
            0b000000_11111_0000_000000_11111_0000, 0b000000_11111_1111_000000_11111_1111
        ])

        ψ = Wavefunction(copy(d))
        DMFT.excitation!(ψ, f, e, 1)
        @test Set(keys(ψ)) == Set([
            0b000000_11111_0000_000000_11111_0000,
            0b000000_11111_1111_000000_11111_1111,
            0b000000_11111_0000_000000_11110_0000,
            0b000000_11111_0000_000001_11111_0000,
            0b000000_11110_0000_000000_11111_0000,
            0b000001_11111_0000_000000_11111_0000,
        ])

        ψ = Wavefunction(copy(d))
        DMFT.excitation!(ψ, f, e, 2)
        @test Set(keys(ψ)) == Set([
            0b000000_11111_0000_000000_11111_0000,
            0b000000_11111_1111_000000_11111_1111,
            0b000000_11111_0000_000000_11110_0000,
            0b000000_11111_0000_000001_11111_0000,
            0b000000_11110_0000_000000_11111_0000,
            0b000001_11111_0000_000000_11111_0000,
            0b000000_11111_0000_000000_11100_0000,
            0b000000_11111_0000_000011_11111_0000,
            0b000000_11100_0000_000000_11111_0000,
            0b000011_11111_0000_000000_11111_0000,
            0b000000_11111_0000_000001_11110_0000,
            0b000001_11110_0000_000000_11111_0000,
            0b000000_11110_0000_000000_11110_0000,
            0b000001_11111_0000_000001_11111_0000,
            0b000001_11111_0000_000000_11110_0000,
        ])

        ψ = Wavefunction(copy(d))
        DMFT.excitation!(ψ, f, e, 3)
        @test Set(keys(ψ)) == Set([
            0b000000_11111_0000_000000_11111_0000,
            0b000000_11111_1111_000000_11111_1111,
            0b000000_11111_0000_000000_11110_0000,
            0b000000_11111_0000_000001_11111_0000,
            0b000000_11110_0000_000000_11111_0000,
            0b000001_11111_0000_000000_11111_0000,
            0b000000_11111_0000_000000_11100_0000,
            0b000000_11111_0000_000011_11111_0000,
            0b000000_11100_0000_000000_11111_0000,
            0b000011_11111_0000_000000_11111_0000,
            0b000000_11111_0000_000001_11110_0000,
            0b000001_11110_0000_000000_11111_0000,
            0b000000_11110_0000_000000_11110_0000,
            0b000001_11111_0000_000001_11111_0000,
            0b000001_11111_0000_000000_11110_0000,
            0b000000_11111_0000_000000_11000_0000,
            0b000000_11111_0000_000111_11111_0000,
            0b000000_11000_0000_000000_11111_0000,
            0b000111_11111_0000_000000_11111_0000,
        ])

        ψ = Wavefunction(copy(d))
        DMFT.excitation!(ψ, f, e, 4)
        @test Set(keys(ψ)) == Set([
            0b000000_11111_0000_000000_11111_0000,
            0b000000_11111_1111_000000_11111_1111,
            0b000000_11111_0000_000000_11110_0000,
            0b000000_11111_0000_000001_11111_0000,
            0b000000_11110_0000_000000_11111_0000,
            0b000001_11111_0000_000000_11111_0000,
            0b000000_11111_0000_000000_11100_0000,
            0b000000_11111_0000_000011_11111_0000,
            0b000000_11100_0000_000000_11111_0000,
            0b000011_11111_0000_000000_11111_0000,
            0b000000_11111_0000_000001_11110_0000,
            0b000001_11110_0000_000000_11111_0000,
            0b000000_11110_0000_000000_11110_0000,
            0b000001_11111_0000_000001_11111_0000,
            0b000001_11111_0000_000000_11110_0000,
            0b000000_11111_0000_000000_11000_0000,
            0b000000_11111_0000_000111_11111_0000,
            0b000000_11000_0000_000000_11111_0000,
            0b000111_11111_0000_000000_11111_0000,
        ])

        ψ = Wavefunction(copy(d))
        @test_throws ArgumentError DMFT.excitation!(ψ, f, e, -1)
    end # excitation!

    @testset "slater_start" begin
        @test slater_start(UInt64, 0b0000, 1, 2, 3, 4) ===
            UInt64(0b00000_111_00_1_00_0000_111_00_1_00)
        @test slater_start(BigMask{2,UInt64}, 0b1001, 10, 11, 12, 13) ===
            BigMask{2,UInt64}((
            0b0000_1111111111_10_0000000000000_111111111111_00000000000_1111111111_01,
            0b0000000000000_111111111111_0000000,
        ))

        # wrong arguments
        # 31 + 2 (bi) = 33 bits which is too much
        @test_throws ArgumentError slater_start(UInt64, zero(UInt8), 31, 0, 0, 0)
        @test_throws ArgumentError slater_start(UInt64, zero(UInt8), 0, 31, 0, 0)
        @test_throws ArgumentError slater_start(UInt64, zero(UInt8), 0, 0, 31, 0)
        @test_throws ArgumentError slater_start(UInt64, zero(UInt8), 0, 0, 0, 31)
        # negative values
        @test_throws ArgumentError slater_start(UInt64, zero(UInt8), -1, 0, 0, 0)
        @test_throws ArgumentError slater_start(UInt64, zero(UInt8), 0, -1, 0, 0)
        @test_throws ArgumentError slater_start(UInt64, zero(UInt8), 0, 0, -1, 0)
        @test_throws ArgumentError slater_start(UInt64, zero(UInt8), 0, 0, 0, -1)
    end # slater_start

    @testset "colorprint" begin
        s = sprint(DMFT.colorprint, zero(UInt64), 1, 2, 3, 4)
        @test s == bitstring(zero(UInt64))
        @test_throws ArgumentError DMFT.colorprint(zero(UInt64), 0, 0, 0, 31)
        @test_throws ArgumentError DMFT.colorprint(zero(UInt64), -1, 0, 0, 0)
        @test_throws ArgumentError DMFT.colorprint(zero(UInt64), 0, -1, 0, 0)
        @test_throws ArgumentError DMFT.colorprint(zero(UInt64), 0, 0, -1, 0)
        @test_throws ArgumentError DMFT.colorprint(zero(UInt64), 0, 0, 0, -1)
    end # colorprint
end # mask
