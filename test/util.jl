using DMFT
using Fermions
using Fermions.Wavefunctions
using Test

@testset "util" begin
    @testset "mul_excitation" begin
        f, e = mask_fe(UInt64, 2, 3, 4) # bi in bit component
        fs = FockSpace(Orbitals(9), FermionicSpin(1//2)) # 2+3+4 sites
        n = occupations(fs)
        H = Operator(fs)
        for i in 1:9
            H += Float64(i) * n[i, -1//2]
        end
        for i in 10:18
            H += Float64(i) * n[mod1(i, 9), 1//2]
        end
        d = Dict{UInt64,Float64}(
            0b0000_111_00_0000_111_00 => 1.0,
            0b0000_111_00_0000_111_01 => 1.0,
            0b0000_111_00_0000_110_00 => 1.0,
            0b0000_111_00_0001_111_00 => 1.0,
            0b0001_011_00_0000_111_00 => 1.0,
        )
        ψ = Wavefunction(d)

        ϕ = mul_excitation(H, ψ, f, e, 0)
        @test ϕ == Wavefunction(
            Dict{UInt64,Float64}(
                0b0000_111_00_0000_111_00 => 51.0, 0b0000_111_00_0000_111_01 => 52.0
            ),
        )

        ϕ = mul_excitation(H, ψ, f, e, 1)
        @test ϕ == Wavefunction(
            Dict{UInt64,Float64}(
                0b0000_111_00_0000_111_00 => 51.0,
                0b0000_111_00_0000_111_01 => 52.0,
                0b0000_111_00_0000_110_00 => 48.0,
                0b0000_111_00_0001_111_00 => 57.0,
            ),
        )

        ϕ = mul_excitation(H, ψ, f, e, 2)
        @test ϕ == Wavefunction(
            Dict{UInt64,Float64}(
                0b0000_111_00_0000_111_00 => 51.0,
                0b0000_111_00_0000_111_01 => 52.0,
                0b0000_111_00_0000_110_00 => 48.0,
                0b0000_111_00_0001_111_00 => 57.0,
                0b0001_011_00_0000_111_00 => 52.0,
            ),
        )
    end # mul_excitation

    @testset "diffkeys" begin
        ϕ1 = Wavefunction(Dict(zero(UInt64) => 1, one(UInt64) => 1))
        ϕ2 = Wavefunction(Dict(zero(UInt64) => 1))
        @test DMFT.diffkeys(ϕ1, ϕ2) == Set(one(UInt64))
        @test DMFT.diffkeys(ϕ2, ϕ1) == Set{UInt64}()
    end # diffkeys

    @testset "starting wave function" begin
        @testset "starting_Wavefunction" begin
            ψ = starting_Wavefunction(Dict{UInt64,Float64}, 1, 2, 3, 4)
            d = Dict(
                UInt64(0b0000_111_00_1_01_0000_111_00_1_10) => 1 / sqrt(2),
                UInt64(0b0000_111_00_1_10_0000_111_00_1_01) => 1 / sqrt(2),
            )
            ϕ = Wavefunction(d)
            @test ψ == ϕ
        end # starting_Wavefunction

        @testset "starting_CIWavefunction" begin
            # e = 0
            ψ = starting_CIWavefunction(Dict{UInt64,Float64}, 1, 2, 3, 4, 0)
            v = [1 / sqrt(2)]
            d = Dict(
                UInt64(0b00_1_01_00_1_10) => copy(v), UInt64(0b00_1_10_00_1_01) => copy(v)
            )
            ϕ = CIWavefunction(d, 5, 3, 4, 0)
            @test ψ == ϕ

            # e = 1
            ψ = starting_CIWavefunction(Dict{UInt64,Float64}, 1, 2, 3, 4, 1)
            v = zeros(1 + 2 * (3 + 4))
            v[1] = 1 / sqrt(2)
            d = Dict(
                UInt64(0b00_1_01_00_1_10) => copy(v), UInt64(0b00_1_10_00_1_01) => copy(v)
            )
            ϕ = CIWavefunction(d, 5, 3, 4, 1)
            @test ψ == ϕ

            # e = 2
            ψ = starting_CIWavefunction(Dict{UInt64,Float64}, 1, 2, 3, 4, 2)
            v = zeros(1 + 14 + 14 * 13 ÷ 2)
            v[1] = 1 / sqrt(2)
            d = Dict(
                UInt64(0b00_1_01_00_1_10) => copy(v), UInt64(0b00_1_10_00_1_01) => copy(v)
            )
            ϕ = CIWavefunction(d, 5, 3, 4, 2)
            @test ψ == ϕ
        end # starting_CIWavefunction
    end # starting wave function

    @testset "get_CI_parameters" begin
        @test get_CI_parameters(10, 5, 1, 1) == (4, 3, 3)
        @test get_CI_parameters(10, 6, 1, 1) == (4, 4, 2)
        @test get_CI_parameters(10, 4, 1, 1) == (4, 2, 4)
        @test get_CI_parameters(10, 5, 2, 1) == (5, 2, 3)
        @test get_CI_parameters(10, 5, 1, 2) == (5, 3, 2)
        @test get_CI_parameters(10, 6, 2, 1) == (5, 3, 2)
        @test get_CI_parameters(10, 4, 1, 2) == (5, 2, 3)
        # non-sensical input values still work
        @test get_CI_parameters(10, 10, 1, 1) == (4, 8, -2)
        @test get_CI_parameters(10, 0, 1, 1) == (4, -2, 8)
    end # get_CI_parameters
end # util
