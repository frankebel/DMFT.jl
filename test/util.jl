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

    @testset "starting_wf" begin
        ψ = starting_wf(Dict{UInt64,Float64}, 1, 2, 3, 4)
        d = Dict(
            UInt64(0b0000_111_00_1_01_0000_111_00_1_10) => 1 / sqrt(2),
            UInt64(0b0000_111_00_1_10_0000_111_00_1_01) => 1 / sqrt(2),
        )
        ϕ = Wavefunction(d)
        @test ψ == ϕ
    end # starting_wf
end # util
