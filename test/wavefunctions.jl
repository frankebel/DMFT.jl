using DMFT
using Fermions
using Fermions.Wavefunctions
using LinearAlgebra
using Test

@testset "wavefunctions" begin
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
        d = Dict(UInt64(0b00_1_01_00_1_10) => copy(v), UInt64(0b00_1_10_00_1_01) => copy(v))
        ϕ = CIWavefunction(d, 5, 3, 4, 0)
        @test ψ == ϕ

        # e = 1
        ψ = starting_CIWavefunction(Dict{UInt64,Float64}, 1, 2, 3, 4, 1)
        v = zeros(1 + 2 * (3 + 4))
        v[1] = 1 / sqrt(2)
        d = Dict(UInt64(0b00_1_01_00_1_10) => copy(v), UInt64(0b00_1_10_00_1_01) => copy(v))
        ϕ = CIWavefunction(d, 5, 3, 4, 1)
        @test ψ == ϕ

        # e = 2
        ψ = starting_CIWavefunction(Dict{UInt64,Float64}, 1, 2, 3, 4, 2)
        v = zeros(1 + 14 + 14 * 13 ÷ 2)
        v[1] = 1 / sqrt(2)
        d = Dict(UInt64(0b00_1_01_00_1_10) => copy(v), UInt64(0b00_1_10_00_1_01) => copy(v))
        ϕ = CIWavefunction(d, 5, 3, 4, 2)
        @test ψ == ϕ
    end # starting_CIWavefunction

    @testset "ground state" begin
        # parameters
        n_bath = 31
        U = 4.0
        μ = U / 2
        n_v_bit = 1
        n_c_bit = 1
        e = 2
        n_kryl = 10
        n_sites = 1 + n_bath

        # applicable to both methods
        Δ = get_hyb(n_bath)
        H_nat, n_occ = to_natural_orbitals(Array(Δ))
        n_bit, n_v_vector, n_c_vector = get_CI_parameters(n_sites, n_occ, n_c_bit, n_v_bit)

        @testset "Wavefunction" begin
            M = 2 * n_sites <= 64 ? UInt64 : BigMask{cld(2 * n_sites, 64),UInt64}
            fs = FockSpace(M, M, Orbitals(n_sites), FermionicSpin(1//2))
            n = occupations(fs)
            H_int = U * n[1, -1//2] * n[1, 1//2]
            H = natural_orbital_operator(H_nat, H_int, -μ, fs, n_occ, n_v_bit, n_c_bit)
            ϕ_start = starting_Wavefunction(
                Dict{M,Float64}, n_v_bit, n_c_bit, n_v_vector, n_c_vector
            )
            E0, ψ0 = DMFT.ground_state(H, ϕ_start, n_kryl)

            Hψ = H * ψ0
            H_avg = dot(ψ0, Hψ)
            H_sqr = dot(Hψ, Hψ)
            var_rel = H_sqr / H_avg^2 - 1
            @test abs(E0 / -41.338736133386504 - 1) < 1E-4
            @test abs(H_avg / E0 - 1) < 1E-14
            @test var_rel < 5E-8
        end # Wavefunction

        @testset "CIWavefunction" begin
            fs = FockSpace(Orbitals(n_bit), FermionicSpin(1//2))
            n = occupations(fs)
            H_int = U * n[1, -1//2] * n[1, 1//2]
            H = natural_orbital_ci_operator(
                H_nat, H_int, -μ, fs, n_occ, n_v_bit, n_c_bit, e
            )
            ψ_start = starting_CIWavefunction(
                Dict{UInt64,Float64}, n_v_bit, n_c_bit, n_v_vector, n_c_vector, e
            )
            _, _, states = lanczos_krylov(H, ψ_start, n_kryl)
            S = Matrix{Float64}(undef, n_kryl, n_kryl) # overlap matrix
            @inbounds for i in 1:n_kryl, j in 1:n_kryl
                S[i, j] = states[i] ⋅ states[j]
            end
            E0, ψ0 = DMFT.ground_state(H, ψ_start, n_kryl)

            Hψ = H * ψ0
            H_avg = dot(ψ0, Hψ)
            H_sqr = dot(Hψ, Hψ)
            var_rel = H_sqr / H_avg^2 - 1
            @test norm(S - I) < 1E-12 # S_ij = δ_ij
            @test abs(E0 / -41.33867543081087 - 1) < 1E-4
            @test abs(H_avg / E0 - 1) < 1E-14
            @test var_rel < 4E-8
        end # CIWavefunction
    end # ground state
end # wavefunctions
