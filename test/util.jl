using DMFT
using Distributions
using Fermions
using Fermions.Lanczos
using Fermions.Wavefunctions
using LinearAlgebra
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
            _, _, states = lanczos_with_states(H, ψ_start, n_kryl)
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

    @testset "init_system" begin
        # parameters
        n_bath = 31
        U = 4.0
        μ = U / 2
        n_v_bit = 1
        n_c_bit = 1
        e = 2
        n_kryl = 10
        Δ = get_hyb(n_bath)
        fs = FockSpace(Orbitals(2 + n_v_bit + n_c_bit), FermionicSpin(1//2))
        n = occupations(fs)
        H_int = U * n[1, -1//2] * n[1, 1//2]
        H, E0, ψ0 = init_system(Δ, H_int, -μ, n_v_bit, n_c_bit, e, n_kryl)
        Hψ = H * ψ0
        H_avg = dot(ψ0, Hψ)
        H_sqr = dot(Hψ, Hψ)
        var_rel = H_sqr / H_avg^2 - 1
        @test H isa CIOperator
        @test abs(E0 / -41.33867543081087 - 1) < 1E-4
        @test abs(H_avg / E0 - 1) < 1E-14
        @test var_rel < 4E-8
    end # init system

    @testset "orthogonalize_states" begin
        v1 = CIWavefunction(Dict(zero(UInt8) => rand(5), one(UInt8) => rand(5)), 4, 1, 1, 1)
        v2 = CIWavefunction(Dict(zero(UInt8) => rand(5)), 4, 1, 1, 1)
        V = [v1 v2]
        W, S_sqrt = orthogonalize_states(V)
        # W^† W = 𝟙
        foo = Matrix{Float64}(undef, 2, 2)
        mul!(foo, W', W)
        @test norm(foo - I) < 8 * eps()
        # V = W S^{1/2}
        bar = zero(V)
        mul!(bar, W, S_sqrt)
        for i in eachindex(V)
            @test norm(bar[i] - V[i]) < 8 * eps()
        end
    end # orthogonalize_states

    @testset "vector IO" begin
        v = rand(10)
        @test write_vector("test.h5", v) === nothing
        foo = read_vector(Float64, "test.h5")
        @test typeof(foo) === Vector{Float64}
        @test foo == v
        rm("test.h5")
    end # vector IO

    @testset "matrix IO" begin
        m = rand(10, 10)
        @test write_matrix("test.h5", m) === nothing
        foo = read_matrix(Float64, "test.h5")
        @test typeof(foo) === Matrix{Float64}
        @test foo == m
        rm("test.h5")
    end # matrix IO

    @testset "η_gaussian" begin
        η_0 = 0.01
        η_∞ = 0.04
        w = collect(-10:0.08:10)
        η1 = η_gaussian(η_0, η_∞, 1.0, w)
        @test typeof(η1) === typeof(w)
        @test length(η1) === length(w)
        @test norm(η1 - reverse(η1)) === 0.0 # symmetric
        @test η1[126] ≈ η_0 rtol = 5 * eps() # w = 0
        @test η1[1] == η_∞
        # smaller broadening
        η2 = η_gaussian(η_0, η_∞, 0.5, w)
        @test all(η2 .>= η1)
    end # η_gaussian

    @testset "Bethe lattice Green's function" begin
        # real frequencies
        ω = collect(-10:0.002:10)
        g = G_bethe(ω)
        @test norm(-imag(g) / π - pdf.(Semicircle(2.0), ω)) < 10 * eps()
        @test norm(real(g) + reverse(real(g))) < 10 * eps()
        # D = 1.0
        g = G_bethe(ω, 1.0)
        @test norm(-imag(g) / π - pdf.(Semicircle(1.0), ω)) < 10 * eps()

        # complex frequencies
        @test_throws ArgumentError G_bethe(ω .- 0.04im) # negative imaginary part
        g = G_bethe(ω .+ 0.04im)
        # (anti)symmetric
        @test norm(real(g) + reverse(real(g))) == 0
        @test norm(imag(g) - reverse(imag(g))) == 0
        # ω = 0.0 is finickey
        @test argmax(-imag(g)) === 5001
        @test real(g[5001]) === 0.0
    end # Bethe lattice Green's function
end # util
