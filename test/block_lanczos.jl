using DMFT
using Fermions
using Fermions.Wavefunctions
using LinearAlgebra
using Test

@testset "block_lanczos" begin
    @testset "block_lanczos" begin
        # parameters
        n_bath = 11
        U = 4.0
        μ = U / 2
        n_v_bit = 1
        n_c_bit = 1
        e = 2
        n_kryl = 5
        n_kryl_gs = 20
        # initial system
        Δ = hybridization_function_bethe_simple(n_bath, 2)
        fs = FockSpace(Orbitals(2 + n_v_bit + n_c_bit), FermionicSpin(1//2))
        c = annihilators(fs)
        n = occupations(fs)
        H_int = U * n[1, 1//2] * n[1, -1//2]
        d_dag = c[1, -1//2]' # d_↓^†
        q_dag = H_int * d_dag - d_dag * H_int  # q_↓^† = [H_int, d^†]

        H, E0, ψ0 = init_system(Δ, H_int, -μ, n_v_bit, n_c_bit, e, n_kryl_gs)
        v1 = d_dag * ψ0
        v2 = q_dag * ψ0
        V0 = [v1 v2]
        # Löwdin orthogonalization
        W, S_sqrt = orthogonalize_states(V0)
        # Block Lanczos
        a, b = DMFT.block_lanczos(H, W, n_kryl)
        @test length(a) === n_kryl
        @test length(b) === n_kryl - 1
        @test all(ishermitian, a)
        @test all(ishermitian, b)

        X = zeros(2 * n_kryl, 2 * n_kryl)
        for i in 1:(n_kryl - 1)
            X[(2 * i - 1):(2 * i), (2 * i - 1 + 2):(2 * i + 2)] = b[i] # upper diagonal, don't need adjoint
            X[(2 * i - 1):(2 * i), (2 * i - 1):(2 * i)] = a[i] # main diagonal
            X[(2 * i + 1):(2 * i + 2), (2 * i - 1):(2 * i)] = b[i] # lower diagonal
        end
        X[(end - 1):end, (end - 1):end] = a[end] # last element
        E = eigvals(X)
        E_ref = [
            -15.678942694187539,
            -15.295686009018686,
            -14.964415736054807,
            -14.76013760998127,
            -14.267760301388584,
            -14.079691680430527,
            -13.417532651434934,
            -12.538689290677826,
            -11.218209037339488,
            -9.818357326067973,
        ]
        @test norm(E - E_ref) < 3E3 * eps()
    end # block_lanczos

    @testset "svd_orthogonalize!" begin
        v1 = CIWavefunction(Dict(zero(UInt8) => rand(5), one(UInt8) => rand(5)), 4, 1, 1, 1)
        v2 = CIWavefunction(Dict(zero(UInt8) => rand(5)), 4, 1, 1, 1)
        V = [v1 v2]
        SVD = zero(V)
        Adj = Vector{eltype(V)}(undef, 2)
        B = Matrix{Float64}(undef, 2, 2)
        M1 = similar(B)
        M2 = similar(B)
        M3 = similar(B)
        DMFT._svd_orthogonalize!(B, V, SVD, Adj, M1, M2, M3)
        @test ishermitian(B)
        # SVD^† SVD = 𝟙
        foo = Matrix{Float64}(undef, 2, 2)
        mul!(foo, SVD', SVD)
        @test norm(foo - I) < 8 * eps()
        # V = SVD B^{1/2}
        bar = zero(V)
        mul!(bar, SVD, B)
        for i in eachindex(V)
            @test norm(bar[i] - V[i]) < 10 * eps()
        end
    end # svd_orthogonalize!
end # block_lanczos
