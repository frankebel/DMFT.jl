using BlockBandedMatrices
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
        Δ = get_hyb(n_bath)
        H, E0, ψ0 = init_system(U, -μ, Δ, n_v_bit, n_c_bit, e, n_kryl_gs)
        # operators A, B
        fs = FockSpace(Orbitals(2 + n_v_bit + n_c_bit), FermionicSpin(1//2))
        c = annihilators(fs)
        # add electron
        A = c[1, -1//2]' # f_↓^†
        B = c[1, -1//2] * c[1, 1//2]' * c[1, 1//2] # f_↓ n_↑
        v1 = A * ψ0
        v2 = B' * ψ0
        V0 = [v1 v2]
        # Löwdin orthogonalization
        W, S_sqrt = orthogonalize_states(V0)
        # Block Lanczos
        a, b = block_lanczos(H, W, n_kryl)
        @test length(a) === n_kryl
        @test length(b) === n_kryl - 1

        # NOTE: change once compatibility is set to >= Julia 1.11
        # X = BlockTridiagonal(b, a, map(adjoint, b))
        X = BlockTridiagonal(b, a, Matrix.(adjoint.(b)))
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
        @test norm(E - E_ref) < 2E3 * eps()
    end # block_lanczos

    @testset "svd_orthogonalize!" begin
        v1 = CIWavefunction(Dict(zero(UInt8) => rand(5), one(UInt8) => rand(5)), 4, 1, 1, 1)
        v2 = CIWavefunction(Dict(zero(UInt8) => rand(5)), 4, 1, 1, 1)
        V = [v1 v2]
        SVD = zero(V)
        B = Matrix{Float64}(undef, 2, 2)
        @test DMFT.svd_orthogonalize!(V, SVD, B) === (V, SVD, B)
        @test ishermitian(B)
        # SVD^† SVD = 𝟙
        foo = Matrix{Float64}(undef, 2, 2)
        mul!(foo, SVD', SVD)
        @test norm(foo - I) < 4 * eps()
        # V = SVD B^{1/2}
        bar = zero(V)
        mul!(bar, SVD, B)
        for i in eachindex(V)
            @test norm(bar[i] - V[i]) < 4 * eps()
        end
    end # svd_orthogonalize!
end # block_lanczos
