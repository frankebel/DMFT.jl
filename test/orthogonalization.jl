using DMFT
using Fermions.Wavefunctions
using LinearAlgebra
using Test

@testset "orthogonalization" begin
    @testset "_orthonormalize_SVD" begin
        # Matrix{ComplexF64}
        V = rand(ComplexF64, 10, 4)
        @inferred DMFT._orthonormalize_SVD(V)
        W, S_sqrt = DMFT._orthonormalize_SVD(V)
        @test norm(W' * W - I) < 100 * eps() # W^† W = 𝟙
        @test norm(V - W * S_sqrt) < 100 * eps() # V = W * S^{1/2}
        @test ishermitian(S_sqrt)

        # CIWavefunction
        v1 = CIWavefunction(Dict(zero(UInt8) => rand(5), one(UInt8) => rand(5)), 4, 1, 1, 1)
        v2 = CIWavefunction(Dict(zero(UInt8) => rand(5)), 4, 1, 1, 1)
        V = [v1 v2]
        @inferred DMFT._orthonormalize_SVD(V)
        W, S_sqrt = DMFT._orthonormalize_SVD(V)
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
        @test issymmetric(S_sqrt)
    end # _orthonormalize_SVD
end # orthogonalization
