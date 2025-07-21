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

    @testset "_orthonormalize_GramSchmidt!" begin
        V = [0 3 1; 0 0 1; 100*eps() 8 0]
        @test DMFT._orthonormalize_GramSchmidt!(V) === V
        @test view(V, :, 1) == zeros(3)
        @test view(V, :, 2) == 1 / sqrt(73) * [3, 0, 8]
        v3 = [64, 73, -24] / 73
        normalize!(v3)
        @test norm(view(V, :, 3) - v3) < eps()
        # applying again has small changes
        foo = copy(V)
        @test DMFT._orthonormalize_GramSchmidt!(V) != foo
        @test V' * V == Diagonal([0, 1, 1])
        # applying again should keep it equal
        foo = copy(V)
        @test DMFT._orthonormalize_GramSchmidt!(V) == foo
    end # _orthonormalize_GramSchmidt!

    @testset "_orthogonalize_states!" begin
        Q_new = rand(ComplexF64, 8, 4)
        Q_old = rand(ComplexF64, 8, 4)
        Q_old, _ = DMFT._orthonormalize_SVD(Q_old)
        DMFT._orthonormalize_GramSchmidt!(Q_old)
        DMFT._orthonormalize_GramSchmidt!(Q_old) # twice because unstable
        M1 = Matrix{ComplexF64}(undef, 4, 4)
        @test DMFT._orthogonalize_states!(M1, Q_new, Q_old) === Q_new
        # overlap to previous state
        foo = norm(Q_old' * Q_new)
        @test foo < 10 * eps()
        # orthogonalize again
        DMFT._orthogonalize_states!(M1, Q_new, Q_old)
        bar = norm(Q_old' * Q_new)
        @test bar <= foo
        @test bar < eps()
    end # _orthogonalize_states!
end # orthogonalization
