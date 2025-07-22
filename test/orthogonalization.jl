using DMFT
using Fermions.Wavefunctions
using LinearAlgebra
using Test

@testset "orthogonalization" begin
    @testset "_orthonormalize_SVD" begin
        # in-place
        q1 = CIWavefunction(Dict(zero(UInt8) => rand(5), one(UInt8) => rand(5)), 4, 1, 1, 1)
        q2 = CIWavefunction(Dict(zero(UInt8) => rand(5)), 4, 1, 1, 1)
        Q = [q1 q2]
        Q_new = similar(Q)
        S_sqrt = Matrix{Float64}(undef, 2, 2)
        V1 = Vector{Float64}(undef, 2)
        M1 = similar(S_sqrt)
        DMFT._orthonormalize_SVD!(V1, M1, S_sqrt, Q_new, Q)
        @test ishermitian(S_sqrt)
        # Q_int^† Q_int = 𝟙
        foo = Matrix{Float64}(undef, 2, 2)
        mul!(foo, Q_new', Q_new)
        @test norm(foo - I) < 8 * eps()
        # V = Q_int B
        bar = similar(Q)
        mul!(bar, Q_new, S_sqrt) # Q = Q_new S^{1/2}
        for i in axes(Q, 2)
            @test norm(bar[i] - Q[i]) < 10 * eps()
        end

        # Matrix{ComplexF64}
        Q = rand(ComplexF64, 10, 4)
        @inferred DMFT._orthonormalize_SVD(Q)
        Q_new, S_sqrt = DMFT._orthonormalize_SVD(Q)
        @test norm(Q_new' * Q_new - I) < 100 * eps() # Q_new^† Q_new = 𝟙
        @test norm(Q - Q_new * S_sqrt) < 100 * eps() # Q = Q_new * S^{1/2}
        @test ishermitian(S_sqrt)

        # CIWavefunction
        q1 = CIWavefunction(Dict(zero(UInt8) => rand(5), one(UInt8) => rand(5)), 4, 1, 1, 1)
        v2 = CIWavefunction(Dict(zero(UInt8) => rand(5)), 4, 1, 1, 1)
        Q = [q1 v2]
        @inferred DMFT._orthonormalize_SVD(Q)
        Q_new, S_sqrt = DMFT._orthonormalize_SVD(Q)
        # Q_new^† Q_new = 𝟙
        foo = Matrix{Float64}(undef, 2, 2)
        mul!(foo, Q_new', Q_new)
        @test norm(foo - I) < 8 * eps()
        # Q = Q_new S^{1/2}
        bar = similar(Q)
        mul!(bar, Q_new, S_sqrt)
        for i in eachindex(Q)
            @test norm(bar[i] - Q[i]) < 8 * eps()
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

        # no allocations
        V = [0 3 1; 0 0 1; 100*eps() 8 0]
        @test iszero(@allocated(DMFT._orthonormalize_GramSchmidt!(V)))
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

        # no allocations
        Q_new = rand(ComplexF64, 8, 4)
        @test iszero(@allocated(DMFT._orthogonalize_states!(M1, Q_new, Q_old)))
    end # _orthogonalize_states!
end # orthogonalization
