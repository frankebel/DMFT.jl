using RAS_DMFT
using LinearAlgebra
using Test

@testset "sytrd!" begin
    d = 1000
    M = rand(d, d)
    hermitianpart!(M)
    A = copy(M)
    d, e, τ = sytrd!('L', A)
    T = SymTridiagonal(d, e)
    orgtr!('L', A, τ)
    @test norm(A' * M * A - T) < 3.0e-12
end
