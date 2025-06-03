using DMFT
using LinearAlgebra
using Test

@testset "sytrd!" begin
    d = 1000
    M = rand(d, d)
    M = M + M'
    A = copy(M)
    d, e, τ = sytrd!('L', A)
    T = SymTridiagonal(d, e)
    orgtr!('L', A, τ)
    @test norm(A' * M * A - T) < 3e-12
end
