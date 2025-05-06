using DMFT
using Test
using LinearAlgebra

@testset "Kramers-Kronig" begin
    # complex function in pole residue
    n_bath = 301
    a = collect(range(2, 6, n_bath ÷ 2))
    a = [-reverse(a); 0; a]
    b = fill(1 / sqrt(n_bath - 1), n_bath ÷ 2)
    b = [b; 0; b]
    G = Poles(a, b)
    # evaluate on grid
    ω = collect(-10:0.002:10)
    Z = ω .+ im * 0.02
    foo = G(Z)
    r = real(foo)
    i = imag(foo)

    # use Kramers-Kronig relations
    rKK = realKK(i, ω)
    iKK = imagKK(r, ω)

    # test real part
    @test maximum(abs.(r - rKK)) < 0.01
    @test norm(r - rKK) < 0.2
    # should be antisymmetric
    @test maximum(abs.(rKK + reverse(rKK))) < 0.01
    @test norm(rKK + reverse(rKK)) < 0.2

    # test imaginary part
    # bad approximation as real part decays ∝ 1/ω which is slow
    @test maximum(abs.(i - iKK)) < 0.4
    @test norm(i - iKK) < 10
    # should be symmetric
    @test maximum(abs.(iKK - reverse(iKK))) < 0.04
    @test norm(iKK - reverse(iKK)) < 0.2

    # Vector mismatch
    @test_throws DimensionMismatch realKK(i, rand(10))
    @test_throws DimensionMismatch imagKK(r, rand(10))
end # Kramers-Kronig
