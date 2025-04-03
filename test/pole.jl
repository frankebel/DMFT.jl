using DMFT
using Distributions
using LinearAlgebra
using Test

@testset "Pole" begin
    V = Vector{Float64}
    M = Matrix{Float64}
    @testset "constructor" begin
        a = sort(rand(10))
        bv = rand(10) # vector
        bm = rand(2, 10) # matrix

        # inner constructor
        P = Pole{V,V}(a, bv)
        @test P.a === a
        @test P.b === bv
        P = Pole{V,M}(a, bm)
        @test P.a === a
        @test P.b === bm

        # outer constructor
        P = Pole(a, bv)
        @test typeof(P) === Pole{V,V}
        @test_throws TypeError Pole(rand(ComplexF64, 10), bv)
        @test_throws DimensionMismatch Pole(a, rand(9))
        @test_throws DimensionMismatch Pole(a, rand(2, 9))

        # conversion of type
        a = collect(1:10)
        b = collect(11:20)
        P = Pole(a, b)
        P_new = Pole{Vector{Float64},Vector{Int}}(P)
        @test typeof(P_new) === Pole{Vector{Float64},Vector{Int}}
        @test P_new.a == P.a
        @test P_new.b == P.b

        @testset "blockdiagonal" begin
            A = [rand(2, 2) for _ in 1:5]
            B = [rand(2, 2) for _ in 1:4]
            E0 = rand(Float64)
            S_sqrt = rand(2, 2)
            P = DMFT._pole(A, B, E0, S_sqrt)
            @test typeof(P.a) == Vector{Float64}
            @test length(P.a) == 10
            @test typeof(P.b) == Matrix{Float64}
            @test size(P.b) == (2, 10)
        end # blockdiagonal
    end # constructor

    @testset "evaluation" begin
        @testset "Lorentzian" begin
            a = collect(1.0:10)
            # single point
            z = 10 + im
            # b::Vector
            b = collect(0.1:0.1:1)
            P = Pole(a, b)
            foo = P(z)
            @test typeof(P(z)) === ComplexF64
            @test abs(P(z) - sum(i -> (0.1 * i)^2 / (10 + im - i), 1:10)) < eps()
            # b::Matrix
            b = reshape(collect(0.1:0.1:2), (2, 10))
            P = Pole(a, b)
            foo = P(z)
            @test typeof(P(z)) === Matrix{ComplexF64}
            @test norm(
                P(z) - [
                    3.419109056634164-5.796080126589452im 3.669440321902302-6.127487634859227im
                    3.669440321902302-6.127487634859227im 3.9413975570979876-6.478614061451364im
                ],
            ) < eps()

            # grid
            z = rand(ComplexF64, 2)
            @test P(z) == [P(z[1]), P(z[2])]
        end # Lorentzian

        @testset "Gaussian" begin
            ω = 0.15
            σ = 0.04
            # b::Matrix
            a = [0.1, 0.2]
            b = reshape(collect(0.1:0.1:0.4), (2, 2))
            P = Pole(a, b)
            @test norm(
                P(ω, σ) - [
                    -1.5277637226549838-1.4345225621076145im -1.90970465331873-2.00833158695066im
                    -1.90970465331873-2.00833158695066im -2.2916455839824765-2.8690451242152295im
                ],
            ) < 10 * eps()

            # semicircular DOS
            G = greens_function_bethe_simple(3001)
            ω = collect(-3:0.01:3)
            σ = 0.01
            # constant broadening
            h = G(ω, σ)
            ex = π .* pdf.(Semicircle(1), ω) # exact solution
            @test norm(ex + imag(h)) < 0.2
            @test maximum(abs.(ex + imag(h))) < 0.12
            @test findmin(imag(h))[2] == cld(length(ω), 2) # symmetric
        end # Gaussian
    end # evaluate

    @testset "Array" begin
        a = collect(1:5)
        b = collect(6:10)
        Δ = Pole(a, b)
        m = Array(Δ)
        @test typeof(m) === Matrix{Int}
        @test m == [
            0 6 7 8 9 10
            6 1 0 0 0 0
            7 0 2 0 0 0
            8 0 0 3 0 0
            9 0 0 0 4 0
            10 0 0 0 0 5
        ]
    end # Array
end # Pole

@testset "Kramers-Kronig" begin
    # complex function in pole residue
    n_bath = 301
    a = collect(range(2, 6, n_bath ÷ 2))
    a = [-reverse(a); 0; a]
    b = fill(1 / sqrt(n_bath - 1), n_bath ÷ 2)
    b = [b; 0; b]
    G = Pole(a, b)
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
