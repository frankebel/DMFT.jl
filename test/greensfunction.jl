using DMFT
using Distributions
using LinearAlgebra
using Test

@testset "Green's function" begin
    @testset "constructor" begin
        V = Vector{Float64}
        a = sort(rand(10))
        b = rand(10)

        # inner constructor
        G = Greensfunction{Float64,V}(a, b)
        @test G.a === a
        @test G.b === b
        @test_throws DimensionMismatch Greensfunction{Float64,V}(a, rand(9))

        # outer constructor
        G = Greensfunction(a, b)
        @test typeof(G) === Greensfunction{Float64,V}

        G_new = Greensfunction{Float64,V}(G)
        @test typeof(G_new) === typeof(G)
        @test G_new !== G
        @test G_new.a == G.a
        @test G_new.b == G.b

        b = rand(2, 10)
        G = Greensfunction(a, b)
        @test G.a === a
        @test G.b === b
        @test_throws DimensionMismatch Greensfunction(a, rand(10, 2))

        @testset "blockdiagonal" begin
            A = [rand(2, 2) for _ in 1:5]
            B = [rand(2, 2) for _ in 1:4]
            E0 = rand(Float64)
            S_sqrt = rand(2, 2)
            G = Greensfunction(A, B, E0, S_sqrt)
            @test typeof(G.a) == Vector{Float64}
            @test length(G.a) == 10
            @test typeof(G.b) == Matrix{Float64}
            @test size(G.b) == (2, 10)
        end # blockdiagonal
    end # constructor

    @testset "evaluate" begin
        @testset "Lorentzian" begin
            # single point
            z = rand(ComplexF64)
            # b::Vector
            a = sort!(rand(10))
            b = rand(10)
            G = Greensfunction(a, b)
            @test G(z) === sum(map((i, j) -> abs2(j) / (z - i), a, b))
            # b::Matrix
            b = rand(2, 10)
            G = Greensfunction(a, b)
            foo = zeros(ComplexF64, 2, 2)
            for i in eachindex(a)
                v = b[:, i]
                foo += v * v' ./ (z - a[i])
            end
            @test foo == G(z)

            # grid
            z = rand(ComplexF64, 2)
            # multiple poles
            a = sort!(rand(10))
            b = rand(10)
            G = Greensfunction(a, b)
            @test G(z) == [G(z[1]), G(z[2])]
        end # Lorentzian

        @testset "Gaussian" begin
            ω = 0.5
            σ = 0.04
            # b::Matrix
            a = sort!(rand(2))
            b = rand(2, 2)
            G = Greensfunction(a, b)
            w1 =
                abs2(b[1, 1]) * pdf(Normal(a[1], σ), ω) +
                abs2(b[1, 2]) * pdf(Normal(a[2], σ), ω)
            w2 =
                b[1, 1] * b[2, 1] * pdf(Normal(a[1], σ), ω) +
                b[1, 2] * b[2, 2] * pdf(Normal(a[2], σ), ω)
            w3 = w2 # symmetry
            w4 =
                abs2(b[2, 1]) * pdf(Normal(a[1], σ), ω) +
                abs2(b[2, 2]) * pdf(Normal(a[2], σ), ω)
            @test imag(G(ω, σ)) == -π * [w1 w2; w3 w4]

            # semicircular DOS
            Δ = get_hyb(301)
            ω = collect(-3:0.01:3)
            σ = 0.04
            # constant broadening
            h = Δ(ω, σ)
            ex = π .* pdf.(Semicircle(2), ω) # exact solution
            @test norm(ex + imag(h)) < 0.2
            @test maximum(abs.(ex + imag(h))) < 0.1
            @test findmin(imag(h))[2] == cld(length(ω), 2) # symmetric
            # variable broadening
            hv = Δ(ω, fill(σ, length(ω)))
            @test h == hv
            foo = map(i -> abs(i) < 2.0 ? 0.04 : 0.08, ω)
            hv = Δ(ω, foo)
            @test h != hv
            @test imag(h) >= imag(hv) # bigger broadening → pdf closer to zero
            @test_throws ArgumentError Δ(ω, [σ])
            # b::Matrix
            a = sort!(rand(2))
            b = rand(2, 2)
            G = Greensfunction(a, b)
            @test @inferred(G(ω, fill(σ, length(ω)))) isa Vector{Matrix{ComplexF64}}
        end # Gaussian
    end # evaluate

    @testset "IO" begin
        # b::Vector
        # write
        a = sort(rand(10))
        b = rand(10)
        G = Greensfunction(a, b)
        @test write("test.h5", G) === nothing
        # read
        foo = Greensfunction{Float64,Vector{Float64}}("test.h5")
        @test typeof(foo) == Greensfunction{Float64,Vector{Float64}}
        @test foo.a == G.a
        @test foo.b == G.b

        # b::Matrix
        # write
        a = sort(rand(10))
        b = rand(10, 10)
        G = Greensfunction(a, b)
        @test write("test.h5", G) === nothing
        # read
        foo = Greensfunction{Float64,Matrix{Float64}}("test.h5")
        @test typeof(foo) == Greensfunction{Float64,Matrix{Float64}}
        @test foo.a == G.a
        @test foo.b == G.b

        rm("test.h5")
    end # IO
end # Green's function

@testset "hybridization" begin
    A = Float64
    B = Vector{Float64}

    @testset "get_hyb" begin
        Δ = get_hyb(101)
        @test typeof(Δ) === Greensfunction{A,B}
        @test length(Δ.a) === 101
        @test length(Δ.b) === 101
        @test sum(abs2.(Δ.b)) ≈ 1.0 rtol = 10 * eps()
        @test Δ.a[51] ≈ 0 atol = 10 * eps()
        @test norm(Δ.a + reverse(Δ.a)) < 50 * eps()
        @test norm(abs2.(Δ.b) - reverse(abs2.(Δ.b))) < 600 * eps()
        Δ = get_hyb(100)
        @test typeof(Δ) === Greensfunction{A,B}
        @test length(Δ.a) === 100
        @test length(Δ.b) === 100
        @test sum(abs2.(Δ.b)) ≈ 1.0 rtol = 10 * eps()
        @test norm(Δ.a + reverse(Δ.a)) < 100 * eps()
        @test norm(abs2.(Δ.b) - reverse(abs2.(Δ.b))) < 600 * eps()
    end # get_hyb

    @testset "get_hyb_equal" begin
        @test_throws ArgumentError get_hyb_equal(2)
        Δ = get_hyb_equal(101)
        @test typeof(Δ) === Greensfunction{A,B}
        @test length(Δ.a) === 101
        @test length(Δ.b) === 101
        @test all(i -> i === 1 / sqrt(101), Δ.b)
        @test norm(Δ.a + reverse(Δ.a)) === 0.0
        @test issorted(Δ.a)
    end # get_hyb_equal

    @testset "Array" begin
        a = collect(1:5)
        b = collect(6:10)
        Δ = Greensfunction(a, b)
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

    @testset "Kramers-Kronig" begin
        # complex function in pole residue
        n_bath = 301
        a = collect(range(2, 6, n_bath ÷ 2))
        a = [-reverse(a); 0; a]
        b = fill(1 / sqrt(n_bath - 1), n_bath ÷ 2)
        b = [b; 0; b]
        G = Greensfunction(a, b)
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
        @test_throws ArgumentError realKK(i, rand(10))
        @test_throws ArgumentError imagKK(r, rand(10))
    end # Kramers-Kronig
end # hybridization
