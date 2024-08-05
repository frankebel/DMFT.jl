using DMFT
using LinearAlgebra
using Test

@testset "Green's function" begin
    @testset "constructor" begin
        V = Vector{Float64}
        a = sort(rand(10))
        b = rand(10)

        # inner constructor
        G = Greensfunction{V,V}(a, b)
        @test G.a === a
        @test G.b === b
        @test_throws ArgumentError Greensfunction{V,V}(a, rand(9))

        # outer constructor
        G = Greensfunction(a, b)
        @test typeof(G) === Greensfunction{V,V}
    end # constructor

    @testset "evaluate" begin
        @testset "single point" begin
            z = rand(ComplexF64)

            # single pole
            a = rand(1)
            b = rand(1)
            G = Greensfunction(a, b)
            @test G(z) === abs2(only(b)) / (z - only(a))

            # multiple poles
            a = sort(rand(10))
            b = rand(10)
            G = Greensfunction(a, b)
            @test G(z) === sum(map((i, j) -> abs2(j) / (z - i), a, b))
        end # single point

        @testset "multiple points" begin
            z = rand(ComplexF64, 2)

            # single pole
            a = rand(1)
            b = rand(1)
            G = Greensfunction(a, b)
            @test G(z) ==
                [abs2(only(b)) / (z[1] - only(a)), abs2(only(b)) / (z[2] - only(a))]

            # multiple poles
            a = sort(rand(10))
            b = rand(10)
            G = Greensfunction(a, b)
            @test G(z) == [G(z[1]), G(z[2])]
        end # multiple points
    end # evaluate

    @testset "IO" begin
        # write
        a = sort(rand(10))
        b = rand(10)
        G = Greensfunction(a, b)
        @test write("test.h5", G) === nothing

        # read
        V = Vector{Float64}
        foo = Greensfunction{V,V}("test.h5")
        @test typeof(foo) == Greensfunction{V,V}
        @test foo.a == G.a
        @test foo.b == G.b
        # fallback default type
        foo = Greensfunction("test.h5")
        @test typeof(foo) == Greensfunction{V,V}
        @test foo.a == G.a
        @test foo.b == G.b

        rm("test.h5")
    end # IO
end # Green's function

@testset "Hybridization function" begin
    a_p = sort(rand(10))
    a_n = sort(-rand(10))
    b_p = rand(10)
    b_n = rand(10)
    V0 = rand()
    G_p = Greensfunction(a_p, b_p)
    G_n = Greensfunction(a_n, b_n)

    @testset "constructor" begin
        Δ = Hybridizationfunction(V0, G_p, G_n)
        V = Vector{Float64}
        @test typeof(Δ) == Hybridizationfunction{Float64,Greensfunction{V,V}}
        @test Δ.V0 === V0
        @test Δ.pos === G_p
        @test Δ.neg === G_n
    end # constructor

    @testset "evaluate" begin
        @testset "single point" begin
            z = rand(ComplexF64)
            Δ = Hybridizationfunction(V0, G_p, G_n)

            @test Δ(z) === G_p(z) + G_n(z) + abs2(V0) / z
        end # single point

        @testset "multiple points" begin
            z = rand(ComplexF64, 2)
            Δ = Hybridizationfunction(V0, G_p, G_n)

            @test Δ(z) == [Δ(z[1]), Δ(z[2])]
        end # multiple points
    end # evaluate

    @testset "IO" begin
        # write
        Δ = Hybridizationfunction(V0, G_p, G_n)
        @test write("test.h5", Δ) === nothing

        # read
        M = Float64
        V = Vector{Float64}
        G = Greensfunction{V,V}
        foo = Hybridizationfunction{M,G}("test.h5")
        @test typeof(foo) == Hybridizationfunction{M,G}
        @test foo.V0 === V0
        @test foo.pos.a == a_p
        @test foo.pos.b == b_p
        @test foo.neg.a == a_n
        @test foo.neg.b == b_n
        # # fallback default type
        foo = Hybridizationfunction("test.h5")
        @test typeof(foo) == Hybridizationfunction{M,G}
        @test foo.V0 === V0
        @test foo.pos.a == a_p
        @test foo.pos.b == b_p
        @test foo.neg.a == a_n
        @test foo.neg.b == b_n

        rm("test.h5")
    end # IO

    @testset "generation" begin
        @testset "get_hyb" begin
            @test_throws ArgumentError get_hyb(2)
            Δ = get_hyb(101)
            V = Vector{Float64}
            @test typeof(Δ) == Hybridizationfunction{Float64,Greensfunction{V,V}}
            @test abs2(Δ.V0) + sum(abs2.(Δ.pos.b)) + sum(abs2.(Δ.neg.b)) ≈ 1.0 rtol = 1E-14
            @test norm(Δ.pos.a + reverse(Δ.neg.a)) < 1E-14
            @test norm(abs2.(Δ.pos.b) - reverse(abs2.(Δ.neg.b))) < 1E-14
            @test length(Δ.pos.a) === length(Δ.neg.a) === 50
        end # get_hyb

        @testset "get_hyb_equal" begin
            @test_throws ArgumentError get_hyb_equal(2)
            Δ = get_hyb_equal(101)
            V = Vector{Float64}
            @test typeof(Δ) == Hybridizationfunction{Float64,Greensfunction{V,V}}
            @test Δ.V0 == 1 / sqrt(101)
            @test abs2(Δ.V0) + sum(abs2.(Δ.pos.b)) + sum(abs2.(Δ.neg.b)) ≈ 1.0 rtol = 1E-14
            @test norm(Δ.pos.a + reverse(Δ.neg.a)) === 0.0
            @test norm(abs2.(Δ.pos.b) - reverse(abs2.(Δ.neg.b))) === 0.0
            @test length(Δ.pos.a) === length(Δ.neg.a) === 50
        end # get_hyb_equal
    end # generation

    @testset "Matrix" begin
        a_p = collect(1:2)
        a_n = collect(3:4)
        b_p = collect(5:6)
        b_n = collect(7:8)
        V0 = 9
        G_p = Greensfunction(a_p, b_p)
        G_n = Greensfunction(a_n, b_n)
        Δ = Hybridizationfunction(V0, G_p, G_n)
        m = Matrix(Δ)
        @test m isa Matrix{Int}
        @test m == [
            0 7 8 9 5 6
            7 3 0 0 0 0
            8 0 4 0 0 0
            9 0 0 0 0 0
            5 0 0 0 1 0
            6 0 0 0 0 2
        ]

        # complex does not work
        foo = rand(ComplexF64, 2)
        G = Greensfunction(foo, foo)
        Δ = Hybridizationfunction(V0, G, G)
        @test_throws MethodError Matrix(Δ)
    end # Matrix
end # Hybridization function
