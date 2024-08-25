using DMFT
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

            # b::Matrix
            b = rand(2, 10)
            G = Greensfunction(a, b)
            foo = zeros(ComplexF64, 2, 2)
            for i in eachindex(a)
                v = b[:, i]
                foo += v * v' ./ (z - a[i])
            end
            @test foo == G(z)
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
        foo = Greensfunction{Float64,Vector{Float64}}("test.h5")
        @test typeof(foo) == Greensfunction{Float64,Vector{Float64}}
        @test foo.a == G.a
        @test foo.b == G.b
        # fallback default type
        foo = Greensfunction("test.h5")
        @test typeof(foo) == Greensfunction{Float64,Vector{Float64}}
        @test foo.a == G.a
        @test foo.b == G.b

        rm("test.h5")
    end # IO
end # Green's function

@testset "hybridization" begin
    A = Float64
    B = Vector{Float64}

    @testset "get_hyb" begin
        @test_throws ArgumentError get_hyb(2)
        Δ = get_hyb(101)
        @test typeof(Δ) === Greensfunction{A,B}
        @test length(Δ.a) === 101
        @test length(Δ.b) === 101
        @test sum(abs2.(Δ.b)) ≈ 1.0 rtol = eps()
        @test all(b -> b > 0, Δ.b)
        @test norm(Δ.a + reverse(Δ.a)) < 50 * eps()
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
end # hybridization
