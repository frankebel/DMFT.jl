using DMFT
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

    @testset "evaluate GF" begin
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
    end # evaluate GF

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
