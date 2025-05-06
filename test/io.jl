using DMFT
using Test

@testset "IO" begin
    @testset "Number" begin
        s = 1 + 2im
        @test write_hdf5("test.h5", s) === nothing
        @test @inferred read_hdf5("test.h5", Complex{Int}) isa Complex{Int}
        @test read_hdf5("test.h5", Complex{Int}) === 1 + 2im
        @test read_hdf5("test.h5", ComplexF64) === 1.0 + 2.0im
        @test_throws InexactError read_hdf5("test.h5", Int)
    end # Number

    @testset "Array" begin
        # Vector{Number}
        v = rand(10)
        @test write_hdf5("test.h5", v) === nothing
        @test @inferred read_hdf5("test.h5", Vector{Float64}) isa Vector{Float64}
        @test read_hdf5("test.h5", Vector{Float64}) == v

        # Matrix{Number}
        m = rand(10, 10)
        @test write_hdf5("test.h5", m) === nothing
        @test @inferred read_hdf5("test.h5", Matrix{Float64}) isa Matrix{Float64}
        @test read_hdf5("test.h5", Matrix{Float64}) == m

        # Vector{Matrix{Number}}
        v = [rand(2, 2) for _ in 1:10]
        @test write_hdf5("test.h5", v) === nothing
        @test @inferred read_hdf5("test.h5", Vector{Matrix{Float64}}) isa
            Vector{Matrix{Float64}}
        @test read_hdf5("test.h5", Vector{Matrix{Float64}}) == v
    end # Array

    @testset "Poles" begin
        V = Vector{Float64}
        M = Matrix{Float64}
        # b::Vector
        # write
        a = sort!(rand(10))
        b = rand(10)
        P = Poles(a, b)
        @test write_hdf5("test.h5", P) === nothing
        # read
        @test @inferred read_hdf5("test.h5", Poles{V,V}) isa Poles{V,V}
        foo = read_hdf5("test.h5", Poles{V,V})
        @test foo.a == P.a
        @test foo.b == P.b

        # b::Matrix
        # write
        a = sort!(rand(10))
        b = rand(10, 10)
        P = Poles(a, b)
        @test write_hdf5("test.h5", P) === nothing
        # read
        @test @inferred read_hdf5("test.h5", Poles{V,M}) isa Poles{V,M}
        @test_throws MethodError read_hdf5("test.h5", Poles{V,V})
        foo = read_hdf5("test.h5", Poles{V,M})
        @test foo.a == P.a
        @test foo.b == P.b
    end # Poles
    isfile("test.h5") && rm("test.h5")
end # IO
