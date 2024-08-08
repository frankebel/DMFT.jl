using DMFT
using Test

include("aqua.jl")

@testset "DMFT.jl" begin
    include("combinatorics.jl")
    include("mask.jl")
    include("greensfunction.jl")
    include("sytrd.jl")
    include("nat_orbs.jl")
    include("util.jl")
end
