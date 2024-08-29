using DMFT
using Logging
using Test

Logging.disable_logging(Logging.Info)

include("aqua.jl")

@testset "DMFT.jl" begin
    include("mask.jl")
    include("greensfunction.jl")
    include("sytrd.jl")
    include("nat_orbs.jl")
    include("util.jl")
    include("block_lanczos.jl")
    include("dmft_step.jl")
    include("Combinatorics.jl")
    include("ED.jl")
end
