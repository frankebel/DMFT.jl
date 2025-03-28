using DMFT
using Logging
using Test

Logging.disable_logging(Logging.Info)

include("aqua.jl")

@testset "DMFT.jl" begin
    include("mask.jl")
    include("pole.jl")
    include("io.jl")
    include("sytrd.jl")
    include("nat_orbs.jl")
    include("util.jl")
    include("block_lanczos.jl")
    include("dmft_step.jl")
    include("Combinatorics.jl")
    include("Debug.jl")
    include("ED.jl")
end
