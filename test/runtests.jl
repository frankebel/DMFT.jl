using DMFT
using Logging
using Test

Logging.disable_logging(Logging.Info)

include("aqua.jl")

@testset "DMFT.jl" begin
    include("bits.jl")
    include("pole.jl")
    include("io.jl")
    include("sytrd.jl")
    include("natural_orbitals.jl")
    include("wavefunctions.jl")
    include("utility.jl")
    include("greens_function.jl")
    include("hybridization_function.jl")
    include("block_lanczos.jl")
    include("dmft_step.jl")
    include("Combinatorics.jl")
    include("Debug.jl")
    include("ED.jl")
end
