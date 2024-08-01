using DMFT
using Test

include("./aqua.jl")

@testset "DMFT.jl" begin
    include("./combinatorics.jl")
    include("./mask.jl")
    include("./wavefunctions.jl")
    include("./greensfunction.jl")
end
