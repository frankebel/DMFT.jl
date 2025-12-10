using Aqua
using RAS_DMFT
using Test

@testset verbose = true "Aqua" begin
    Aqua.test_all(RAS_DMFT; ambiguities = false)
end
