using Aqua
using DMFT
using Test

@testset verbose = true "Aqua" begin
    Aqua.test_all(DMFT; ambiguities = false)
end
