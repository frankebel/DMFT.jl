using Aqua

@testset verbose = true "Aqua" begin
    Aqua.test_all(DMFT; ambiguities=false)
end
