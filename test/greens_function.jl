using DMFT
using LinearAlgebra
using Test

@testset "Green's function" begin
    Hk = [[1+0.0im 2; 2 1], [3 4; 4 3]]
    Z = collect(-10:-9) .+ 0.1im
    Σ = [Diagonal([0, 5 + im]), Diagonal([0, 6 + im])] # self-energy only on [2, 2] index

    @testset "local non-interacting" begin
        G0 = greens_function_local(Z, 0, Hk)
        @test length(G0) == 2
        @test norm(
            G0[1] - [
                -0.08948370259088873-0.0008516301906910084im 0.021613692792397263+0.0003827853135677249im
                0.021613692792397263+0.0003827853135677249im -0.08948370259088873-0.0008516301906910084im
            ],
        ) < eps()
        @test norm(
            G0[2] - [
                -0.09894651224745543-0.001052379439830884im 0.0260339595538256+0.0005098764576851289im
                0.0260339595538256+0.0005098764576851289im -0.09894651224745543-0.001052379439830884im
            ],
        ) < eps()
    end # local non-interacting

    @testset "local interacting" begin
        G = greens_function_local(Z, 0, Hk, Σ)
        @test length(G) == 2
        @test norm(
            G[1] - [
                -0.08778121899483271-0.0005616185151491649im 0.014949094763816347-0.0006950450017352803im
                0.014949094763816347-0.0006950450017352803im -0.061604402244826835+0.0034066891091674998im
            ],
        ) < eps()
        @test norm(
            G[2] - [
                -0.09626381282560724-0.0006774835502274772im 0.01636751358241116-0.0007517320959592241im
                0.01636751358241116-0.0007517320959592241im -0.06186055815770346+0.003430278158287015im
            ],
        ) < eps()
    end # local interacting
end # Green's function
