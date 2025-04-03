using DMFT
using LinearAlgebra
using Test

@testset "Green's function" begin
    @testset "Bethe lattice" begin
        V = Vector{Float64}

        @testset "simple" begin
            # 101 poles
            G = greens_function_bethe_simple(101)
            @test typeof(G) === Pole{V,V}
            @test length(G.a) === length(G.b) === 101
            @test sum(abs2.(G.b)) ≈ 1 rtol = 10 * eps()
            @test G.a[51] ≈ 0 atol = 10 * eps()
            @test norm(G.a + reverse(G.a)) < 50 * eps()
            @test norm(abs2.(G.b) - reverse(abs2.(G.b))) < 600 * eps()
            # 100 poles
            G = greens_function_bethe_simple(100)
            @test typeof(G) === Pole{V,V}
            @test length(G.a) === length(G.b) === 100
            @test sum(abs2.(G.b)) ≈ 1 rtol = 10 * eps()
            @test norm(G.a + reverse(G.a)) < 100 * eps()
            @test norm(abs2.(G.b) - reverse(abs2.(G.b))) < 600 * eps()
            # 101 poles, D = 2
            G = greens_function_bethe_simple(101, 2)
            @test sum(abs2.(G.b)) ≈ 1 rtol = 10 * eps()
        end # simple

        @testset "equal weight" begin
            @test_throws DomainError greens_function_bethe_equal_weight(2)
            G = greens_function_bethe_equal_weight(101)
            @test typeof(G) === Pole{V,V}
            @test length(G.a) === length(G.b) === 101
            @test all(i -> i === 1 / sqrt(101), G.b)
            @test norm(G.a + reverse(G.a)) === 0.0
            @test issorted(G.a)
        end # equal weight
    end # Bethe lattice

    @testset "user supplied dispersion" begin
        Hk = [[1+0.0im 2; 2 1], [3 4; 4 3]]
        Z = collect(-10:-9) .+ 0.1im
        Σ = [Diagonal([0, 5 + im]), Diagonal([0, 6 + im])] # self-energy only on [2, 2] index

        @testset "non-interacting" begin
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
        end # non-interacting

        @testset "interacting" begin
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
        end # interacting

        @testset "partial Green's function" begin
            G = [rand(5, 5) for _ in 1:2]
            @test greens_function_partial(G, 1:5) == [tr(foo) for foo in G]
            Gp = greens_function_partial(G, (1, 4))
            @test Gp[1] == G[1][1, 1] + G[1][4, 4]
            @test Gp[2] == G[2][1, 1] + G[2][4, 4]
        end # partial Green's function

        @testset "spectrum Gauss" begin
            W = [-1, 1]
            A = spectral_function_gauss(W, 0, Hk, 0.05)
            @test typeof(A) === Vector{Matrix{Float64}}
            @test norm(
                A[1] - [
                    3.989422804014326 -3.989422804014326
                    -3.989422804014326 3.989422804014326
                ],
            ) < eps()
            @test iszero(A[2])
        end # spectrum Gauss
    end # user supplied dispersion
end # Green's function
