using DMFT
using DMFT.ED
using Fermions
using LinearAlgebra
using Test

@testset "ED" begin
    n_bath = 3
    U = 4.0
    ϵ_imp = -U / 2
    Δ = get_hyb(n_bath)
    fs = FockSpace(Orbitals(n_bath + 1), FermionicSpin(1//2))
    n = occupations(fs)
    H_int = U * n[1, 1//2] * n[1, -1//2]

    @testset "solve_impurity_ed" begin
        G = solve_impurity_ed(Δ, H_int, ϵ_imp)
        b = abs2.(G.b)
        n_sites = n_bath + 1
        @test length(G.a) ==
            2 * binomial(n_sites, n_sites ÷ 2) * binomial(n_sites, n_bath ÷ 2)
        @test sum(b) ≈ 1 rtol = 10 * eps()
        # PHS
        @test norm(G.a + reverse(G.a)) < 200 * eps()
        @test norm(b - reverse(b)) < 40 * eps()
    end # solve_impurity_ed
end # ED
