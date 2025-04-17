# correlator with Lanczos
"""
    dmft_step(
        Δ0::Pole{V,V},
        Δ::Pole{V,V},
        H_int::Op,
        μ::Real,
        ϵ_imp::Real,
        n_v_bit::Int,
        n_c_bit::Int,
        e::Int,
        O::Op, # creator
        Ogs::AbstractVector{<:Op}, # operators to measure on ground state
        n_kryl::Int,
        n_kryl_gs::Int,
    ) where {V<:AbstractVector{<:Real},Op<:Operator}

Calculate a single step of DMFT.

Returns
- `G_imp::Pole{V,V}`: impurity Green's function
- `Σ_H::eltype{V}`: self-energy Hartree term
- `Σ::Pole{V,V}`: self-energy excluding Hartree term
- `Δ_new::Pole{V,V}`: new hybridization function
- `E0::eltype{V}`: ground-state energy
- `expectation_values::V`: expectation values for `Ogs`
"""
function dmft_step(
    Δ0::Pole{V,V},
    Δ::Pole{V,V},
    H_int::Op,
    μ::Real,
    ϵ_imp::Real,
    n_v_bit::Int,
    n_c_bit::Int,
    e::Int,
    O::Op, # creator
    Ogs::AbstractVector{<:Op}, # operators to measure on ground state
    n_kryl::Int,
    n_kryl_gs::Int,
) where {V<:AbstractVector{<:Real},Op<:Operator}
    # initialize system
    H, E0, ψ0 = init_system(Δ, H_int, ϵ_imp, n_v_bit, n_c_bit, e, n_kryl_gs)

    # expectation values on ground state
    expectation_values = V(undef, length(Ogs))
    for i in eachindex(Ogs)
        expectation_values[i] = dot(ψ0, Ogs[i] * ψ0)
    end

    # impurity Green's function
    G_plus = DMFT._pos(H, E0, ψ0, O, n_kryl)
    G_minus = DMFT._neg(H, E0, ψ0, O, n_kryl)
    G_imp = Pole([G_minus.a; G_plus.a], [G_minus.b; G_plus.b])
    sort!(G_imp)
    merge_equal_poles!(G_imp)

    # self-energy
    Σ_H, Σ = self_energy_pole(ϵ_imp, Δ0, G_imp)

    # new hybridization function
    Δ_new = update_hybridization_function(Δ0, μ, Σ_H, Σ)

    return G_imp, Σ_H, Σ, Δ_new, E0, expectation_values
end

# multiple correlators with block Lanczos
"""
    dmft_step(
        Δ0::Pole{V,V},
        Δ::Pole{V,V},
        H_int::Op,
        μ::Real,
        ϵ_imp::Real,
        Z::AbstractVector{<:Complex},
        n_v_bit::Int,
        n_c_bit::Int,
        e::Int,
        O::AbstractVector{<:Op},
        n_kryl::Int,
        n_kryl_gs::Int,
        n_dis::Int,
        η::Real,
        improved_self_energy::Bool,
    ) where {V<:AbstractVector{<:Real},Op<:Operator}

Calculate a single step of DMFT.

Returns
- `G_plus`: for positive freqeuncies
- `G_minus`: for negative freqeuncies
- `Δ_new`: new hybridization function
- `Δ_grid`: new hybridization function on a grid
"""
function dmft_step(
    Δ0::Pole{V,V},
    Δ::Pole{V,V},
    H_int::Op,
    μ::Real,
    ϵ_imp::Real,
    Z::AbstractVector{<:Complex},
    n_v_bit::Int,
    n_c_bit::Int,
    e::Int,
    O::AbstractVector{<:Op},
    n_kryl::Int,
    n_kryl_gs::Int,
    n_dis::Int,
    η::Real,
) where {V<:AbstractVector{<:Real},Op<:Operator}
    G_plus, G_minus, Σ_H = solve_impurity(
        Δ, H_int, ϵ_imp, n_v_bit, n_c_bit, e, n_kryl_gs, n_kryl, O
    )
    Σ = self_energy_IFG(G_plus, G_minus, Z, Σ_H)
    Δ_grid = update_hybridization_function(Δ0, μ, Z, Σ)
    Δ_new = equal_weight_discretization(-imag(Δ_grid), real(Z), η, n_dis)
    return G_plus, G_minus, Δ_new, Δ_grid
end

function dmft_step_gauss(
    Δ0::Pole{V,V},
    Δ::Pole{V,V},
    H_int::Op,
    μ::Real,
    ϵ_imp::Real,
    ω::AbstractVector{<:Real},
    n_v_bit::Int,
    n_c_bit::Int,
    e::Int,
    O::AbstractVector{<:Op},
    n_kryl::Int,
    n_kryl_gs::Int,
    n_dis::Int,
    σ::Real,
) where {V<:AbstractVector{<:Real},Op<:Operator}
    G_plus, G_minus, Σ_H = solve_impurity(
        Δ, H_int, ϵ_imp, n_v_bit, n_c_bit, e, n_kryl_gs, n_kryl, O
    )
    Σ = self_energy_IFG_gauss(G_plus, G_minus, ω, σ, Σ_H)
    Δ_grid = update_hybridization_function(Δ0, μ, ω, Σ)
    Δ_new = equal_weight_discretization(-imag(Δ_grid), real(ω), σ, n_dis)
    return G_plus, G_minus, Δ_new, Δ_grid
end

"""
    equal_weight_discretization(
        imΔ::AbstractVector{<:Real}, w::AbstractVector{<:Real}, η::Real, n::Int
    )

Discretize the given function `imΔ` on `n` poles, such that each pole has equal weight.

Assumes that `w` is an equidistant grid.
Assumes that `w` has odd number of values.
Assumes that `w` is a symmetric interval.
Assumes that `imΔ` has only semipositive values.
"""
function equal_weight_discretization(
    imΔ::AbstractVector{<:Real}, w::AbstractVector{<:Real}, η::Real, n::Int
)
    n >= 3 || throw(ArgumentError("at least 3 poles necessary"))
    isodd(n) || throw(ArgumentError("need odd `n`"))

    N = length(w)
    dw = w[2] - w[1]
    M = cld(N, 2)
    total = sum(imΔ) * dw
    v = total / n
    v0 = imΔ[M] * dw
    i = 1
    # pole at w = 0 : weight = min(v, weight in [-η,η])
    while w[M + i] <= η && M + i <= N && v0 < v
        v0 += imΔ[M + i] * dw + imΔ[M - i] * dw
        i += 1
    end
    # calculate total remaining weight for positive freqeuncies:
    wght_right = sum(imΔ[(M + i):end]) * dw
    # calculate total remaining weight for negative freqeuncies:
    wght_left = sum(imΔ[1:(M - i)]) * dw
    j = i # remember i for later

    # discretize for positive frequencies:
    P_plus = Float64[]
    V_plus = Float64[]
    vp = 0.0
    pp = 0.0
    v = wght_right / ((n - 1) ÷ 2)
    partial = 0.0
    while partial < wght_right && M + i <= N
        while vp < v && M + i <= N # accumulate weight until v is exceeded or end of grid
            vp += imΔ[M + i] * dw
            pp += w[M + i] * imΔ[M + i] * dw
            i += 1
        end
        δv = vp - v
        if δv > 0 # if vp exceeds v carry over the difference for next pole
            pp -= w[M + i - 1] * δv
            partial += v
            push!(V_plus, sqrt(v / π))
            push!(P_plus, pp / v)
            vp = δv
            pp = w[M + i - 1] * δv
        elseif vp > 10 * eps()
            # no overshoot, but still some weight
            partial += vp
            push!(V_plus, sqrt(vp / π))
            push!(P_plus, pp / vp)
            vp = 0.0
            pp = 0.0
        end
    end

    # discretize for negative frequencies:
    P_minus = Float64[]
    V_minus = Float64[]
    vm = 0.0
    pm = 0.0
    v = wght_left / ((n - 1) ÷ 2)
    partial = 0.0
    while partial < wght_left && M - j > 0
        while vm < v && M - j > 0 # accumulate weight until v is exceeded or end of grid
            vm += imΔ[M - j] * dw
            pm += w[M - j] * imΔ[M - j] * dw
            j += 1
        end
        δv = vm - v
        if δv > 0 # if vm exceeds v carry over the difference for next pole
            pm -= w[M - j + 1] * δv
            partial += v
            push!(V_minus, sqrt(v / π))
            push!(P_minus, pm / v)
            vm = δv
            pm = w[M - j + 1] * δv
        elseif vm > 10 * eps()
            # no overshoot, but still some weight
            partial += vm
            push!(V_minus, sqrt(vm / π))
            push!(P_minus, pm / vm)
            vm = 0.0
            pm = 0.0
        end
    end
    a = [reverse!(P_minus); 0; P_plus]
    b = [reverse!(V_minus); sqrt(v0 / π); V_plus]
    return Pole(a, b)
end
