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
    remove_poles_with_zero_weight!(Δ) # TODO: What about pole at zero energy?
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
    G_imp = to_grid(G_imp, Δ0.a)

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
- `G_plus`: for positive frequencies
- `G_minus`: for negative frequencies
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
