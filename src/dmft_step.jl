# multiple correlators with block Lanczos
"""
    dmft_step(
        Δ0::Poles{V,V},
        Δ::Poles{V,V},
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
    Δ0::Poles{V,V},
    Δ::Poles{V,V},
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
    Δ0::Poles{V,V},
    Δ::Poles{V,V},
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
