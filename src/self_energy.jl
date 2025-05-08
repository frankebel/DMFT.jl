# various methods of calculating self-energy from Poles representation

"""
    self_energy_poles(
    ϵ_imp::Real, Δ0::Poles{<:V,<:V}, G_imp::Poles{<:V,<:V}
) where {V<:AbstractVector{<:Real}}

Calculate the self-energy purely in [`Poles`](@ref) representation using the Dyson equation.

```math
Σ(ω)
= G_{\\mathrm{imp},0}(ω)^{-1} - G_\\mathrm{imp}(ω)^{-1}
= ω - ϵ_\\mathrm{imp} - Δ(ω) - G_\\mathrm{imp}(ω)^{-1}
```

Poles with negative weight are moved into neighbors such that the zeroth and first moment
is conserved locally.
"""
function self_energy_poles(
    ϵ_imp::Real, Δ0::Poles{<:Any,<:AbstractVector}, G_imp::Poles{<:Any,<:AbstractVector}
)

    # invert impurity Green's function
    a0, G_imp_inv = inv(G_imp)

    # Hartree term
    Σ_H = a0 - ϵ_imp

    # Σ = G_imp_inv - Δ0
    Σ = Poles([locations(G_imp_inv); locations(Δ0)], [weights(G_imp_inv); -weights(Δ0)])
    sort!(Σ)
    _merge_degenerate_poles_square!(Σ)
    merge_negative_weight!(Σ)
    remove_poles_with_zero_weight!(Σ)
    Σ.b .= sqrt.(Σ.b) # back to amplitudes

    return Σ_H, Σ
end

# https://doi.org/10.1088/0953-8984/10/37/021
"""
    self_energy_FG(
        G_plus::GF, G_minus::GF, Z::AbstractVector{<:Complex}
    ) where {GF<:Poles{<:Any,<:AbstractMatrix{<:Number}}}

Calculate self-energy as ``Σ(Z) = F(Z) G^{-1}(Z)``.
"""
function self_energy_FG(
    G_plus::P, G_minus::P, Z::AbstractVector{<:Complex}
) where {P<:Poles{<:Any,<:AbstractMatrix{<:Number}}}
    gp = G_plus(Z)
    gm = G_minus(Z) # transpose to access F component
    G = map(g -> g[1, 1], gm) .+ map(g -> g[1, 1], gp)
    F = map(g -> g[1, 2], gm) .+ map(g -> g[2, 1], gp)
    return F ./ G
end

# https://doi.org/10.1103/PhysRevB.105.245132
"""
    self_energy_IFG(
        G_plus::GF, G_minus::GF, Z::AbstractVector{<:Complex}, Σ_H::Real
    ) where {GF<:Poles{<:Any,<:AbstractMatrix{<:Number}}}

Calculate self-energy as ``Σ(Z) = Σ^\\mathrm{H} + I(Z) - F^\\mathrm{L}(Z) G^{-1}(Z) F^\\mathrm{R}(Z)``.
"""
function self_energy_IFG(
    G_plus::P, G_minus::P, Z::AbstractVector{<:Complex}, Σ_H::Real
) where {P<:Poles{<:Any,<:AbstractMatrix{<:Number}}}
    gp = G_plus(Z)
    gm = G_minus(Z) # transpose to access F components
    G = map(g -> g[1, 1], gm) .+ map(g -> g[1, 1], gp)
    F_R = map(g -> g[2, 1], gm) .+ map(g -> g[1, 2], gp)
    F_L = map(g -> g[1, 2], gm) .+ map(g -> g[2, 1], gp)
    I = map(g -> g[2, 2], gm) .+ map(g -> g[2, 2], gp)
    return Σ_H .+ I - F_L ./ G .* F_R
end

"""
    self_energy_IFG_gauss(
        G_plus::P, G_minus::P, W::AbstractVector{<:Real}, σ::Real, Σ_H::Real
    ) where {P<:Poles{<:Any,<:AbstractMatrix{<:Number}}}

Calculate self-energy as ``Σ(W) = Σ^\\mathrm{H} + I(W) - F^\\mathrm{L}(W) G^{-1}(W) F^\\mathrm{R}(W)``
with Gaussian broadening.

Real part is obtained by Kramers-Kronig relation.
"""
function self_energy_IFG_gauss(
    G_plus::P, G_minus::P, W::AbstractVector{<:Real}, σ::Real, Σ_H::Real
) where {P<:Poles{<:Any,<:AbstractMatrix{<:Number}}}
    gp = G_plus(W, σ)
    gm = G_minus(W, σ) # transpose to access F components
    G = map(g -> g[1, 1], gm) .+ map(g -> g[1, 1], gp)
    F_R = map(g -> g[1, 2], gm) .+ map(g -> g[2, 1], gp)
    F_L = map(g -> g[1, 2], gm) .+ map(g -> g[2, 1], gp)
    I = map(g -> g[2, 2], gm) .+ map(g -> g[2, 2], gp)
    return Σ_H .+ I - F_L ./ G .* F_R
end
