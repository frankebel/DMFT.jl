# various methods of calculating self-energy from Poles representation

"""
    self_energy_poles(
        ϵ_imp::Real, Δ0::Poles{<:Any,<:AbstractVector}, G_imp::Poles{<:Any,<:AbstractVector}
    )


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
    _merge_degenerate_poles_weights!(Σ, 0)
    _merge_negative_weight!(Σ)
    remove_poles_with_zero_weight!(Σ)
    Σ.b .= sqrt.(Σ.b) # back to amplitudes

    return Σ_H, Σ
end

# https://doi.org/10.1088/0953-8984/10/37/021
"""
    self_energy_FG(
        C_plus::P, C_minus::P, Z::AbstractVector{<:Complex}
    ) where {P<:Poles{<:Any,<:AbstractMatrix}}

Calculate self-energy as ``Σ_z = F_z (G_z)^{-1}``.
"""
function self_energy_FG(
    C_plus::P, C_minus::P, Z::AbstractVector{<:Complex}
) where {P<:Poles{<:Any,<:AbstractMatrix}}
    cp = C_plus(Z)
    cm = C_minus(Z) # transpose to access F component
    G = map(c -> c[1, 1], cm) .+ map(c -> c[1, 1], cp)
    F = map(c -> c[1, 2], cm) .+ map(c -> c[2, 1], cp)
    return F ./ G
end

# https://doi.org/10.1103/PhysRevB.105.245132
"""
    self_energy_IFG(
        C_plus::P, C_minus::P, Z::AbstractVector{<:Complex}, Σ_H::Real
    ) where {P<:Poles{<:Any,<:AbstractMatrix}}

Calculate self-energy as ``Σ_z = Σ^\\mathrm{H} + I_z - F^\\mathrm{L}_z (G_z)^{-1} F^\\mathrm{R}_z``.
"""
function self_energy_IFG(
    C_plus::P, C_minus::P, Z::AbstractVector{<:Complex}, Σ_H::Real
) where {P<:Poles{<:Any,<:AbstractMatrix}}
    cp = C_plus(Z)
    cm = C_minus(Z) # transpose to access F components
    G = map(c -> c[1, 1], cm) .+ map(c -> c[1, 1], cp)
    F_R = map(c -> c[2, 1], cm) .+ map(c -> c[1, 2], cp)
    F_L = map(c -> c[1, 2], cm) .+ map(c -> c[2, 1], cp)
    I = map(c -> c[2, 2], cm) .+ map(c -> c[2, 2], cp)
    return Σ_H .+ I - F_L ./ G .* F_R
end

"""
    self_energy_IFG_gauss(
        G_plus::P, G_minus::P, W::AbstractVector{<:Real}, σ::Real, Σ_H::Real
    ) where {P<:Poles{<:Any,<:AbstractMatrix{<:Number}}}

Calculate self-energy as ``Σ_z = Σ^\\mathrm{H} + I_z - F^\\mathrm{L}_z (G_z)^{-1} F^\\mathrm{R}_z``
with Gaussian broadening.

Real part is obtained by Kramers-Kronig relation.
"""
function self_energy_IFG_gauss(
    C_plus::P, C_minus::P, W::AbstractVector{<:Real}, σ::Real, Σ_H::Real
) where {P<:Poles{<:Any,<:AbstractMatrix}}
    cp = C_plus(W, σ)
    cm = C_minus(W, σ) # transpose to access F components
    G = map(c -> c[1, 1], cm) .+ map(c -> c[1, 1], cp)
    F_R = map(c -> c[1, 2], cm) .+ map(c -> c[2, 1], cp)
    F_L = map(c -> c[1, 2], cm) .+ map(c -> c[2, 1], cp)
    I = map(c -> c[2, 2], cm) .+ map(c -> c[2, 2], cp)
    return Σ_H .+ I - F_L ./ G .* F_R
end
