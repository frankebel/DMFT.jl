# various methods of calculating self-energy from Poles representation

"""
    self_energy_poles(
        ϵ_imp::Real,
        Δ0::PolesSum,
        G_imp::PolesSum,
        grid::AbstractVector{<:Real}=locations(Δ0),
    )

Calculate the self-energy purely in [`PolesSum`](@ref) representation
using the Dyson equation.

```math
Σ(ω)
= G_{\\mathrm{imp},0}(ω)^{-1} - G_\\mathrm{imp}(ω)^{-1}
= ω - ϵ_\\mathrm{imp} - Δ_0(ω) - G_\\mathrm{imp}(ω)^{-1}
```

Poles with negative weight are moved into neighbors such that the zeroth and first moment
is conserved locally.
"""
function self_energy_dyson(
    ϵ_imp::Real, Δ0::PolesSum, G_imp::PolesSum, grid::AbstractVector{<:Real}=locations(Δ0)
)

    # invert impurity Green's function
    a0, G_imp_inv = inv(G_imp)

    # Hartree term
    Σ_H = a0 - ϵ_imp

    # sum of poles
    Σ = G_imp_inv - Δ0
    Σ = to_grid(Σ, grid)
    merge_negative_weight!(Σ)
    remove_zero_weight!(Σ)
    return Σ_H, Σ
end

# https://doi.org/10.1103/PhysRevB.105.245132
"""
    self_energy_IFG(Σ_H::Real, C::PolesSumBlock, W, δ)

Calculate self-energy as
``Σ_z = Σ^\\mathrm{H} + I_z - F^\\mathrm{L}_z (G_z)^{-1} F^\\mathrm{R}_z``
with Lorentzian broadening.

Assumes that `C` is given as

```math
\\begin{pmatrix}
I             & F^\\mathrm{L} \\\\
F^\\mathrm{R} & G
\\end{pmatrix}.
```
"""
function self_energy_IFG_lorentzian(Σ_H::Real, C::PolesSumBlock, W, δ)
    c = evaluate_lorentzian(C, W, δ)
    I = map(c -> c[1, 1], c)
    F_L = map(c -> c[1, 2], c)
    F_R = map(c -> c[2, 1], c)
    G = map(c -> c[2, 2], c)
    return Σ_H .+ I - F_L ./ G .* F_R
end

"""
    self_energy_IFG_gaussian(Σ_H::Real, C::PolesSumBlock, W, σ)

Calculate self-energy as
``Σ_z = Σ^\\mathrm{H} + I_z - F^\\mathrm{L}_z (G_z)^{-1} F^\\mathrm{R}_z``
with Gaussian broadening.

Assumes that `C` is given as

```math
\\begin{pmatrix}
I             & F^\\mathrm{L} \\\\
F^\\mathrm{R} & G
\\end{pmatrix}.
```
"""
function self_energy_IFG_gaussian(Σ_H::Real, C::PolesSumBlock, W, σ)
    c = evaluate_gaussian(C, W, σ)
    I = map(c -> c[1, 1], c)
    F_L = map(c -> c[1, 2], c)
    F_R = map(c -> c[2, 1], c)
    G = map(c -> c[2, 2], c)
    return Σ_H .+ I - F_L ./ G .* F_R
end
