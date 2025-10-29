# various methods of calculating self-energy from Poles representation

"""
    self_energy_dyson(
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
        ϵ_imp::Real, Δ0::PolesSum, G_imp::PolesSum, grid::AbstractVector{<:Real} = locations(Δ0)
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
    self_energy_IFG(C::PolesSumBlock)

Calculate self-energy as
``Σ_ω = I_ω - F^\\mathrm{L}_ω (G_ω)^{-1} F^\\mathrm{R}_ω``.

Assumes that `C` is given as

```math
\\begin{pmatrix}
I             & F^\\mathrm{L} \\\\
F^\\mathrm{R} & G
\\end{pmatrix}.
```
"""
function self_energy_IFG(C::PolesSumBlock)
    # Thank you Aleksandrs Začinskis for showing me the method of
    # - matrix inversion
    # - taking [1,1] element
    # - inversion again

    size(C) == (2, 2) || throw(ArgumentError("C must have size (2, 2)"))

    # convert C to continued fraction representation
    C_cf = PolesContinuedFractionBlock(C)
    iszero(scale(C_cf)[1, 2]) || throw(ArgumentError("`scale(C)` must be diagonal")) # TODO: extend this
    S_inv = inv(scale(C_cf))
    A1 = locations(C_cf)[1]

    # Shift continued fraction representation by one index and take [1,1] block
    C_cf_shift = PolesContinuedFractionBlock(
        locations(C_cf)[2:end], amplitudes(C_cf)[2:end], amplitudes(C_cf)[1]
    )
    C_shift = PolesSumBlock(C_cf_shift)
    T = PolesSum(C_shift, 1, 1) # C_shift_[1,1]
    # Reduce noise in T by removing all weight smaller than relative ϵ.
    merge_small_weight!(T, moment(T, 0) * eps())

    # create new sum representation by adding S_[1,1] and [A_1]_[1,1] for Anderson representation.
    s = inv(S_inv[1, 1]) # new scale, is U/2 for half-filling
    foo = Array(T)
    foo[1, 1] = A1[1, 1]
    F = eigen!(Hermitian(foo))
    loc = F.values
    wgt = abs2(s) .* abs2.(view(F.vectors, 1, :))
    return PolesSum(loc, wgt)
end

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
