# Bethe lattice trick: Δ(z) = Δ0(z + μ - Σ(z))

"""
    update_hybridization_function(
        Δ0::Pole{<:V,<:V}, μ::Real, Z::AbstractVector{<:Number}, Σ::AbstractVector{<:Complex}
    ) where {V<:AbstractVector{<:Real}}

Calculate the new hybridization function from the lattice hybridization `Δ0` and
the impurity self energy `Σ` on grid `Z`.

```math
Δ(Z) = Δ_0(Z + μ - Σ(Z))
```
"""
function update_hybridization_function(
    Δ0::Pole{<:V,<:V}, μ::Real, Z::AbstractVector{<:Number}, Σ::AbstractVector{<:Complex}
) where {V<:AbstractVector{<:Real}}
    return Δ0(Z .+ μ - Σ)
end

"""
    update_hybridization_function(
        Δ0::Pole{<:V,<:V}, μ::R, Σ_H::R, Σ::Pole{<:V,<:V}
    ) where {V<:AbstractVector{<:Real},R<:Real}


Calculate the new hybridization function in [`Pole`](@ref) representation.

```math
Δ(z) = Δ_0(z + μ - Σ(z))
```

The poles are taken from the input hybridization: `Δ.a == Δ0.a`
"""
function update_hybridization_function(
    Δ0::Pole{<:V,<:V}, μ::R, Σ_H::R, Σ::Pole{<:V,<:V}
) where {V<:AbstractVector{<:Real},R<:Real}
    Σ = remove_poles_with_zero_weight(Σ)
    n = length(Σ) + 1
    n_tot = length(Δ0) * n
    a = V(undef, n_tot)
    b = V(undef, n_tot)
    result = Pole(a, b)
    foo = Array(Σ) # create once for speed

    # diagonalization for each pole in Δ0
    Threads.@threads for i in eachindex(Δ0.a)
        idx_low = 1 + n * (i - 1)
        idx_high = idx_low + n - 1
        bar = copy(foo)
        bar[1, 1] = Σ_H - μ + Δ0.a[i]
        a[idx_low:idx_high], T = eigen!(bar)
        b[idx_low:idx_high] = Δ0.b[i] * view(T, 1, :) # multiply each weight with original b_i
    end

    # put on the same grid as Δ0
    result = to_grid(result, Δ0.a)

    return result
end
