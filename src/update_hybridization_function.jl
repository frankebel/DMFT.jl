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
