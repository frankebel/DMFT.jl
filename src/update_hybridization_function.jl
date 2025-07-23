# Bethe lattice trick: Δ(z) = Δ0(z + μ - Σ(z))

"""
    update_hybridization_function(
        Δ0::Poles{<:V,<:V}, μ::R, Σ_H::R, Σ::Poles{<:V,<:V}
    ) where {V<:AbstractVector{<:Real},R<:Real}


Calculate the new hybridization function in [`PolesSum`](@ref) representation.

```math
Δ(ω) = Δ_0(ω + μ - Σ(ω))
```
"""
function update_hybridization_function(
    Δ0::PolesSum{R,R}, μ::R, Σ_H::R, Σ::PolesSum{R,R}
) where {R<:Real}
    Σ = remove_zero_weight(Σ)
    foo = Matrix{R}(Array(Σ)) # create once for speed

    n = length(Σ) + 1
    n_tot = length(Δ0) * n
    loc_new = Vector{R}(undef, n_tot)
    wgt_new = Vector{R}(undef, n_tot)
    result = PolesSum(loc_new, wgt_new)

    # diagonalization for each pole in Δ0
    Threads.@threads for i in eachindex(Δ0)
        idx_low = 1 + n * (i - 1)
        idx_high = idx_low + n - 1
        bar = copy(foo)
        bar[1, 1] = Σ_H - μ + locations(Δ0)[i]
        loc_new[idx_low:idx_high], T = eigen!(Symmetric(bar))
        wgt_new[idx_low:idx_high] = weight(Δ0, i) .* abs2.(view(T, 1, :)) # multiply new weights with original
    end

    sort!(result)
    merge_degenerate_poles!(result, eps(R))

    return result
end
