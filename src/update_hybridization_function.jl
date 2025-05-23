# Bethe lattice trick: Δ(z) = Δ0(z + μ - Σ(z))

"""
    update_hybridization_function(
        Δ0::Poles{<:V,<:V}, μ::R, Σ_H::R, Σ::Poles{<:V,<:V}
    ) where {V<:AbstractVector{<:Real},R<:Real}


Calculate the new hybridization function in [`Poles`](@ref) representation.

```math
Δ(ω) = Δ_0(ω + μ - Σ(ω))
```
"""
function update_hybridization_function(
    Δ0::P, μ::R, Σ_H::R, Σ::P
) where {P<:Poles{<:Any,<:AbstractVector},R<:Real}
    Σ = remove_poles_with_zero_weight(Σ)
    n = length(Σ) + 1
    n_tot = length(Δ0) * n
    a = Vector{R}(undef, n_tot)
    b = Vector{R}(undef, n_tot)
    result = Poles(a, b)
    foo = Array(Σ) # create once for speed

    loc = locations(Δ0)
    amp = amplitudes(Δ0)

    # diagonalization for each pole in Δ0
    Threads.@threads for i in eachindex(loc)
        idx_low = 1 + n * (i - 1)
        idx_high = idx_low + n - 1
        bar = copy(foo)
        bar[1, 1] = Σ_H - μ + loc[i]
        a[idx_low:idx_high], T = eigen!(bar)
        b[idx_low:idx_high] = amp[i] * view(T, 1, :) # multiply each weight with original b_i
    end

    sort!(result)
    merge_degenerate_poles!(result, eps(R))
    b .= abs.(b) # positve amplitudes are easier

    return result
end
