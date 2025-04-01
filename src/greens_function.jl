# Methods related to lattice Green's functions
# where the user supplies a dispersion relation H_k.

"""
    greens_function_local(
        W::AbstractVector{<:Number},
        μ::Real,
        Hk::AbstractVector{<:AbstractMatrix{<:Number}},
        [Σ::AbstractVector{<:AbstractMatrix{<:Number}},]
    )

Calculate the (non-)interacting local Green's function for a dispersion relation `Hk`
and frequency grid `W`.

```math
G_\\mathrm{loc}(ω) = \\frac{1}{N_k} ∑_k ((ω + μ)I - H_k - Σ(ω))^{-1}
```

The self-energy `Σ` is optional.
"""
function greens_function_local(
    W::AbstractVector{<:Number}, μ::Real, Hk::AbstractVector{<:AbstractMatrix{<:Number}}
)
    # check input
    nb = LinearAlgebra.checksquare(first(Hk)) # number of bands
    all(i -> size(i) == (nb, nb), Hk) ||
        throw(DimensionMismatch("different matrix sizes in Hk"))

    m = Matrix{ComplexF64}(undef, nb, nb) # matrix container to reduce allocations
    G_loc = [zero(m) for _ in eachindex(W)]
    for k in eachindex(Hk)
        copyto!(m, Hk[k])
        E, V = LAPACK.syev!('V', 'U', m)
        Threads.@threads for i in eachindex(W)
            @inbounds G_loc[i] .+= V * Diagonal(inv.(W[i] + μ .- E)) * V'
        end
    end
    rmul!.(G_loc, 1 / length(Hk)) # prefactor 1/N_k
    return G_loc
end

# interaction with self-energy Σ
function greens_function_local(
    W::AbstractVector{<:Number},
    μ::Real,
    Hk::AbstractVector{<:AbstractMatrix{<:Number}},
    Σ::AbstractVector{<:AbstractMatrix{<:Number}},
)
    # check input
    nb = LinearAlgebra.checksquare(first(Hk)) # number of bands
    all(i -> size(i) == (nb, nb), Hk) ||
        throw(ArgumentError("different matrix sizes in Hk"))
    all(i -> size(i) == (nb, nb), Σ) ||
        throw(DimensionMismatch("different matrix sizes in Σ"))
    length(W) == length(Σ) || throw(DimensionMismatch("length mismatch: W, Σ"))

    m = Matrix{ComplexF64}(undef, nb, nb) # matrix container to reduce allocations
    G_loc = [zero(m) for _ in eachindex(W)] # local Green's function
    # Calculate local Green's function
    Threads.@threads for i in eachindex(W)
        foo = similar(m)
        for H in Hk
            # foo = (ω + μ)I - H_k - Σ
            copyto!(foo, (W[i] + μ) * I)
            foo .-= H
            foo .-= Σ[i]
            @inbounds G_loc[i] .+= inv(foo)
        end
    end
    rmul!.(G_loc, 1 / length(Hk)) # prefactor 1/N_k
    return G_loc
end

"""
    greens_function_partial(
        G::AbstractVector{<:AbstractMatrix{<:T}}, indices
    ) where {T<:Number}

Calculate the partial Green's function my summing over diagonal terms given by `indices`.
"""
function greens_function_partial(
    G::AbstractVector{<:AbstractMatrix{<:T}}, indices
) where {T<:Number}
    result = zeros(T, length(G))
    for i in eachindex(G)
        for idx in indices
            result[i] += G[i][idx, idx]
        end
    end
    return result
end

"""
    spectral_function_gauss(
        W::AbstractVector{<:Real}, μ::Real, Hk::AbstractVector{<:AbstractMatrix{<:T}}, σ::Real
    ) where {T<:Number}

Calculate the non-interacting local spectrum using Gaussian broadening.

# Arguments
- `W::AbstractVector{<:Real}`: frequency grid
- `μ::Real`: chemical potential
- `Hk::AbstractVector{<:AbstractMatrix{<:T}}`: dispersion relation
- `σ::Real`: broadening of Gaussian
"""
function spectral_function_gauss(
    W::AbstractVector{<:Real}, μ::Real, Hk::AbstractVector{<:AbstractMatrix{<:T}}, σ::Real
) where {T<:Number}
    # check input
    nb = LinearAlgebra.checksquare(first(Hk)) # number of bands
    all(i -> size(i) == (nb, nb), Hk) ||
        throw(DimensionMismatch("different matrix sizes in Hk"))
    σ > 0 || throw(ArgumentError("negative broadening σ"))

    m = Matrix{ComplexF64}(undef, nb, nb) # matrix container to reduce allocations
    Ac::Vector{Matrix{T}} = [zero(m) for _ in eachindex(W)]
    for k in eachindex(Hk)
        copyto!(m, Hk[k])
        E, V = LAPACK.syev!('V', 'U', m)
        Threads.@threads for i in eachindex(W)
            D = Diagonal([exp(-0.5 * ((W[i] + μ - ϵ) / σ)^2) for ϵ in E])
            Ac[i] .+= V * D * V'
        end
    end
    A = map(real, Ac)
    rmul!.(A, 1 / (sqrt(2π) * σ * length(Hk))) # prefactor Gaussian, prefactor 1/N_k
    return A
end
