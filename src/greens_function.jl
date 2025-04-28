# Methods related to Green's functions.

# Bethe lattice.

"""
    greens_function_bethe_analytic(z::Number, D::Real=1.0)
    greens_function_bethe_analytic(Z::AbstractVector{<:Number}, D::Real=1.0)

Calculate the Green's function for a Bethe lattice
given a frequency `z` in the upper complex plane,
and half-bandwidth `D`.

```math
G(z) = \\frac{2}{D^2} (z - \\mathrm{sgn}(\\mathrm{Re}(z)) \\sqrt{z^2 - D^2})
```

with ``\\mathrm{sgn}(0) = \\mathrm{sgn}(0^±)``.
"""
function greens_function_bethe_analytic(z::Number, D::Real=1.0)
    D > 0 || throw(DomainError(D, "negative half-bandwidth"))
    s = (-1)^signbit(real(z)) # sign(0) = sign(0^±)
    return 2 / D^2 * (z - s * sqrt((z + 0.0im)^2 - D^2))
end

function greens_function_bethe_analytic(Z::AbstractVector{<:Number}, D::Real=1.0)
    return map(z -> greens_function_bethe_analytic(z, D), Z)
end

"""
    greens_function_bethe_simple(n_bath::Int, D::Real=1.0)

Return the [`Pole`](@ref) representation of the semicircular density of states
with half-bandwidth `D` on `n_bath` poles.

Poles are found by diagonalizing a tridiagonal matrix with hopping ``t=D/2``.

See also:
[`greens_function_bethe_grid`](@ref),
[`greens_function_bethe_equal_weight`](@ref).
"""
function greens_function_bethe_simple(n_bath::Int, D::Real=1.0)
    # check input
    D > 0 || throw(DomainError(D, "negative half-bandwidth"))

    dv = zeros(n_bath)
    ev = fill(D / 2, n_bath - 1) # hopping t = D/2
    H0 = SymTridiagonal(dv, ev)
    a, T = eigen(H0)
    b = abs.(T[:, 1]) # positive values for simplicity
    return Pole(a, b)
end

"""
    greens_function_bethe_grid(grid::AbstractVector{<:Real}, D::Real=1.0)

Return the [`Pole`](@ref) representation of the semicircular density of states
with half-bandwidth `D` with poles given in `grid`.

See also:
[`greens_function_bethe_simple`](@ref),
[`greens_function_bethe_equal_weight`](@ref).
"""
function greens_function_bethe_grid(grid::AbstractVector{<:Real}, D::Real=1.0)
    # check input
    issorted(grid) || throw(ArgumentError("grid is not sorted"))
    allunique(grid) || throw(ArgumentError("grid has duplicate poles"))
    D > 0 || throw(DomainError(D, "negative half-bandwidth"))

    s = Semicircle(D)
    a = copy(grid)
    b = similar(grid)
    if length(grid) == 1
        b[firstindex(b)] = 1
        return Pole(a, b)
    end
    # For each pole location a[i] we bisect the interval to its neighbors
    # a_low = 0.5 * (a[i-1] + a[i])
    # a_high = 0.5 * (a[i] + a[i+1])
    # and calculate the weight as
    # b ∝ cdf(a_high) - cdf(a_low)
    for i in eachindex(a)
        if i == firstindex(a)
            # cdf(-Inf) = 0
            @inbounds b[i] = cdf(s, 0.5 * (a[i] + a[i + 1]))
        elseif i == lastindex(a)
            # cdf(Inf) = 1
            @inbounds b[i] = 1 - cdf(s, 0.5 * (a[i - 1] + a[i]))
        else
            @inbounds b[i] =
                cdf(s, 0.5 * (a[i] + a[i + 1])) - cdf(s, 0.5 * (a[i - 1] + a[i]))
        end
    end
    map!(sqrt, b, b)
    return Pole(a, b)
end

"""
    greens_function_bethe_grid_hubbard3(
        grid::AbstractVector{<:Real}, U::Real=0.0, D::Real=1.0
    )

Return the [`Pole`](@ref) representation of the Hubbard III approximation
with half-bandwidth `D` and poles given in `grid`.

Created using two semicircles at ``±U/2``.
"""
function greens_function_bethe_grid_hubbard3(
    grid::AbstractVector{<:Real}, U::Real=0.0, D::Real=1.0
)
    # check input
    issorted(grid) || throw(ArgumentError("grid is not sorted"))
    allunique(grid) || throw(ArgumentError("grid has duplicate poles"))
    D > 0 || throw(DomainError(D, "negative half-bandwidth"))

    s = Semicircle(D)
    a = copy(grid)
    b = similar(grid)
    if length(grid) == 1
        b[firstindex(b)] = 1
        return Pole(a, b)
    end
    # For each pole location a[i] we bisect the interval to its neighbors
    # a_low = 0.5 * (a[i-1] + a[i])
    # a_high = 0.5 * (a[i] + a[i+1])
    # and calculate the weight as
    # b ∝ cdf(a_high) - cdf(a_low)
    for i in eachindex(a)
        if i == firstindex(a)
            # cdf(-Inf) = 0
            a_high = 0.5 * (a[i] + a[i + 1])
            @inbounds b[i] = cdf(s, a_high + U / 2) + cdf(s, a_high - U / 2)
        elseif i == lastindex(a)
            # cdf(Inf) = 2
            a_low = 0.5 * (a[i - 1] + a[i])
            @inbounds b[i] = 2 - cdf(s, a_low + U / 2) - cdf(s, a_low - U / 2)
        else
            a_low = 0.5 * (a[i - 1] + a[i])
            a_high = 0.5 * (a[i] + a[i + 1])
            @inbounds b[i] =
                cdf(s, a_high + U / 2) + cdf(s, a_high - U / 2) - cdf(s, a_low + U / 2) -
                cdf(s, a_low - U / 2)
        end
    end
    b ./= 2 # noralize 2 distributions
    map!(sqrt, b, b)
    return Pole(a, b)
end

"""
    greens_function_bethe_equal_weight(n_bath::Int, D::Real=1.0)

Return the [`Pole`](@ref) representation of the semicircular density of states
with half-bandwidth `D` on `n_bath` poles.

Each Pole has the same hybridization ``V^2 = 1/n_b``.

See also:
[`greens_function_bethe_simple`](@ref),
[`greens_function_bethe_grid`](@ref).
"""
function greens_function_bethe_equal_weight(n_bath::Int, D::Real=1.0)
    isodd(n_bath) || throw(DomainError(n_bath, "number of bath sites must be odd"))

    V_sqr = 1 / n_bath
    V = sqrt(V_sqr)
    s = Semicircle(D)

    # calculate only negative half, mirror due to symmetry
    q = collect(0:V_sqr:0.5) # equal weight for each pole
    v = quantile.(Semicircle(D), q) # I_l

    # ϵ_l = 1/V_sqr ∫_{I_l} dω ω f(ω)
    # trapezoid rule with `n_p` points
    a = Vector{Float64}(undef, n_bath ÷ 2)
    n_p = 128 # arbitrary number
    for i in eachindex(v)
        i == length(v) && break
        # subtract half of border values
        α = -v[i] * pdf(s, v[i]) - v[i + 1] * pdf(s, v[i + 1])
        α /= 2
        for j in LinRange(v[i], v[i + 1], n_p)
            α += j * pdf(s, j)
        end
        α *= (v[i + 1] - v[i]) / n_p # Δω = I_l/n_p
        a[i] = α
    end
    a .*= n_bath # a .*= 1/V_sqr

    a = [a; 0; -reverse(a)]
    b = fill(V, n_bath)
    return Pole(a, b)
end

# Dispersion relation H_k supplied by user.

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
