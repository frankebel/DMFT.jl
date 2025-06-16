# utility functions

"""
    get_CI_parameters(n_sites::Int, n_occ::Int, n_v_bit::Int, n_c_bit::Int)

Return `n_bit`, `n_v_vector`, `n_c_vector`.
"""
function get_CI_parameters(n_sites::Int, n_occ::Int, n_v_bit::Int, n_c_bit::Int)
    n_bit = 2 + n_v_bit + n_c_bit
    n_emp = n_sites - n_occ
    n_v_vector = n_occ - 1 - n_v_bit
    n_c_vector = n_emp - 1 - n_c_bit
    return n_bit, n_v_vector, n_c_vector
end

"""
    init_system(
        Δ::Poles{V,V},
        H_int::Operator,
        ϵ_imp::Real,
        n_v_bit::Int,
        n_c_bit::Int,
        e::Int,
        n_kryl::Int,
    ) where {V<:AbstractVector{<:Real}}

Return Hamiltonian, ground state energy, and ground state.
"""
function init_system(
    Δ::Poles{V,V},
    H_int::Operator,
    ϵ_imp::Real,
    n_v_bit::Int,
    n_c_bit::Int,
    e::Int,
    n_kryl::Int,
) where {V<:AbstractVector{<:Real}}
    arr = Array(Δ)
    n_sites = size(arr, 1)
    H_nat, n_occ = to_natural_orbitals(arr)
    n_bit, n_v_vector, n_c_vector = get_CI_parameters(n_sites, n_occ, n_c_bit, n_v_bit)
    fs = FockSpace(Orbitals(n_bit), FermionicSpin(1//2))
    H = natural_orbital_ci_operator(H_nat, H_int, ϵ_imp, fs, n_occ, n_v_bit, n_c_bit, e)
    ψ_start = starting_CIWavefunction(
        Dict{UInt64,Float64}, n_v_bit, n_c_bit, n_v_vector, n_c_vector, e
    )
    E0, ψ0 = DMFT.ground_state(H, ψ_start, n_kryl)
    return H, E0, ψ0
end

"""
    orthogonalize_states(V::AbstractMatrix{<:C}) where {C<:CIWavefunction}

Löwdin orthogonalization for given states in `V`.

Calulate overlap matrix ``S``

```math
\\begin{aligned}
S_{ij} &= ⟨v_i|v_j⟩ \\\\
S    &= V^†V,
\\end{aligned}
```

and diagonalize

```math
\\begin{aligned}
S        &= T Λ T^† \\\\
S^{1/2}  &= T Λ^{1/2} T^† \\\\
S^{-1/2} &= T Λ^{-1/2} T^†.
\\end{aligned}
```

Returns ``W = V S^{-1/2}`` and ``S^{1/2}``.
"""
function orthogonalize_states(V::AbstractMatrix{<:C}) where {C<:CIWavefunction}
    n = size(V, 2)
    T = scalartype(C)
    S = Matrix{T}(undef, n, n)
    mul!(S, V', V)
    E, T = LAPACK.syev!('V', 'U', S)
    S_sqrt_inv = T * Diagonal([1 / sqrt(x) for x in E]) * T'
    S_sqrt = T * Diagonal([sqrt(x) for x in E]) * T'
    W = zero(V)
    mul!(W, V, S_sqrt_inv, true, true)
    return W, S_sqrt
end

"""
   δ_gaussian(δ_0::R, δ_∞::R, σ::R, ω::AbstractVector{<:R}) where {R<:Real}

Return

```math
δ(ω) = δ_∞ + (δ_0 - δ_∞) \\exp\\left(-\\frac{ω^2}{2σ^2}\\right)
```

evaluated on a grid `ω`.
"""
function δ_gaussian(δ_0::R, δ_∞::R, σ::R, ω::AbstractVector{<:R}) where {R<:Real}
    return map(i -> δ_∞ .+ (δ_0 - δ_∞) .* exp.(-i^2 / (2 * σ^2)), ω)
end

"""
    temperature_kondo(U::Real, ϵ::Real, Δ0::Real)

Calculate the Kondo temperature for
an interaction `U`,
on-site with energy `ϵ`,
and hybridization `Δ0`.

```math
T_\\mathrm{K} = \\sqrt{\\frac{UΔ_0}{2}} \\exp(\\frac{π ϵ(ϵ+U)}{2UΔ_0})
```
"""
function temperature_kondo(U::Real, ϵ::Real, Δ0::Real)
    return sqrt(U * Δ0 / 2) * exp(π * ϵ * (ϵ + U) / (2 * U * Δ0))
end

"""
    find_chemical_potential(
        W::AbstractVector{<:Number},
        Hk::AbstractVector{<:AbstractMatrix{<:Number}},
        Σ::AbstractVector{<:AbstractMatrix{<:Number}},
        n::Real;
        tol::Real=1e-3, # tolerance Δμ
        n_max::Int=30, # maximum number of steps
    )

Get chemical potential ``μ``, such that desired filling `n` is fulfilled.

```math
∫_{-∞}^0 \\mathrm{d}ω \\mathrm{Tr}\\left[-\\frac{1}{π}\\mathrm{Im}~G(ω)\\right] ≡ n
```

with

```math
G(ω) = \\frac{1}{N} ∑_k [(ω+μ)I - H_k - Σ(ω)]^{-1}
```

If the self-energy `Σ` is included, it is applied to each diagonal entry of `indices`.

A bisection algorithm is used which stops once `Δμ < tol`
or `n_max` iterations are surpassed.

Returns the calculated chemical potential and effective filling.
"""
function find_chemical_potential(
    W::AbstractVector{<:Number},
    Hk::AbstractVector{<:AbstractMatrix{<:Number}},
    Σ::AbstractVector{<:AbstractMatrix{<:Number}},
    n::Real;
    tol::Real=1e-3, # tolerance Δμ
    n_max::Int=30, # maximum number of steps
)
    # check input
    nb = LinearAlgebra.checksquare(first(Hk)) # number of bands
    all(i -> size(i) == (nb, nb), Hk) ||
        throw(DimensionMismatch("different matrix sizes in Hk"))
    all(i -> size(i) == (nb, nb), Σ) ||
        throw(DimensionMismatch("different matrix sizes in Σ"))
    length(W) == length(Σ) || throw(ArgumentError("length mismatch: W, Σ"))
    Base.require_one_based_indexing(W)
    Base.require_one_based_indexing(Σ)
    tol > 0 || throw(ArgumentError("negative tolerance tol"))
    n_max > 0 || throw(ArgumentError("negative number of steps n_max"))

    # Loop through each combination of (k, ω) and store the eigenvalues of H_k + Σ(ω).
    ev = Array{ComplexF64}(undef, nb, length(Hk), length(Σ)) # storage for eigenvalues
    Threads.@threads for iΣ in eachindex(Σ)
        # foo = Hk + Σ(ω)
        foo = Matrix{ComplexF64}(undef, nb, nb)
        for iH in eachindex(Hk)
            copyto!(foo, Hk[iH])
            foo .+= Σ[iΣ]
            ev[:, iH, iΣ] .= eigvals!(foo)
        end
    end

    # Use bisection to find μ.
    # For starting values use the quantiles: desired filling ± 10 %
    n_low, n_high = n / nb - 0.1, n / nb + 0.1
    # Avoid filling n ∉ [0, 1]
    n_low < 0 && (n_low = zero(n_low))
    n_high > 1 && (n_high = one(n_low))
    # Get inital values.
    μ_low, μ_high = quantile(vec(real(ev)), (n_low, n_high))
    n_low = _get_filling(W, μ_low, ev)
    n_high = _get_filling(W, μ_high, ev)
    n_low <= n <= n_high || throw(ArgumentError("could not find chemical potential"))
    # Bisection with limited number of steps and tolerance.
    steps = 0
    while steps <= n_max && μ_high - μ_low >= tol
        steps += 1
        μ_new = (μ_low + μ_high) / 2
        n_new = _get_filling(W, μ_new, ev)
        n_new >= n ? μ_high = μ_new : μ_low = μ_new # update
    end
    μ = (μ_high + μ_low) / 2
    filling = _get_filling(W, μ, ev)
    return μ, filling
end

function _get_filling(
    W::AbstractArray{<:Number}, # frequency grid
    μ::Real, # chemical potential
    ev::AbstractArray{<:Number,3}, # eigenvalues
)
    # n ∝ ∑_{b,k,ω≤0} (ω+μ-ev_{bkω})^{-1}
    filling = zero(ComplexF64)
    ω0 = findlast(i -> real(i) <= 0, W) # sum all indices ω <= 0
    for iω in 1:ω0
        for k in axes(ev, 2)
            for b in axes(ev, 1)
                @inbounds filling += inv(W[iω] + μ - ev[b, k, iω])
            end
        end
    end
    nk = size(ev, 2) # number of k-points
    dω = real(W[2] - W[1]) # assume equidistant grid
    return -imag(filling) / π / nk * dω
end
