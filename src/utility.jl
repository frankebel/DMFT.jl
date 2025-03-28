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
        Δ::Pole{V,V},
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
    Δ::Pole{V,V},
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
    η_gaussian(η_0::R, η_∞::R, σ::R, ω::AbstractVector{<:R}) where {R<:Real}

Return

```math
η(ω) = η_∞ + (η_0 - η_∞) \\exp\\left(-\\frac{ω^2}{2σ^2}\\right)
```

evaluated on a grid `ω`.
"""
function η_gaussian(η_0::R, η_∞::R, σ::R, ω::AbstractVector{<:R}) where {R<:Real}
    return map(i -> η_∞ .+ (η_0 - η_∞) .* exp.(-i^2 / (2 * σ^2)), ω)
end

_sign1(x::Real) = x >= zero(x) ? one(x) : -one(x)

"""
    G_bethe(z::AbstractVector{<:Number}, D::Real=2.0)

Calculate Bethe lattice Green's function for frequencies `z`, half-bandwidth `D`,
and fermi-energy `ϵ_0`:

```math
G(z) = \\frac{2}{D^2} (z - \\sqrt{z^2 - D^2}).
```

The sign of the real and imagiary parts of ``\\sqrt{z^2 - D^2}``
are corrected accordingly.
"""
function G_bethe(z::AbstractVector{<:Complex}, D::Real=2.0)
    result = similar(z, ComplexF64)
    for i in eachindex(z)
        imag(z[i]) >= 0 || throw(ArgumentError("negative imaginary part"))
        foo = sqrt(z[i]^2 - D^2)
        # correct sign
        foo *= _sign1(real(z[i]))
        result[i] = 2 / D^2 * (z[i] - foo)
    end
    return result
end

function G_bethe(ω::AbstractVector{<:Real}, D::Real=2.0)
    result = similar(ω, ComplexF64)
    for i in eachindex(ω)
        foo = sqrt(complex(ω[i]^2 - D^2))
        # correct sign (only real part)
        bar = sign(ω[i]) * real(foo) + im * imag(foo)
        result[i] = 2 / D^2 * (ω[i] - bar)
    end
    return result
end
