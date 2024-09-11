# utility functions

"""
    mul_excitation(
        H::Operator, ψ::Wavefunction, m_filled::S, m_empty::S, excitation::Int
    ) where {S}

Calculate `H * ψ` excluding higher excitations than `excitation`.

Assumes that `ψ` only contains Slater determiants with
excitation <= `excitation` in the first place.

`m_filled`, `m_empty` are the masks of default filled/empty sites
in vector of `CIWavefunction`.
"""
function mul_excitation(
    H::Operator, ψ::Wavefunction, m_filled::S, m_empty::S, excitation::Int
) where {S<:Unsigned}
    ϕ = Wavefunction(ψ)
    for (k, v) in ψ, t in H.terms
        k_new, phase = Fermions.Wavefunctions.mul(t, k)
        iszero(phase) && continue
        DMFT.get_excitation(k_new, m_filled, m_empty) > excitation && continue
        if haskey(ϕ, k_new)
            ϕ[k_new] += phase * v * t.value
            iszero(ϕ[k_new]) && delete!(ϕ, k_new)
        else
            ϕ[k_new] = phase * v * t.value
        end
    end
    return ϕ
end

# Return all keys not present in other Wavefunction
function diffkeys(ϕ1::Wavefunction{T}, ϕ2::Wavefunction{T}) where {T}
    length(ϕ1) >= length(ϕ2)
    result = keytype(T)[]
    for k in keys(ϕ1)
        haskey(ϕ2, k) || push!(result, k)
    end
    return Set(result)
end

# # Check sign of amplitude.
# # Check if amplitudes are similar.
# function showdiff(ϕ1::Wavefunction{T}, ϕ2::Wavefunction{T}; kwargs...) where {T}
#     length(ϕ1) == length(ϕ2) || throw(ArgumentError("Wavefunction length mismatch"))
#     for (k, v) in ϕ1
#         try
#             # check anticommutator sign relation
#             sign(v) == sign(ϕ2[k]) || @info "sign mismatch" k ϕ1[k] ϕ2[k]
#             # check if values are similar
#             isapprox(v, ϕ2[k]; kwargs...) || @info "amplitudes" k ϕ1[k] ϕ2[k]
#         catch err
#             isa(err, KeyError) && return KeyError(k)
#         end
#     end
# end

# """
#     highest_amplitudes(ψ::Wavefunction, n_det::Int=20)
#
# Print `n_det` most important amplitudes of `ψ`.
# """
# function highest_amplitudes(ψ::Wavefunction, n_det::Int=20)
#     return display(sort(collect(ψ.terms); by=x -> abs(x[2]), rev=true)[1:n_det])
# end

"""
    starting_Wavefunction(
        D::Type, n_v_bit::Int, n_c_bit::Int, n_v_vector::Int, n_c_vector::Int
    )

Return 2-determinant-state `Wavefunction` with
opposite filling in impurity, mirror bath site.
"""
function starting_Wavefunction(
    D::Type, n_v_bit::Int, n_c_bit::Int, n_v_vector::Int, n_c_vector::Int
)
    K, V = keytype(D), valtype(D)
    s1 = slater_start(K, 0b0110, n_v_bit, n_c_bit, n_v_vector, n_c_vector)
    s2 = slater_start(K, 0b1001, n_v_bit, n_c_bit, n_v_vector, n_c_vector)
    result = Wavefunction(Dict(s1 => one(V), s2 => one(V)))
    normalize!(result)
    return result
end

"""
    starting_CIWavefunction(
        D::Type,
        n_v_bit::Int,
        n_c_bit::Int,
        n_v_vector::Int,
        n_c_vector::Int,
        excitation::Int,
    )

Return 2-determinant-state `CIWavefunction` with
opposite filling in impurity, mirror bath site.

E.g. `D = Dict{UInt64,Float64}` for bit component `UInt64`
and vector component `Vector{Float64}`.
"""
function starting_CIWavefunction(
    D::Type, n_v_bit::Int, n_c_bit::Int, n_v_vector::Int, n_c_vector::Int, excitation::Int
)
    K, V = keytype(D), valtype(D)
    s1 = slater_start(K, 0b0110, n_v_bit, n_c_bit, 0, 0)
    s2 = slater_start(K, 0b1001, n_v_bit, n_c_bit, 0, 0)
    v_dim = sum(i -> binomial(2 * (n_v_vector + n_c_vector), i), 0:excitation)
    v = zeros(V, v_dim)
    v[1] = one(V)
    result = CIWavefunction(
        Dict(s1 => copy(v), s2 => copy(v)),
        2 + n_v_bit + n_c_bit,
        n_v_vector,
        n_c_vector,
        excitation,
    )
    normalize!(result)
    return result
end

"""
    ground_state(H::Operator, ψ_start::Wavefunction, n_kryl::Int)
    ground_state(H::CIOperator, ψ_start::CIWavefunction, n_kryl::Int)

Get approximate ground state and energy using `n_kryl` Krylov cycles.
"""
function ground_state(H::Operator, ψ_start::Wavefunction, n_kryl::Int)
    ψ0, E0 = Fermions.ground_state(H, ψ_start; n_kryl=n_kryl, precision=5E-8, verbose=false)
    return E0, ψ0
end

function ground_state(H::CIOperator, ψ_start::CIWavefunction, n_kryl::Int)
    α, β, states = lanczos_with_states(H, ψ_start, n_kryl)
    E, T = LAPACK.stev!('V', α, β)
    E0 = E[1]
    ψ0 = zero(ψ_start)
    @inbounds for i in eachindex(E)
        axpy!(T[i, 1], states[i], ψ0)
    end
    normalize!(ψ0) # possible orthogonality loss in Lanczos
    Hψ = H * ψ0
    H_avg = dot(ψ0, Hψ)
    H_sqr = dot(Hψ, Hψ)
    var_rel = H_sqr / H_avg^2 - 1
    # compare eigenvalue with expectation value
    rdiff = H_avg / E0 - 1
    abs(rdiff) < 1E-14 || @warn "discrepancy eigenvalue to eigenstate"
    @info "ground state" E0 var_rel length(ψ0)
    return E0, ψ0
end

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
        Δ::Greensfunction{<:T,<:AbstractVector{<:T}},
        H_int::Operator,
        ϵ_imp::T,
        n_v_bit::Int,
        n_c_bit::Int,
        e::Int,
        n_kryl::Int,
    ) where {T<:Real}

Return Hamiltonian, ground state energy, and ground state.
"""
function init_system(
    Δ::Greensfunction{<:T,<:AbstractVector{<:T}},
    H_int::Operator,
    ϵ_imp::T,
    n_v_bit::Int,
    n_c_bit::Int,
    e::Int,
    n_kryl::Int,
) where {T<:Real}
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
    mul!(W, V, S_sqrt_inv)
    return W, S_sqrt
end

"""
    write_vector(s::AbstractString, V::AbstractVector{<:Number})

Write `V` to the given filepath `s`.
"""
function write_vector(s::AbstractString, V::AbstractVector{<:Number})
    h5open(s, "w") do fid
        fid["V"] = V
    end
    return nothing
end

"""
    read_vector(::Type{T}, s::AbstractString) where {T<:Number}

Read `Vector{T}` from filepath `s`.
"""
function read_vector(::Type{T}, s::AbstractString) where {T<:Number}
    h5open(s, "r") do fid
        V::Vector{T} = read(fid, "V")
        return V
    end
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
