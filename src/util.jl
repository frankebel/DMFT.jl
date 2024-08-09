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
    starting_wf(D::Type, n_v_bit::Int, n_c_bit::Int, n_v_vector::Int, n_c_vector::Int)

Return starting wave function with opposite filling in impurity, mirror bath site.
"""
function starting_wf(D::Type, n_v_bit::Int, n_c_bit::Int, n_v_vector::Int, n_c_vector::Int)
    K, V = keytype(D), valtype(D)
    s1 = slater_start(K, 0b0110, n_v_bit, n_c_bit, n_v_vector, n_c_vector)
    s2 = slater_start(K, 0b1001, n_v_bit, n_c_bit, n_v_vector, n_c_vector)
    result = Wavefunction(Dict(s1 => one(V), s2 => one(V)))
    normalize!(result)
    return result
end

# function starting_ciwf(
#     D::Type, n_v_bit::Int, n_c_bit::Int, n_v_vector::Int, n_c_vector::Int, e::Int
# )
#     K, V = keytype(D), valtype(D)
#     s1 = slater_start(K, 0b0110, n_v_bit, n_c_bit, 0, 0)
#     s2 = slater_start(K, 0b1001, n_v_bit, n_c_bit, 0, 0)
#     v_dim = sum(i -> binomial(2 * (n_v_vector + n_c_vector), i), 0:e)
#     v = zeros(V, v_dim)
#     v[1] = one(V)
#     result = CIWavefunction{2 + n_v_bit + n_c_bit,n_v_vector,n_c_vector,e}(
#         Dict(s1 => copy(v), s2 => copy(v))
#     )
#     normalize!(result)
#     return result
# end

# function ground_state(
#     H::CIOperator,
#     ψ_start::CIWavefunction;
#     n_kryl::Int=50,
#     precision::Real=1E-8,
#     n_iter::Int=100,
#     verbose::Bool=true,
# )
#     E0 = 0
#     ψ = deepcopy(ψ_start)
#     for i in 1:n_iter
#         α, β = lanczos(H, ψ, n_kryl)
#         E, T = LAPACK.stev!('V', α, β)
#         E0 = E[1]
#         ψ_new = similar(ψ)
#         v = ψ
#         w = ψ
#         for (i, amp) in enumerate(T[:, 1])
#             if i == 1
#                 w = H * v
#                 α = dot(w, v)
#                 Fermions.Wavefunctions.add!(w, v, -α)
#                 Fermions.Wavefunctions.add!(ψ_new, normalize(w), amp)
#             else
#                 v_old = v
#                 β = norm(w)
#                 Fermions.Wavefunctions.rmul!(w, inv(β))
#                 v = w
#                 w = H * v
#                 α = dot(w, v)
#                 Fermions.Wavefunctions.add!(w, v, -α)
#                 Fermions.Wavefunctions.add!(w, v_old, -β)
#                 Fermions.Wavefunctions.add!(ψ_new, normalize(w), amp)
#             end
#         end
#         # Normaliztion is neccessary since Lanczos is unstable.
#         # verbose && @info "Lanczos without normalization $(norm(ψ_new))"
#         normalize!(ψ_new)
#         foo = H * ψ_new
#         bar = dot(ψ_new, foo)
#         baz = dot(foo, foo)
#         prc = baz / bar^2 - 1
#         verbose && @info "iteration $i with var = $prc"
#         if abs(prc) < precision # sometimes negative variance
#             verbose && @info "ground state converged at E0 = $E0"
#             return E0, ψ_new
#         end
#         ψ = ψ_new
#     end
#     @warn "ground state not converged"
#     return E0, ψ
# end

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
