# Various hybridization functions.

# For the Bethe lattice take the  Green's function and rescale by D^/4
# as Δ(ω) = D^2/4 G(ω).

"""
    hybridization_function_bethe_simple(n_bath::Int, D::Real=1.0)

Return the [`Pole`](@ref) representation of the semicircular density of states
with half-bandwidth `D` on `n_bath` poles.

Poles are found by diagonalizing a tridiagonal matrix with hopping ``t=D/2``.

See also: [`hybridization_function_bethe_equal_weight`](@ref).
"""
function hybridization_function_bethe_simple(n_bath::Int, D::Real=1.0)
    # Take Green's function and rescale weights by D/2.
    Δ = greens_function_bethe_simple(n_bath, D)
    Δ.b .*= D / 2
    return Δ
end

"""
    hybridization_bethe_equal_weight(n_bath::Int, D::Real=1.0)

Return the [`Pole`](@ref) representation of the semicircular density of states
with half-bandwidth `D` on `n_bath` poles.

Each site has the same hybridization ``V^2``.

See also: [`hybridization_function_bethe_simple`](@ref).
"""
function hybridization_function_bethe_equal_weight(n_bath::Int, D::Real=1.0)
    # Take Green's function and rescale weights by D/2.
    Δ = greens_function_bethe_equal_weight(n_bath, D)
    Δ.b .*= D / 2
    return Δ
end
