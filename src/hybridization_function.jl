# Various hybridization functions.

# For the Bethe lattice take the  Green's function and rescale by D^/4
# as Δ(ω) = D^2/4 G(ω).

"""
    hybridization_function_bethe_analytic(z::Number, D::Real=1.0)
    hybridization_function_bethe_analytic(Z::AbstractVector{<:Number}, D::Real=1.0)

Calculate the hybrid function for a Bethe lattice
given a frequency `z` in the upper complex plane,
and half-bandwidth `D`.

```math
Δ(z) = \\frac{1}{2} (z - \\sign(\\mathrm{Re}(z)) \\sqrt{z^2 - D^2})
```

with ``\\sign(0) = \\sign(0^±)``.
"""
function hybridization_function_bethe_analytic(z::Number, D::Real=1.0)
    return greens_function_bethe_analytic(z, D) * D^2 / 4
end

function hybridization_function_bethe_analytic(Z::AbstractVector{<:Number}, D::Real=1.0)
    return map(z -> hybridization_function_bethe_analytic(z, D), Z)
end

"""
    hybridization_function_bethe_simple(n_bath::Int, D::Real=1.0)

Return the [`Pole`](@ref) representation of the semicircular density of states
with half-bandwidth `D` on `n_bath` poles.

Poles are found by diagonalizing a tridiagonal matrix with hopping ``t=D/2``.

See also:
[`greens_function_bethe_grid`](@ref),
[`hybridization_function_bethe_equal_weight`](@ref).
"""
function hybridization_function_bethe_simple(n_bath::Int, D::Real=1.0)
    # Take Green's function and rescale weights by D/2.
    Δ = greens_function_bethe_simple(n_bath, D)
    Δ.b .*= D / 2
    return Δ
end

"""
    hybridization_function_bethe_grid(grid::AbstractVector{<:Real}, D::Real=1.0)

Return the [`Pole`](@ref) representation of the semicircular density of states
with half-bandwidth `D` with poles given in `grid`.

See also:
[`hybridization_function_bethe_simple`](@ref),
[`hybridization_function_bethe_equal_weight`](@ref).
"""
function hybridization_function_bethe_grid(grid::AbstractVector{<:Real}, D::Real=1.0)
    Δ = greens_function_bethe_grid(grid, D)
    Δ.b .*= D / 2
    return Δ
end

"""
    hybridization_function_bethe_grid_hubbard3(
    grid::AbstractVector{<:Real}, U::Real=0.0, D::Real=1.0
)

Return the [`Pole`](@ref) representation of the Hubbard III approximation
with half-bandwidth `D` and poles given in `grid`.

Created using two semicircles at ``±U/2``.
"""
function hybridization_function_bethe_grid_hubbard3(
    grid::AbstractVector{<:Real}, U::Real=0.0, D::Real=1.0
)
    Δ = greens_function_bethe_grid_hubbard3(grid, U, D)
    Δ.b .*= D / 2
    return Δ
end

"""
    hybridization_bethe_equal_weight(n_bath::Int, D::Real=1.0)

Return the [`Pole`](@ref) representation of the semicircular density of states
with half-bandwidth `D` on `n_bath` poles.

Each site has the same hybridization ``V^2``.

See also:
[`hybridization_function_bethe_simple`](@ref),
[`hybridization_function_bethe_grid`](@ref).
"""
function hybridization_function_bethe_equal_weight(n_bath::Int, D::Real=1.0)
    # Take Green's function and rescale weights by D/2.
    Δ = greens_function_bethe_equal_weight(n_bath, D)
    Δ.b .*= D / 2
    return Δ
end
