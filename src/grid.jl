# creation, manipulation of grids

"""
    grid_log(ω_max::Real, Λ::Real, n::Int)

Create `n` sorted poles on a logarithmic grid with highest value `ω_max`.

The poles have constant ratio

```math
\\frac{ω_i}{ω_{i+1}} =  \\frac{1}{Λ}.
```
"""
function grid_log(ω_max::Real, Λ::Real, n::Int)
    ω_max >= 0 || throw(ArgumentError("negative frequency ω_max"))
    Λ > 1 || throw(ArgumentError("invalid discretization parameter Λ"))
    n >= 1 || throw(ArgumentError("invalid number of points n"))
    result = map(i -> Λ^(-i) * ω_max, 0.0:(n - 1))
    reverse!(result) # from small to big values
    return result
end
