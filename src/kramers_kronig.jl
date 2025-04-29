# PERF: use convolution theorem and FFT for higher performance

"""
    realKK(A::V, ω::V) where {V<:AbstractVector{<:Real}}

Calculate the real part of a function given its imaginary part `A`
using Kramers-Kronig relations

```math
\\mathrm{Re}~A(ω) = \\frac{1}{π} 𝒫 ∫_{-∞}^∞ \\frac{A(ω')}{ω' - ω} \\mathrm{d}ω'.
```

See also [`imagKK`](@ref).
"""
function realKK(A::V, ω::V) where {V<:AbstractVector{<:Real}}
    length(A) == length(ω) || throw(DimensionMismatch("length mismatch"))
    result = zero(A)
    # all frequencies ω
    for i in eachindex(A)
        # all frequencies ω'
        for j in eachindex(ω)
            i == j && continue # exclude ω' == ω
            # trapezoidal rule
            j == 1 && continue # skip first index
            result[i] += 0.5 * (A[j - 1] + A[j]) / (ω[j] - ω[i]) * (ω[j] - ω[j - 1])
        end
    end

    result .*= 1 / π
    return result
end

"""
    imagKK(A::V, ω::V) where {V<:AbstractVector{<:Real}}

Calculate the imaginary part of a function given its real part `A`
using Kramers-Kronig relations

```math
\\mathrm{Im}~A(ω) = -\\frac{1}{π} 𝒫 ∫_{-∞}^∞ \\frac{A(ω')}{ω' - ω} \\mathrm{d}ω'.
```

See also [`realKK`](@ref).
"""
function imagKK(A::V, ω::V) where {V<:AbstractVector{<:Real}}
    return -realKK(A, ω)
end
