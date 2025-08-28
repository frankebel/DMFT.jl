"""
    lanczos!(
        states::Vector{CWF},
        a::Vector{Float64},
        b::Vector{Float64},
        H::CIOperator,
        q::CWF,
        M::Integer,
    ) where {CWF<:CIWavefunction}

Calculate `M` Lanczos steps
storing coefficients in `a`, `b` and intermediate states in `states`.
"""
function lanczos!(
    a::Vector{Float64},
    b::Vector{Float64},
    states::Vector{CWF},
    H::CIOperator,
    q::CWF,
    M::Integer,
) where {CWF<:CIWavefunction}
    # check input
    length(states) == M + 1 || throw(ArgumentError("length of states must be M+1")) # need space for one more
    length(a) == M || throw(ArgumentError("length of a must be M-1"))
    length(b) == M - 1 || throw(ArgumentError("length of a must be M"))
    M >= 1 || throw(ArgumentError("M must be >= 1"))
    # first step
    copy!(states[1], q)
    mul!(states[2], H, states[1])
    a[1] = states[1] ⋅ states[2]
    axpy!(-a[1], states[1], states[2])
    for j in 2:M
        b[j - 1] = norm(states[j])
        rmul!(states[j], inv(b[j - 1])) # normalize
        # new state
        mul!(states[j + 1], H, states[j])
        axpy!(-b[j - 1], states[j - 1], states[j + 1]) # orthogonalize against 2nd previous
        a[j] = states[j] ⋅ states[j + 1] # overlap to previous
        axpy!(-a[j], states[j], states[j + 1]) # orthogonalize against previous
    end
    return nothing
end
