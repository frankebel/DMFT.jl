# Manipulation of masks.

"""
    mask_fe(slaterdet::Type{<:Unsigned}, nbit::Int, nfilled::Int, nempty::Int)

Return masks of filled/empty sites in vector of `CIWavefunction`.

# Arguments
- `slaterdet`: Type of the mask.
- `nbit`: Number of sites in bit component of `CIWavefunction`.
- `nfilled`: Number of filled sites in vector of `CIWavefunction`.
- `nempty`: Number of empty sites in vector of `CIWavefunction`.
"""
function mask_fe(slaterdet::Type{<:Unsigned}, nbit::Int, nfilled::Int, nempty::Int)
    # test input
    nsites = nbit + nfilled + nempty
    2 * nsites <= bitsize(slaterdet) || throw(ArgumentError("insufficient bitsize"))
    nbit >= 0 || throw(ArgumentError("`nbit` must be >= 0"))
    nfilled >= 0 || throw(ArgumentError("`nfilled` must be >= 0"))
    nempty >= 0 || throw(ArgumentError("`nempty` must be >= 0"))

    m_filled = least_significant(slaterdet, nfilled) << nbit
    m_filled |= least_significant(slaterdet, nfilled) << (nbit + nsites)
    m_empty = least_significant(slaterdet, nempty) << (nbit + nfilled)
    m_empty |= least_significant(slaterdet, nempty) << (nbit + nfilled + nsites)
    return m_filled, m_empty
end

"""
    get_excitation(s::S, m_filled::S, m_empty::S) where {S<:Unsigned}

Return excitation of Slater determinant `s`.

`m_filled`, `m_empty` are the masks of default filled/empty sites
in vector of `CIWavefunction`.
"""
@inline function get_excitation(s::S, m_filled::S, m_empty::S) where {S<:Unsigned}
    return count_ones((s & (m_filled | m_empty) ⊻ m_filled))
end

"""
    excitation!(
        ψ::Wavefunction, m_filled::S, m_empty::S, excitation::Int
    ) where {S<:Unsigned}

In Wavefunction `ψ` remove entries higher than `excitation`.

`m_filled`, `m_empty` are the masks of default filled/empty sites
in vector of `CIWavefunction`.
"""
function excitation!(
    ψ::Wavefunction, m_filled::S, m_empty::S, excitation::Int
) where {S<:Unsigned}
    excitation >= 0 || throw(ArgumentError("`excitation` >= 0"))
    for k in keys(ψ)
        get_excitation(k, m_filled, m_empty) <= excitation || delete!(ψ, k)
    end
end

"""
    slater_start(
        slaterdet::Type{<:Unsigned},
        bi::UInt8,
        nfilled_bit::Int,
        nempty_bit::Int,
        nfilled::Int,
        nempty::Int,
    )

Return Slater determinant with impurity and bath site b filling given by `bi`,
valence bath sites filled fully.

For impurity and b the last four bits of `bi` are taken for occupation.
If `0bwxyz` is given the occupation are taken as:

- `n[b, 2] = w`
- `n[i, 2] = x`
- `n[b, 1] = y`
- `n[i, 1] = z`

E.g. for opposite occupation: `0b1001`, `0b0110`.
"""
function slater_start(
    slaterdet::Type{<:Unsigned},
    bi::UInt8,
    nfilled_bit::Int,
    nempty_bit::Int,
    nfilled::Int,
    nempty::Int,
)
    # test input
    nbit = 2 + nfilled_bit + nempty_bit
    nvector = nfilled + nempty
    nsites = nbit + nfilled + nempty
    2 * nsites <= bitsize(slaterdet) || throw(ArgumentError("insufficient bitsize"))
    nfilled_bit >= 0 || throw(ArgumentError("`nfilled_bit` must be >= 0"))
    nempty_bit >= 0 || throw(ArgumentError("`nempty_bit` must be >= 0"))
    nfilled >= 0 || throw(ArgumentError("`nfilled` must be >= 0"))
    nempty >= 0 || throw(ArgumentError("`nempty` must be >= 0"))

    b2 = (0b1000 & bi) >> 3
    i2 = (0b0100 & bi) >> 2
    b1 = (0b0010 & bi) >> 1
    i1 = 0b0001 & bi
    result = (b2 == 0 ? zero(slaterdet) : one(slaterdet)) << (nsites + 1)
    result |= (i2 == 0 ? zero(slaterdet) : one(slaterdet)) << nsites
    result |= (b1 == 0 ? zero(slaterdet) : one(slaterdet)) << 1
    result |= (i1 == 0 ? zero(slaterdet) : one(slaterdet))
    m_bit, _ = mask_fe(slaterdet, 2, nfilled_bit, nempty_bit + nvector)
    m_vector, _ = mask_fe(slaterdet, nbit, nfilled, nempty)
    result |= m_vector
    result |= m_bit
    return result
end

# Colored print of Slater determinant
function colorprint(
    io::IO,
    s::Unsigned,
    nfilled_bit::Int=0,
    nempty_bit::Int=0,
    nfilled::Int=0,
    nempty::Int=0,
)
    # test input
    nsites = 2 + nfilled_bit + nempty_bit + nfilled + nempty
    2 * nsites <= bitsize(s) || throw(ArgumentError("insufficient bitsize"))
    nfilled_bit >= 0 || throw(ArgumentError("`nfilled_bit` must be >= 0"))
    nempty_bit >= 0 || throw(ArgumentError("`nempty_bit` must be >= 0"))
    nfilled >= 0 || throw(ArgumentError("`nfilled` must be >= 0"))
    nempty >= 0 || throw(ArgumentError("`nempty` must be >= 0"))

    bs = bitstring(s)
    i = length(bs) - 2 * nsites
    print(io, bs[1:i]) # overflowing bits
    # (n, 2)
    printstyled(io, bs[(i + 1):(i += nempty)]; color=:blue)
    printstyled(io, bs[(i + 1):(i += nfilled)]; color=:cyan)
    printstyled(io, bs[(i + 1):(i += nempty_bit)]; color=:green)
    printstyled(io, bs[(i + 1):(i += nfilled_bit)]; color=:magenta)
    printstyled(io, bs[(i + 1):(i += 2)]; color=:red)
    # (n, 1)
    printstyled(io, bs[(i + 1):(i += nempty)]; color=:blue)
    printstyled(io, bs[(i + 1):(i += nfilled)]; color=:cyan)
    printstyled(io, bs[(i + 1):(i += nempty_bit)]; color=:green)
    printstyled(io, bs[(i + 1):(i += nfilled_bit)]; color=:magenta)
    printstyled(io, bs[(i + 1):(i += 2)]; color=:red)
    return nothing
end

# default stdout
function colorprint(
    s::Unsigned, nfilled_bit::Int=0, nempty_bit::Int=0, nfilled::Int=0, nempty::Int=0
)
    return colorprint(stdout, s, nfilled_bit, nempty_bit, nfilled, nempty)
end
