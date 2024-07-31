module DMFT

using Fermions
using Fermions.Bits
using Fermions.Wavefunctions

export
    # Functions
    get_excitation,
    mask_fe,
    ndet,
    ndet0,
    ndet0_bit,
    ndet1,
    ndet1_bit,
    ndet2,
    ndet2_bit,
    ndet_bit,
    slater_start

include("./combinatorics.jl")
include("./mask.jl")

end
