module DMFT

using Fermions
using Fermions.Bits
using Fermions.Wavefunctions
using HDF5
using LinearAlgebra

export
    # Types
    Greensfunction,

    # Functions
    get_excitation,
    mask_fe,
    mul_excitation,
    ndet,
    ndet0,
    ndet0_bit,
    ndet1,
    ndet1_bit,
    ndet2,
    ndet2_bit,
    ndet_bit,
    slater_start,
    starting_wf

include("./combinatorics.jl")
include("./mask.jl")
include("./wavefunctions.jl")
include("./greensfunction.jl")

end
