module DMFT

using Distributions
using Fermions
using Fermions.Bits
using Fermions.Wavefunctions
using HDF5
using LinearAlgebra

export
    # Types
    Greensfunction,
    Hybridizationfunction,

    # Functions
    get_excitation,
    get_hyb,
    get_hyb_equal,
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
    orgtr!,
    slater_start,
    starting_wf,
    sytrd!

include("combinatorics.jl")
include("mask.jl")
include("wavefunctions.jl")
include("greensfunction.jl")
include("sytrd.jl")

end
