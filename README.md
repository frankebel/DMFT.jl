# DMFT

[![Build Status](https://github.com/frankebel/DMFT.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/frankebel/DMFT.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/frankebel/DMFT.jl/graph/badge.svg?token=5ACAMMA64E)](https://codecov.io/gh/frankebel/DMFT.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Source code for DMFT calculations for my master's thesis.

## Documentation

The documentation resides in `docs`.
Currently, it needs to be compiled manually

```sh
julia --project=path/to/package/docs --exec 'using Pkg; Pkg.develop(path=".."); Pkg.instantiate()'
julia --project=path/to/package/docs make.jl
```

It can then be viewed with, e.g. [LiveServer.jl](https://github.com/tlienart/LiveServer.jl)

```julia
using LiveServer
serve(dir="build")
```
