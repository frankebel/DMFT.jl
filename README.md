# DMFT

[![Build Status](https://github.com/frankebel/DMFT.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/frankebel/DMFT.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/frankebel/DMFT.jl/graph/badge.svg?token=5ACAMMA64E)](https://codecov.io/gh/frankebel/DMFT.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

Source code for DMFT calculations for my master's thesis.

## Installation

As the package is not inside the [General registry](https://github.com/JuliaRegistries/General),
it needs to added
[manually](https://pkgdocs.julialang.org/v1/managing-packages/#Adding-unregistered-packages).
The package is not public yet, thus an authentication method is necessary when accessing it.
We recommend to set up an SSH key pair and assume the existence of one for the following commands.

If you do not plan to modify the package,
you can add it by running the following commands in the shell

```sh
export JULIA_PKG_USE_CLI_GIT="true"
julia --project=path/to/project --eval 'using Pkg; Pkg.add(url="git@github.com:frankebel/DMFT.jl")'
```

If you plan to modify it, clone the package locally first and use
[`Pkg.develop`](https://pkgdocs.julialang.org/v1/api/#Pkg.develop)

```sh
git clone git@github.com:frankebel/DMFT.jl path/to/local/package
julia --project=path/to/project --eval 'using Pkg; Pkg.develop(path=path/to/local/package)'
```

If the package is installed, you can run all tests with

```julia
julia --project=path/to/project --eval 'using Pkg; Pkg.test("DMFT")'
```

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
