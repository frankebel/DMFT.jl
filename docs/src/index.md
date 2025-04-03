```@meta
CurrentModule = DMFT
```

# Real frequency DMFT solver

`DMFT.jl` can be used to calculate a single Hubbard band using DMFT on the real frequency axis.

Given a single impurity Anderson Hamiltonian (SIAM) $H=H_0 + H_\mathrm{int}$ with

```math
H_0
= \sum_\sigma \epsilon_d d_\sigma^\dagger d_\sigma
+ \sum_{k\sigma} \epsilon_{k\sigma} c_{k\sigma}^\dagger c_{k\sigma}
+ \sum_{k\sigma} (V_{k\sigma} d_\sigma^\dagger c_{k\sigma}
  + V_{k\sigma}^* c_{k\sigma}^\dagger d_\sigma),
```

and interaction

```math
H_\mathrm{int} = U d_\uparrow^\dagger d_\downarrow^\dagger d_\downarrow d_\uparrow,
```

it will calculate the retarded Green's function

```math
\begin{aligned}
G(t) &= - \mathrm{i} \Theta(t) \langle \{ d_\alpha^\dagger(t), d_\alpha \} \rangle \\
G(z) &= \int_{-\infty}^\infty G(t) \mathrm{e}^{\mathrm{i}zt} \mathrm{d}t.
\end{aligned}
```

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

If you plan to modify it, we first clone the package locally and use
[`Pkg.develop`](https://pkgdocs.julialang.org/v1/api/#Pkg.develop)

```sh
git clone git@github.com:frankebel/DMFT.jl path/to/local/package
julia --project=path/to/project --eval 'using Pkg; Pkg.develop(path=path/to/local/package)'
```

If the package is installed, you can run all tests with

```julia
julia --project=path/to/project --eval 'using Pkg; Pkg.test("DMFT")'
```

## Modules

The main module is called `DMFT` and can be put into the namespace by

```julia
using DMFT
```

There are also submodules

- `DMFT.Combinatorics`
- `DMFT.Debug`
- `DMFT.ED`

which are not necessary to be called in 99 % of all cases.
