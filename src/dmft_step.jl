"""
    dmft_step(
        Δ0::Greensfunction{<:T,<:AbstractVector{<:T}},
        Δ::Greensfunction{<:T,<:AbstractVector{<:T}},
        H_int::Op,
        μ::T,
        ϵ_imp::T,
        Z::AbstractVector{<:Complex},
        n_v_bit::Int,
        n_c_bit::Int,
        e::Int,
        O::AbstractVector{<:Op},
        n_kryl::Int,
        n_kryl_gs::Int,
        n_dis::Int,
        η::T,
        improved_self_energy::Bool,
    ) where {T<:Real,Op<:Operator}

Calculate a single step of DMFT.

Returns
- `G_plus`: for positive freqeuncies
- `G_minus`: for negative freqeuncies
- `Δ_new`: new hybridization function
- `Δ_grid`: new hybridization function on a grid
"""
function dmft_step(
    Δ0::Greensfunction{<:T,<:AbstractVector{<:T}},
    Δ::Greensfunction{<:T,<:AbstractVector{<:T}},
    H_int::Op,
    μ::T,
    ϵ_imp::T,
    Z::AbstractVector{<:Complex},
    n_v_bit::Int,
    n_c_bit::Int,
    e::Int,
    O::AbstractVector{<:Op},
    n_kryl::Int,
    n_kryl_gs::Int,
    n_dis::Int,
    η::T,
    improved_self_energy::Bool,
) where {T<:Real,Op<:Operator}
    G_plus, G_minus, Σ_H = solve_impurity(
        Δ, H_int, ϵ_imp, n_v_bit, n_c_bit, e, n_kryl_gs, n_kryl, O
    )
    if improved_self_energy
        Σ = self_energy_improved(G_plus, G_minus, Z, Σ_H)
    else
        Σ = self_energy(G_plus, G_minus, Z)
    end
    Δ_grid = update_weiss_field(Δ0, μ, Z, Σ)
    Δ_new = equal_weight_discretization(-imag(Δ_grid), real(Z), η, n_dis)
    return G_plus, G_minus, Δ_new, Δ_grid
end

"""
    self_energy(
        G_plus::GF, G_minus::GF, Z::AbstractVector{<:Complex}
    ) where {GF<:Greensfunction{<:Real,<:AbstractMatrix{<:Number}}}

Calculate self-energy as ``Σ(Z) = F(Z) G^{-1}(Z)``.
"""
function self_energy(
    G_plus::GF, G_minus::GF, Z::AbstractVector{<:Complex}
) where {GF<:Greensfunction{<:Real,<:AbstractMatrix{<:Number}}}
    gp = G_plus(Z)
    gm = G_minus(Z)
    G = map(g -> g[1, 1], gm) .+ map(g -> g[1, 1], gp)
    F = map(g -> g[1, 2], gm) .+ map(g -> g[2, 1], gp)
    return F ./ G
end

"""
    self_energy_improved(
        G_plus::GF, G_minus::GF, Z::AbstractVector{<:Complex}, Σ_H::Real
    ) where {GF<:Greensfunction{<:Real,<:AbstractMatrix{<:Number}}}

Calculate self-energy as ``Σ(Z) = Σ^H + I(Z) - F^L(Z) G^{-1}(Z) F^R(Z)``.
"""
function self_energy_improved(
    G_plus::GF, G_minus::GF, Z::AbstractVector{<:Complex}, Σ_H::Real
) where {GF<:Greensfunction{<:Real,<:AbstractMatrix{<:Number}}}
    gp = G_plus(Z)
    gm = G_minus(Z)
    G = map(g -> g[1, 1], gm) .+ map(g -> g[1, 1], gp)
    F_R = map(g -> g[1, 2], gm) .+ map(g -> g[2, 1], gp)
    F_L = map(g -> g[1, 2], gm) .+ map(g -> g[2, 1], gp)
    I = map(g -> g[2, 2], gm) .+ map(g -> g[2, 2], gp)
    return Σ_H .+ I - F_L ./ G .* F_R
end

"""
    update_weiss_field(
        Δ0::Greensfunction,
        μ::Real,
        Z::AbstractVector{<:Complex},
        Σ::AbstractVector{<:Complex},
    )

Calculate the new Weiss field from the lattice hybridization `Δ0` and
the impurity self energy `Σ` on grid `Z` (with small imaginary part).
"""
update_weiss_field(
    Δ0::Greensfunction, μ::Real, Z::AbstractVector{<:Complex}, Σ::AbstractVector{<:Complex}
) = Δ0(Z .+ μ - Σ)

"""
    equal_weight_discretization(
        imΔ::AbstractVector{<:Real}, w::AbstractVector{<:Real}, η::Real, n::Int
    )

Discretize the Weiss field on `n` poles, such that each pole has equal weight.

`imΔ` is the negative imaginary part of the Weiss field.
Assumes that `w` is an equidistant grid.
Assumes that `w` has odd number of values.
Assumes that `w` is a symmetric interval.
Assumes that `imΔ` has only semipositive values.
"""
function equal_weight_discretization(
    imΔ::AbstractVector{<:Real}, w::AbstractVector{<:Real}, η::Real, n::Int
)
    n >= 3 || throw(ArgumentError("at least 3 poles necessary"))
    isodd(n) || throw(ArgumentError("need odd `n`"))

    N = length(w)
    dw = w[2] - w[1]
    M = cld(N, 2)
    total = sum(imΔ) * dw
    v = total / n
    v0 = imΔ[M] * dw
    i = 1
    # pole at w = 0 : weight = min(v, weight in [-η,η])
    while w[M + i] <= η && M + i <= N && v0 < v
        v0 += imΔ[M + i] * dw + imΔ[M - i] * dw
        i += 1
    end
    # calculate total remainung weight for positive freqeuncies:
    wght_right = sum(imΔ[(M + i):end]) * dw
    # calculate total remainung weight for negative freqeuncies:
    wght_left = sum(imΔ[1:(M - i)]) * dw
    j = i # remember i for later

    # discretize for positive frequencies:
    P_plus = Float64[]
    V_plus = Float64[]
    vp = 0.0
    pp = 0.0
    v = wght_right / ((n - 1) ÷ 2)
    partial = 0.0
    while partial < wght_right && M + i <= N
        while vp < v && M + i <= N # accumulate weight until v is exceeded or end of grid
            vp += imΔ[M + i] * dw
            pp += w[M + i] * imΔ[M + i] * dw
            i += 1
        end
        δv = vp - v
        if δv > 0 # if vp exceeds v carry over the difference for next pole
            pp -= w[M + i - 1] * δv
            partial += v
            push!(V_plus, sqrt(v / π))
            push!(P_plus, pp / v)
            vp = δv
            pp = w[M + i - 1] * δv
        elseif vp > 0
            partial += vp
            push!(V_plus, sqrt(vp / π))
            push!(P_plus, pp / vp)
            vp = 0.0
            pp = 0.0
        else
            @error "eql_wght_discrt: pole without weight at M+i = $((M+i,N))"
        end
    end

    # discretize for negative frequencies:
    P_minus = Float64[]
    V_minus = Float64[]
    vm = 0.0
    pm = 0.0
    v = wght_left / ((n - 1) ÷ 2)
    partial = 0.0
    while partial < wght_left && M - j > 0
        while vm < v && M - j > 0 # accumulate weight until v is exceeded or end of grid
            vm += imΔ[M - j] * dw
            pm += w[M - j] * imΔ[M - j] * dw
            j += 1
        end
        δv = vm - v
        if δv > 0 # if vm exceeds v carry over the difference for next pole
            pm -= w[M - j + 1] * δv
            partial += v
            push!(V_minus, sqrt(v / π))
            push!(P_minus, pm / v)
            vm = δv
            pm = w[M - j + 1] * δv
        elseif vm > 0
            partial += vm
            push!(V_minus, sqrt(vm / π))
            push!(P_minus, pm / vm)
            vm = 0.0
            pm = 0.0
        else
            @error "eql_wght_discrt: pole without weight at M-j = $(M-j)"
        end
    end
    a = [reverse!(P_minus); 0; P_plus]
    b = [reverse!(V_minus); sqrt(v0 / π); V_plus]
    return Greensfunction(a, b)
end
