"""
Simulation for VarDP3 module.
- avoids rebuilding Mesh inside rollouts
- optional: progress printing
"""

using MDPs: IntMDP
using Random
using Statistics

if !isdefined(Main, :VarDP3)
    include("VarDP3.jl")
end
using .VarDP3

# -----------------------------
# One-step sampling
# -----------------------------
function sample_next(mdp::IntMDP, s::Int, a::Int, rng::AbstractRNG)
    nxt = getnext(mdp, s, a)
    ps  = nxt.probabilities
    u = rand(rng)
    acc = 0.0
    @inbounds for i in eachindex(ps)
        acc += ps[i]
        if u <= acc
            return nxt.states[i], nxt.rewards[i]
        end
    end
    return nxt.states[end], nxt.rewards[end]
end

# -------------------------------------------------------
# Rollout (Mesh passed in).
# A single trajectory simulator following τ-dependent 
# greedy policy table π[t+1][s,k].
# Returns discounted return ρ = ∑_{t=0}^{T-1} γ^t r_t.
# -------------------------------------------------------
function rollout_return(mdp::IntMDP, πtab::Vector{Matrix{Int}}, mesh::VarDP3.Mesh,
                        s0::Int, τ0::Float64, T::Int;
                        γ::Float64=0.9, side::Symbol=:plus,
                        rng::AbstractRNG=Random.default_rng())

    s = s0
    τ = τ0

    ρ = 0.0
    γpow = 1.0

    @inbounds for t in 0:T-1
        k = (side == :plus) ? VarDP3.hp(mesh, τ) : VarDP3.hm(mesh, τ)
        a = πtab[t+1][s, k]

        s_next, r = sample_next(mdp, s, a, rng)

        ρ += γpow * r
        γpow *= γ

        τ = (τ - r) / γ
        s = s_next
    end

    return ρ
end

# --------------------------------------------------------------------
# MC CDF curve 
# Empirical estimate of v^{π}(s0,τ) = P(ρ < τ), where π is τ-dependent
# and initialized at τ0=τ. Returns vhat(τ) for each τ in τgrid (finite subset).
# ----------------------------------------------------------------
function mc_value_curve(mdp::IntMDP, πtab::Vector{Matrix{Int}}, Xgrid::Vector{Float64},
                        s0::Int, T::Int;
                        γ::Float64=0.9, side::Symbol=:plus,
                        N::Int=2_000, seed::Int=1, stride::Int=500,
                        verbose::Bool=true)

    rng = MersenneTwister(seed)
    mesh = VarDP3.Mesh(Xgrid)  # build ONCE

    finite_idx = findall(isfinite, Xgrid)
    τgrid_full = Xgrid[finite_idx]
    τgrid = τgrid_full[1:stride:end]

    v_mc = Vector{Float64}(undef, length(τgrid))

    for i in eachindex(τgrid)
        τ = τgrid[i]
        cnt = 0
        @inbounds for n in 1:N
            ρ = rollout_return(mdp, πtab, mesh, s0, τ, T; γ=γ, side=side, rng=rng)
            cnt += (ρ < τ) ? 1 : 0
        end
        v_mc[i] = cnt / N

        if verbose && (i == 1 || i % 25 == 0 || i == length(τgrid))
            @info "MC curve progress" i= i total=length(τgrid)
        end
    end

    return τgrid, v_mc
end

# ------------------------------------------------------------------------------
# MC VaR 
# Estimate the empirical VaR_α  from N Monte Carlo rollouts of the
# return distribution produced by executing π starting from τ0 (usually τ0 = VaR^+).
# ------------------------------------------------------------------------------
function mc_var(mdp::IntMDP, πtab::Vector{Matrix{Int}}, Xgrid::Vector{Float64},
                s0::Int, τ0::Float64, T::Int, α::Float64;
                γ::Float64=0.9, side::Symbol=:plus, N::Int=50_000, seed::Int=2)

    rng  = MersenneTwister(seed)
    mesh = VarDP3.Mesh(Xgrid)  # build ONCE

    rets = Vector{Float64}(undef, N)
    @inbounds for n in 1:N
        rets[n] = rollout_return(mdp, πtab, mesh, s0, τ0, T; γ=γ, side=side, rng=rng)
    end
    sort!(rets)

    idx = clamp(ceil(Int, α*N), 1, N)
    return rets[idx]
end
































# """
# Simulation utilities for VarDP3 module.
# Includes sampling, rollout, and Monte Carlo estimation functions.
# """

# using MDPs: IntMDP
# using Random
# using Statistics

# # Assume VarDP3 is already loaded; if not, include it
# if !isdefined(Main, :VarDP3)
#     include("VarDP3.jl")
# end
# using .VarDP3

# """
#     sample_next(mdp::IntMDP, s::Int, a::Int, rng::AbstractRNG)

# Sample one transition outcome from IntMDP kernel at (s,a).
# Returns (s_next, r).
# """
# function sample_next(mdp::IntMDP, s::Int, a::Int, rng::AbstractRNG)
#     nxt = getnext(mdp, s, a)
#     ps  = nxt.probabilities
#     u = rand(rng)
#     acc = 0.0
#     @inbounds for i in eachindex(ps)
#         acc += ps[i]
#         if u <= acc
#             return nxt.states[i], nxt.rewards[i]
#         end
#     end
#     return nxt.states[end], nxt.rewards[end]
# end

# """
#     rollout_return(mdp::IntMDP, πtab::Vector{Matrix{Int}}, Xgrid::Vector{Float64},
#                    s0::Int, τ0::Float64, T::Int; γ::Float64=0.9, side::Symbol=:plus, rng::AbstractRNG=Random.default_rng())

# Roll out ONE trajectory following a τ-dependent greedy policy table π[t+1][s,k].
# Returns discounted return ρ = ∑_{t=0}^{T-1} γ^t r_t.
# """
# function rollout_return(mdp::IntMDP, πtab::Vector{Matrix{Int}}, Xgrid::Vector{Float64},
#                         s0::Int, τ0::Float64, T::Int;
#                         γ::Float64=0.9, side::Symbol=:plus, rng::AbstractRNG=Random.default_rng())

#     mesh = VarDP3.Mesh(Xgrid)
#     s = s0
#     τ = τ0

#     ρ = 0.0
#     γpow = 1.0  # = γ^t, updated each step

#     for t in 0:T-1
#         k = (side == :plus) ? VarDP3.hp(mesh, τ) : VarDP3.hm(mesh, τ)
#         a = πtab[t+1][s, k]

#         s_next, r = sample_next(mdp, s, a, rng)

#         ρ += γpow * r
#         γpow *= γ

#         # τ-update consistent with DP recursion: ρ = r + γρ'  < τ  ⇔  ρ' < (τ-r)/γ
#         τ = (τ - r) / γ
#         s = s_next
#     end

#     return ρ
# end

# """
#     mc_value_curve(mdp::IntMDP, πtab::Vector{Matrix{Int}}, Xgrid::Vector{Float64},
#                    s0::Int, T::Int; γ::Float64=0.9, side::Symbol=:plus,
#                    N::Int=20_000, seed::Int=1, stride::Int=10)

# Empirical estimate of v^{π}(s0,τ) = P(ρ < τ), where π is τ-dependent and initialized at τ0=τ.
# Returns vhat(τ) for each τ in τgrid (finite subset).
# """
# function mc_value_curve(mdp::IntMDP, πtab::Vector{Matrix{Int}}, Xgrid::Vector{Float64},
#                         s0::Int, T::Int;
#                         γ::Float64=0.9, side::Symbol=:plus,
#                         N::Int=20_000, seed::Int=1, stride::Int=10)

#     rng = MersenneTwister(seed)

#     finite_idx = findall(isfinite, Xgrid)
#     τgrid_full = Xgrid[finite_idx]

#     # optional: evaluate only every `stride`-th τ to reduce cost
#     τgrid = τgrid_full[1:stride:end]

#     v_mc = similar(τgrid)

#     for (i, τ) in pairs(τgrid)
#         cnt = 0
#         @inbounds for n in 1:N
#             ρ = rollout_return(mdp, πtab, Xgrid, s0, τ, T; γ=γ, side=side, rng=rng)
#             cnt += (ρ < τ) ? 1 : 0
#         end
#         v_mc[i] = cnt / N
#     end

#     return τgrid, v_mc
# end

# """
#     mc_var(mdp::IntMDP, πtab::Vector{Matrix{Int}}, Xgrid::Vector{Float64},
#            s0::Int, τ0::Float64, T::Int, α::Float64;
#            γ::Float64=0.9, side::Symbol=:plus, N::Int=50_000, seed::Int=2)

# Estimate VaRα of the return distribution produced by executing π starting from τ0 (usually τ0 = VaR^+).
# VaR here is the left-quantile: sup{τ : P(ρ<τ) ≤ α}. For samples, we return the empirical α-quantile.
# """
# function mc_var(mdp::IntMDP, πtab::Vector{Matrix{Int}}, Xgrid::Vector{Float64},
#                 s0::Int, τ0::Float64, T::Int, α::Float64;
#                 γ::Float64=0.9, side::Symbol=:plus, N::Int=50_000, seed::Int=2)

#     rng = MersenneTwister(seed)
#     rets = Vector{Float64}(undef, N)
#     @inbounds for n in 1:N
#         rets[n] = rollout_return(mdp, πtab, Xgrid, s0, τ0, T; γ=γ, side=side, rng=rng)
#     end
#     sort!(rets)

#     # empirical left-quantile (smallest τ with CDF >= α)
#     idx = clamp(ceil(Int, α*N), 1, N)
#     return rets[idx]
# end
