module VarDP3

using MDPs
using MDPs: IntMDP, state_count, actions, getnext, load_mdp
using CSV
using Random
using Statistics


# -------------------------
# Mesh + Approximation indices
# -------------------------

struct Mesh
    xs::Vector{Float64}
    function Mesh(xs::AbstractVector{<:Real})
        v = collect(float.(xs))
        @assert issorted(v)
        @assert first(v) == -Inf
        @assert last(v)  ==  Inf
        new(v)
    end
end

@inline function hm(mesh::Mesh, x::Float64)
    i = searchsortedlast(mesh.xs, x)   # largest i with xs[i] ≤ x
    @assert i ≥ 1
    return i
end

@inline function hp(mesh::Mesh, x::Float64)
    i = searchsortedfirst(mesh.xs, x)  # smallest i with xs[i] ≥ x
    @assert i ≤ length(mesh.xs)
    return i
end

function load_intmdp(path::AbstractString; idoutcome=nothing, docompress=false)
    return load_mdp(
        CSV.File(path);
        idoutcome = idoutcome,
        zerobased = false,
        docompress = docompress,
    )
end

function max_abs_reward(mdp::IntMDP)
    max_r = 0.0
    S = state_count(mdp)
    for s in 1:S
        for a in actions(mdp, s)
            nxt = getnext(mdp, s, a)
            for r in nxt.rewards
                max_r = max(max_r, abs(r))
            end
        end
    end
    return max_r
end

# -------------------------------------------------------
# Backward DP for τ-augmented probability value functions
#   v_t(s,τ) = inf_{π} P( ρ_t < τ | S_t=s )
#
# Approximation:
#   side = :minus uses h^-(τ - r / γ) 
#   side = :plus  uses h^+(τ - r / γ) 
# -------------------------------------------------------

function vi(mdp::IntMDP,
            X::Vector{Vector{Float64}},
            T::Int;
            side::Symbol = :minus,
            γ::Float64 = 0.9)

    S = state_count(mdp)
    @assert length(X) == T + 1

    V = Vector{Matrix{Float64}}(undef, T + 1)
    π = Vector{Matrix{Int}}(undef, T)

    # Terminal condition at t = T: v_T(s,τ) = 1{ τ > 0 }
    XT = X[T+1]
    KT = length(XT)
    V[T+1] = zeros(Float64, S, KT)
    for s in 1:S, k in 1:KT
        τ = XT[k]
        V[T+1][s,k] = (τ > 0.0) ? 1.0 : 0.0
    end

    # Backward recursion for t = T-1, ..., 0
    for t in (T-1):-1:0
        X_cur = X[t+1]      # grid for time t
        K     = length(X_cur)
        V[t+1] = zeros(Float64, S, K)
        π[t+1] = zeros(Int, S, K)

        mesh_next = Mesh(X[t+2])  # grid for time t+1
        hm_idx(x::Float64) = hm(mesh_next, x)
        hp_idx(x::Float64) = hp(mesh_next, x)

        for s in 1:S, k in 1:K
            τ = X_cur[k]

            best_val = +Inf
            best_a   = first(actions(mdp, s))

            for a in actions(mdp, s)
                nxt = getnext(mdp, s, a)
                states_ns = nxt.states
                probs     = nxt.probabilities
                rewards   = nxt.rewards

                exp_val = 0.0
                @inbounds for i in eachindex(states_ns, probs, rewards)
                    ns = states_ns[i]
                    p  = probs[i]
                    p == 0.0 && continue

                    r_sa = rewards[i]
                    
                    nτ = (τ - r_sa) / γ   #  discounted τ-update

                    j = (side == :plus) ? hp_idx(nτ) : hm_idx(nτ)
                    exp_val += p * V[t+2][ns, j]  # v_{t+1}(ns, approx nτ)
                end

                if exp_val < best_val
                    best_val = exp_val
                    best_a   = a
                end
            end

            V[t+1][s,k] = best_val
            π[t+1][s,k] = best_a
        end
    end

    return V, π
end

# ---------------------------------------
# Left-quantile VaR extraction on a grid:
#   VaRα(s) = sup{ τ ∈ grid : v_t(s,τ) ≤ α }
# ---------------------------------------
function var(V::Vector{Matrix{Float64}},
                  X::Vector{Vector{Float64}},
                  t::Int,
                  s::Int,
                  α::Float64)

    grid = X[t+1]
    vals = @view V[t+1][s, :]

    # find largest τ with v ≤ α
    last_ok = nothing
    for k in eachindex(grid)
        if vals[k] ≤ α
            last_ok = grid[k]
        else
            break
        end
    end
    return last_ok
end

end # module



######



