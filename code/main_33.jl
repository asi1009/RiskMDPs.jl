###############################################################################
# run_vardp_with_mc.jl
#
# Full updated script:
# - Runs VarDP3 DP bounds (V_plus/V_minus) and greedy policies (π_plus/π_minus)
# - Plots DP bounds v^- and v^+
# - Overlays Monte Carlo CDF curve for greedy policy induced by v^+ on Figure 1
# - Computes VaR at alpha = 0.25 for:
#     * DP VaR^+ (from V_plus)
#     * MC VaR under greedy(π_plus) starting at τ0 = DP VaR^+
###############################################################################

using MDPs
using CSV
using Printf
using Plots
using Random
using Statistics

# -----------------------------
# Load modules
# -----------------------------
include(joinpath(@__DIR__, "VarDP3.jl"))
using .VarDP3

include(joinpath(@__DIR__, "simulate.jl"))

# -----------------------------
# Load MDP
# -----------------------------
mdp_path = joinpath(@__DIR__, "..", "data", "machine.csv")
mdp = VarDP3.load_intmdp(mdp_path)
S   = state_count(mdp)

# -----------------------------
# Parameters
# -----------------------------
T  = 100
t0 = 0
α  = 0.25
s0 = 1
γ  = 0.9

rmax = VarDP3.max_abs_reward(mdp)

# -----------------------------
# Fixed τ-grid (same for all t)
# -----------------------------
Δτ = 0.005   # (0.5, 0.05, 0.001, 0.0005, ...)

geom = (1 - γ^T) / (1 - γ)
M = rmax * geom
Xgrid = vcat(-Inf, collect(-M:Δτ:M), Inf)

# Same grid at every t
X = [Xgrid for _ in 0:T]

println("Using horizon T=$T, α=$α, Δτ=$Δτ")

# -----------------------------
# Run DPs
# -----------------------------
println("\nRunning VaR-DP with side = :plus (h^+) ...")
V_plus,  π_plus  = VarDP3.vi(mdp, X, T; side=:plus, γ=γ)
println("  V_plus length: ", length(V_plus))
println("  π_plus length: ", length(π_plus))

println("\nRunning VaR-DP with side = :minus (h^-) ...")
V_minus, π_minus = VarDP3.vi(mdp, X, T; side=:minus, γ=γ)
println("  V_minus length: ", length(V_minus))
println("  π_minus length: ", length(π_minus))

# -----------------------------
# VaR bounds at time t0 for all the states
# -----------------------------
VaR_minus_all = Vector{Union{Nothing,Float64}}(undef, S)
VaR_plus_all  = Vector{Union{Nothing,Float64}}(undef, S)

for s in 1:S
    VaR_minus_all[s] = VarDP3.var(V_minus, X, t0, s, α)
    VaR_plus_all[s]  = VarDP3.var(V_plus,  X, t0, s, α)
end

println("\nVaR bounds at t=$t0 (left-quantile level α=$α)")
for s in 1:S
    println("State $s:  VaR^- = $(VaR_minus_all[s])   VaR^+ = $(VaR_plus_all[s])")
end

# -----------------------------
# Save policy-at-VaR table to CSV
# -----------------------------
grid0 = X[t0+1]
mesh0 = VarDP3.Mesh(grid0)

output_dir = joinpath(@__DIR__, "..", "output")
mkpath(output_dir)

policy_file = joinpath(output_dir, "policy_at_VaR_t$(t0).csv")
open(policy_file, "w") do f
    write(f, "state,VaR_minus,action_minus,VaR_plus,action_plus\n")
    for s in 1:S
        τm = VaR_minus_all[s]
        τp = VaR_plus_all[s]
        if τm === nothing || τp === nothing
            continue
        end
        km = VarDP3.hm(mesh0, τm)
        kp = VarDP3.hp(mesh0, τp)
        am = π_minus[t0+1][s, km]
        ap = π_plus[t0+1][s, kp]
        write(f, "$s,$τm,$am,$τp,$ap\n")
    end
end
println("\n✓ Saved: $policy_file")

# -----------------------------
# Figure 1: DP value function bounds for state s0
# -----------------------------
τgrid_all  = X[t0+1]
finite_idx = findall(isfinite, τgrid_all)
τgrid      = τgrid_all[finite_idx]

v_minus = vec(V_minus[t0+1][s0, finite_idx])
v_plus  = vec(V_plus[t0+1][s0,  finite_idx])

p1 = plot(
    τgrid, v_minus,
    xlabel = "τ",
    ylabel = "v₀(s₀, τ) = P(ρ < τ)",
    label  = "DP v⁻ (h⁻)",
    lw     = 2,
    title  = "DP Bounds + MC Greedy Check at t=$t0, s₀=$s0"
)
plot!(p1, τgrid, v_plus, label="DP v⁺ (h⁺)", lw=2)

# ------------------------------------------------
# Monte Carlo overlay: greedy policy induced by v^+
# Evaluate the whole CDF curve (subset via stride)
# -------------------------------------------------
Nmc    = 200  # 300,  500  episodes per τ point
stride = 4000  #3000, 2000     # evaluate every stride-th τ point (speed vs resolution)
seed   = 123

τ_mc, v_mc_plus = mc_value_curve(mdp, π_plus, Xgrid, s0, T;
                                 γ=γ, side=:plus, N=Nmc, seed=seed, stride=stride)

plot!(p1, τ_mc, v_mc_plus, lw=2, ls=:dash, label="MC v^{π^+}(s₀,τ)")

display(p1)

# Optional dominance diagnostics: MC - DP(v^+) at nearest grid point
mesh_full = VarDP3.Mesh(Xgrid)                 # must include ±Inf
v_plus_full = vec(V_plus[t0+1][s0, :])         # DP values on full grid

viol = Float64[]
for (τ, vmc) in zip(τ_mc, v_mc_plus)
    k = VarDP3.hp(mesh_full, τ)
    push!(viol, vmc - v_plus_full[k])
end

println("\nDominance check for v^{π^+} vs v^+ (sampled τ points):")
println("  Max(MC - DP v^+)  = ", maximum(viol))
println("  Mean(MC - DP v^+) = ", mean(viol))

# ---------------------------------------------
# MC VaR at α = 0.25 under greedy(π_plus)
# Start τ0 at DP VaR^+(α)
# --------------------------------------------
τ0_plus = VaR_plus_all[s0]
@assert τ0_plus !== nothing

Nvar = 25000 # 40000, 100_000 episodes for VaR estimation (adjust for accuracy vs speed)
var_mc = mc_var(mdp, π_plus, Xgrid, s0, τ0_plus, T, α; γ=γ, side=:plus, N=Nvar, seed=999)

println("\nVaR consistency check at α=$α, state s0=$s0:")
println("  DP VaR^+ (α)          = ", τ0_plus)
println("  MC VaR_α under greedy = ", var_mc)
println("  Difference (MC - DP)  = ", var_mc - τ0_plus)

# -----------------------------------------------------------------
# Figure 2: VaR bounds for state s0 across multiple α values (DP only)
# --------------------------------------------------------------
alphas = sort(vcat(collect(0.1:0.1:0.9), α))

bounds_file = joinpath(output_dir, "var_bounds_by_alpha_t$(t0).csv")
open(bounds_file, "w") do f
    write(f, "alpha,state,VaR_minus,VaR_plus\n")
    for αi in alphas
        for s in 1:S
            VaR_minus_i = VarDP3.var(V_minus, X, t0, s, αi)
            VaR_plus_i  = VarDP3.var(V_plus,  X, t0, s, αi)
            if VaR_minus_i !== nothing && VaR_plus_i !== nothing
                write(f, "$αi,$s,$VaR_minus_i,$VaR_plus_i\n")
            end
        end
    end
end
println("\n✓ Saved: $bounds_file")

p2 = plot(
    xlabel = "α",
    ylabel = "τ",
    xticks = alphas,
    title  = "DP VaR Bounds at t=$t0 for s₀=$s0",
    legend = :topleft
)

alphas_plus  = Float64[]
taus_plus    = Float64[]
alphas_minus = Float64[]
taus_minus   = Float64[]

for αi in alphas
    VaR_minus_i = VarDP3.var(V_minus, X, t0, s0, αi)
    VaR_plus_i  = VarDP3.var(V_plus,  X, t0, s0, αi)

    if VaR_plus_i === nothing || VaR_minus_i === nothing
        continue
    end

    plot!(p2, [αi, αi], [VaR_plus_i, VaR_minus_i], lw=1.5, color=:gray, label="")
    push!(alphas_plus,  αi); push!(taus_plus,  VaR_plus_i)
    push!(alphas_minus, αi); push!(taus_minus, VaR_minus_i)
end

scatter!(p2, alphas_plus,  taus_plus,  ms=6, color=:red,  markerstrokecolor=:red,  label="VaR^+")
scatter!(p2, alphas_minus, taus_minus, ms=6, color=:blue, markerstrokecolor=:blue, label="VaR^-")

display(p2)