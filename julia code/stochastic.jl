using Catalyst
using OrdinaryDiffEq      
using Plots
using SymbolicIndexingInterface: getname  # for nice labels
using JumpProcesses
# Define the reaction network (enzyme kinetics)
model = @reaction_network begin
    (k1, k1), A <--> B
    (k2, k2), A + B <--> 2B
    (k3, k3), B <--> C
    (r_a, r_a), A <--> A_
    (r_c, r_c), C <--> C_
end

u0    = [:A => 20, :B => 2, :C => 3, :A_ => 2, :C_ => 4]
tspan = (0, 1000.0)
ps    = [:k1 => 1, :k2 => 1, :k3 => 1, :r_a => 1, :r_c => 1]

jump_sol = solve(JumpProblem(JumpInputs(model, u0, tspan, ps)))

idxA = findfirst(p -> p.first == :A, u0)
@assert idxA !== nothing "Couldn't find :A in u0 — check how u0 is specified."

# ---- extract times and A-values ----
tvals = jump_sol.t
Avals = [state[idxA] for state in jump_sol.u]   # each state is a vector

# ---- plot just [A] ----
plot(tvals, Avals;
     lw = 2,
     xlabel = "time",
     ylabel = "[A]",
     label = "A")