using Catalyst
using OrdinaryDiffEq      # ✅ this exists
using Plots
using SymbolicIndexingInterface: getname  # for nice labels

# Define the reaction network (enzyme kinetics)
model = @reaction_network begin
    (k1, k1), A <--> B
    (k2, k2), A + B <--> 2B
    (k3, k3), B <--> C
    (r_a, r_a), A <--> A_
    (r_c, r_c), C <--> C_
end

# Initial conditions and parameters
u0    = [:A => 1.0, :B => 2.0, :C => 3.0, :A_ => 2, :C_ => 4]
tspan = (1e-3, 1000.0)
ps    = [:k1 => 1, :k2 => 1, :k3 => 1, :r_a => 1, :r_c => 1]

# Build and solve ODE problem
ode = ODEProblem(model, u0, tspan, ps)
sol = solve(ode, Tsit5())   # Explicitly pick a solver

plot(sol; lw=2, xlabel="time", ylabel="concentration",  xaxis=:log)
# plot(sol(0); lw=2, xlabel="time", ylabel="concentration", label=labels)
