# solve_system.jl

# To run this script, ensure you have Julia installed and the 
# HomotopyContinuation.jl package added to your environment.
# You can run it from the Julia REPL using: include("solve_system.jl")

using HomotopyContinuation

# 1. Define the polynomial variables using the @polyvar macro
# This creates symbolic variables x and y
@polyvar x y

# 2. Define the system of polynomial equations
# F1: x^2 + y^2 - 1.0 = 0 (A circle)
# F2: x^2 - y - 0.5 = 0 (A parabola)
F = [
    x^2 + y^2 - 1.0,
    x^2 - y - 0.5
]

println("--- Solving Polynomial System ---")
println("System of equations:")
println("  F1: $F[1]")
println("  F2: $F[2]")

# 3. Solve the system
result = solve(F)

# 4. Print the results
println("\n--- Solution Results ---")
println("Total number of paths tracked: $(result.ntracked)")
println("Number of solutions found: $(result.nresults)")

println("\n--- All Complex Solutions (x, y) ---")

# The solutions() function extracts the complex solution vectors
for (i, sol_vec) in enumerate(solutions(result))
    # sol_vec is a vector containing the (x, y) coordinates
    x_sol = sol_vec[1]
    y_sol = sol_vec[2]

    # Displaying the real and imaginary parts
    println("Solution $i:")
    println("  x = $(real(x_sol)) + $(imag(x_sol))i")
    println("  y = $(real(y_sol)) + $(imag(y_sol))i")
end
