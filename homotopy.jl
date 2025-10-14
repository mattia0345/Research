# solve_system.jl
using HomotopyContinuation

# declare variables
@var x y

# system of polynomial equations
F = [
x^2 + y^2 - 1.0,   # circle
x^2 - y - 0.5      # parabola
]

println("--- Solving Polynomial System ---")
println("  F1: ", F[1])
println("  F2: ", F[2])

# solve
result = solve(F)

# extract solutions
sols = solutions(result)

println("\n--- Solution Results ---")
println("Number of solutions found: ", length(sols))

# print solutions (showing small imaginary parts)
function show_complex(z; tol = 1e-10)
re = real(z); im = imag(z)
if abs(im) < tol
    return string(re)        # effectively real
else
    return "$(re) + $(im)im" # complex
end
end

println("\n--- All Complex Solutions (x, y) ---")
for (i, sol) in enumerate(sols)
x_sol = sol[1]
y_sol = sol[2]
println("Solution $i:")
println("  x = ", show_complex(x_sol))
println("  y = ", show_complex(y_sol))
end

# optionally print only the (numerically) real solutions
real_sols = [(real(sol[1]), real(sol[2])) for sol in sols if abs(imag(sol[1])) < 1e-8 && abs(imag(sol[2])) < 1e-8]
println("\n--- Real Solutions (within tolerance) ---")
for (i, (xr, yr)) in enumerate(real_sols)
println("Real solution $i: x = $xr, y = $yr")
end
