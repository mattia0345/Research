# Check and install required packages
import Pkg
for pkg in ["DifferentialEquations", "Distributions", "Plots", "JLD2", "LinearAlgebra", "Printf"]
    if !haskey(Pkg.project().dependencies, pkg)
        println("Installing $pkg...")
        Pkg.add(pkg)
    end
end

using DifferentialEquations
using LinearAlgebra
using Random
using Distributions
using Plots
using JLD2
using Printf
using Base.Threads

# Include the functions from the previous file
# Assuming PHOS_FUNCTIONS.jl contains: MATRIX_FINDER, MATTIA_FULL, guess_generator, 
# matrix_clip, is_duplicate, AllParameterGeneration, alpha_parameter_generation, etc.
include("mattia_phos_functions.jl")


"""
    stable_event_condition(u, t, integrator)

Condition function for DifferentialEquations event detection.
Returns value that should cross zero when system reaches steady state.
"""
function stable_event_condition(u, t, integrator)
    n = integrator.p[1]
    a_tot = integrator.p[2]
    x_tot = integrator.p[3]
    y_tot = integrator.p[4]
    Kp, Pp, G, H, Q, M_mat, D, N_mat = integrator.p[5:12]
    
    state_dot = MATTIA_FULL(t, u, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)
    
    eps = 1e-14  # Smaller epsilon for better precision
    threshold = 1e-10  # Stricter threshold for steady state detection
    
    state_norm = norm(u)
    dot_norm = norm(state_dot)
    
    denom = max(state_norm, eps)
    rel_change = dot_norm / denom
    
    value = rel_change - threshold
    
    return value
end


"""
    stable_event_affect!(integrator)

Affect function for event detection - terminates integration.
"""
function stable_event_affect!(integrator)
    terminate!(integrator)
end


"""
    mattia_full_ode!(du, u, p, t)

ODE function wrapper for DifferentialEquations.jl
"""
function mattia_full_ode!(du, u, p, t)
    n, a_tot, x_tot, y_tot = p[1:4]
    Kp, Pp, G, H, Q, M_mat, D, N_mat = p[5:12]
    
    result = MATTIA_FULL(t, u, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)
    du .= result
end


"""
    phosphorylation_system_solver(parameters_tuple, initial_states_array, final_t, plot_solution)

Solve the phosphorylation ODE system.

# Arguments
- `parameters_tuple`: Tuple of (n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)
- `initial_states_array::Vector{Float64}`: Initial conditions
- `final_t::Float64`: Final time for integration
- `plot_solution::Bool`: Whether to plot the solution

# Returns
- `Vector{Float64}`: Final state of the system
"""
function phosphorylation_system_solver(parameters_tuple, 
                                      initial_states_array::Vector{Float64}, 
                                      final_t::Float64, 
                                      plot_solution::Bool)
    
    n, a_tot = parameters_tuple[1:2]
    N = 2^n
    
    @assert all(initial_states_array .>= 0) "All initial states must be non-negative"
    
    initial_t = 0.0
    tspan = (initial_t, final_t)
    
    # Improved tolerances for better accuracy
    abstol = 1e-10  # Stricter absolute tolerance
    reltol = 1e-8   # Stricter relative tolerance
    
    @assert length(initial_states_array) == 3*N - 3 "Initial state array has wrong length"
    
    # Create callback for stable event detection
    stable_cb = ContinuousCallback(stable_event_condition, stable_event_affect!)
    
    # Set up ODE problem
    prob = ODEProblem(mattia_full_ode!, initial_states_array, tspan, parameters_tuple)
    
    # Solve with QNDF algorithm and stricter tolerances
    sol = solve(prob, QNDF(autodiff=false), 
                abstol=abstol, reltol=reltol,
                callback=stable_cb,
                maxiters=1e7)  # Allow more iterations for convergence
    
    if plot_solution
        plot_phosphorylation_dynamics(sol, n, a_tot)
    end
    
    return sol.u[end]
end


"""
    plot_phosphorylation_dynamics(sol, n, a_tot)

Plot the phosphorylation dynamics over time.
"""
function plot_phosphorylation_dynamics(sol, n::Int, a_tot::Float64)
    N = 2^n
    
    # Extract solution components
    t = sol.t
    a_solution = [sol.u[i][1:N-1] for i in 1:length(sol)]
    b_solution = [sol.u[i][N:2*N-2] for i in 1:length(sol)]
    c_solution = [sol.u[i][2*N-1:3*N-3] for i in 1:length(sol)]
    
    # Convert to matrices for plotting
    a_mat = hcat(a_solution...)'
    b_mat = hcat(b_solution...)'
    c_mat = hcat(c_solution...)'
    
    # Calculate A_N using conservation law
    a_N = a_tot .- sum(a_mat, dims=2) .- sum(b_mat, dims=2) .- sum(c_mat, dims=2)
    
    # Create plot
    p = plot(title="Phosphorylation dynamics for n = $n",
             xlabel="Time", ylabel="Concentration",
             legend=:outerright, size=(1000, 600),
             xlims=(t[1] - 0.1, t[end] + 0.1),
             ylims=(-0.05, 1.1))
    
    # Plot each A_i species
    for i in 1:N-1
        plot!(p, t, a_mat[:, i], label="A_$i", lw=3, alpha=0.9)
    end
    
    # Plot A_N
    plot!(p, t, vec(a_N), label="A_$(N-1)", lw=3, alpha=0.9)
    
    display(p)
end


"""
    fp_finder(n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)

Find fixed points of the phosphorylation system.

# Returns
- `Matrix{Float64}`: Array of stable fixed points (each row is a fixed point)
"""
function fp_finder(n::Int, a_tot::Float64, x_tot::Float64, y_tot::Float64,
                   Kp, Pp, G, H, Q, M_mat, D, N_mat)
    
    N = 2^n
    
    # Improved tolerances for duplicate detection
    norm_tol = 1e-6  # Tighter tolerance for Euclidean distance
    relative_tol = 1e-6  # Relative tolerance for element-wise comparison
    final_t = 1e8
    stable_points = Vector{Vector{Float64}}()
    plot_solution = false
    
    attempt_total_num = 12
    guesses = Vector{Vector{Float64}}()
    
    # Add initial guesses
    push!(guesses, 1e-14 * ones(3*N - 3))
    push!(guesses, vcat([1 - 1e-14], zeros(3*N - 4)))
    
    # Generate random guesses
    for i in 1:attempt_total_num
        push!(guesses, vcat([rand()], zeros(3*N - 4)))
        push!(guesses, guess_generator(n, a_tot, x_tot, y_tot))
    end
    
    parameters_tuple = (n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)
    
    for guess in guesses
        initial_states_array = Vector{Float64}(guess)
        
        reduced_sol = phosphorylation_system_solver(parameters_tuple, 
                                                   initial_states_array, 
                                                   final_t, 
                                                   plot_solution)
        
        # Check if it's a valid fixed point with stricter tolerance
        residuals = MATTIA_FULL(0.0, reduced_sol, n, a_tot, x_tot, y_tot, 
                               Kp, Pp, G, H, Q, M_mat, D, N_mat)
        
        if maximum(abs.(residuals)) > 1e-9
            # println("Point with max residual $(maximum(abs.(residuals))) is not a valid zero.")
            continue
        end
        
        # Compute Jacobian numerically
        function mattia_full_wrapper(state)
            return MATTIA_FULL(0.0, state, n, a_tot, x_tot, y_tot, 
                             Kp, Pp, G, H, Q, M_mat, D, N_mat)
        end
        
        jacobian_mattia_full = compute_jacobian(mattia_full_wrapper, reduced_sol)
        eigenvalues = eigvals(jacobian_mattia_full)
        eigenvalues_real = real.(eigenvalues)
        
        # Check for duplicates with improved comparison
        if is_duplicate_improved(reduced_sol, stable_points, norm_tol, relative_tol)
            continue
        end
        
        push!(stable_points, reduced_sol)
    end
    
    # Consolidate fixed points to remove any remaining near-duplicates
    stable_points = consolidate_fixed_points(stable_points, norm_tol, relative_tol)
    
    println("# of stable points found is $(length(stable_points))")
    
    if length(stable_points) == 0
        return Matrix{Float64}(undef, 0, 0)
    end
    
    # Convert to matrix (each row is a fixed point)
    return hcat(stable_points...)' |> Matrix{Float64}
end


"""
    is_duplicate_improved(candidate, collection, euclidean_distance_tol, relative_tol)

Check if candidate vector is close to any vector in collection using multiple criteria.

Uses both absolute Euclidean distance and relative element-wise comparison.
"""
function is_duplicate_improved(candidate::Vector{Float64}, 
                              collection::Vector{Vector{Float64}}, 
                              euclidean_distance_tol::Float64,
                              relative_tol::Float64)::Bool
    if isempty(collection)
        return false
    end
    
    for v in collection
        # Check Euclidean distance
        distance = norm(v - candidate)
        
        # Check relative difference for each element
        max_relative_diff = 0.0
        for i in 1:length(v)
            if abs(v[i]) > 1e-10 || abs(candidate[i]) > 1e-10
                # Use relative difference when values are not near zero
                denom = max(abs(v[i]), abs(candidate[i]))
                rel_diff = abs(v[i] - candidate[i]) / denom
                max_relative_diff = max(max_relative_diff, rel_diff)
            else
                # For near-zero values, use absolute difference
                max_relative_diff = max(max_relative_diff, abs(v[i] - candidate[i]))
            end
        end
        
        # Consider duplicate if BOTH conditions are met:
        # 1. Euclidean distance is small
        # 2. Maximum relative difference is small
        if distance < euclidean_distance_tol && max_relative_diff < relative_tol
            return true
        end
    end
    
    return false
end


"""
    consolidate_fixed_points(points, norm_tol, relative_tol)

Remove near-duplicate fixed points from a collection.
Returns consolidated list with unique fixed points only.
"""
function consolidate_fixed_points(points::Vector{Vector{Float64}}, 
                                  norm_tol::Float64, 
                                  relative_tol::Float64)
    if isempty(points)
        return points
    end
    
    consolidated = Vector{Vector{Float64}}()
    
    for point in points
        if !is_duplicate_improved(point, consolidated, norm_tol, relative_tol)
            push!(consolidated, point)
        end
    end
    
    return consolidated
end


"""
    compute_jacobian(f, x; epsilon=1e-8)

Compute Jacobian matrix numerically using finite differences.
"""
function compute_jacobian(f::Function, x::Vector{Float64}; epsilon::Float64=1e-8)
    n = length(x)
    m = length(f(x))
    J = zeros(m, n)
    
    for j in 1:n
        x_plus = copy(x)
        x_plus[j] += epsilon
        J[:, j] = (f(x_plus) - f(x)) / epsilon
    end
    
    return J
end


"""
    process_sample_script(i, n, a_tot_value, x_tot_value_parameter_array, ...)

Process a single sample to find multistable states.
"""
function process_sample_script(i::Int,
                              n::Int,
                              a_tot_value::Float64,
                              x_tot_value_parameter_array::Vector{Vector{Float64}},
                              y_tot_value_parameter_array::Vector{Vector{Float64}},
                              alpha_matrix_parameter_array::Vector{Matrix{Float64}},
                              beta_matrix_parameter_array::Vector{Matrix{Float64}},
                              k_positive_parameter_array::Vector{Vector{Float64}},
                              k_negative_parameter_array::Vector{Vector{Float64}},
                              p_positive_parameter_array::Vector{Vector{Float64}},
                              p_negative_parameter_array::Vector{Vector{Float64}})
    
    N = 2^n
    alpha_matrix = alpha_matrix_parameter_array[i]
    beta_matrix = beta_matrix_parameter_array[i]
    k_positive_rates = k_positive_parameter_array[i]
    k_negative_rates = k_negative_parameter_array[i]
    p_positive_rates = p_positive_parameter_array[i]
    p_negative_rates = p_negative_parameter_array[i]
    x_tot_value = x_tot_value_parameter_array[i][1]
    y_tot_value = y_tot_value_parameter_array[i][1]
    
    rate_min, rate_max = 1e-2, 1e2
    
    # Normalize rates
    k_positive_rates = k_positive_rates / mean(k_positive_rates)
    k_negative_rates = k_negative_rates / mean(k_negative_rates)
    p_positive_rates = p_positive_rates / mean(p_positive_rates)
    p_negative_rates = p_negative_rates / mean(p_negative_rates)
    alpha_matrix = alpha_matrix / mean(alpha_matrix)
    beta_matrix = beta_matrix / mean(beta_matrix)
    
    # Clip rates
    k_positive_rates = clamp.(k_positive_rates, rate_min, rate_max)
    k_negative_rates = clamp.(k_negative_rates, rate_min, rate_max)
    p_positive_rates = clamp.(p_positive_rates, rate_min, rate_max)
    p_negative_rates = clamp.(p_negative_rates, rate_min, rate_max)
    alpha_matrix = matrix_clip(alpha_matrix, rate_min, rate_max)
    beta_matrix = matrix_clip(beta_matrix, rate_min, rate_max)
    
    multistable_results = nothing
    
    Kp, Pp, G, H, Q, M_mat, D, N_mat = MATRIX_FINDER(n, alpha_matrix, beta_matrix,
                                                      k_positive_rates, k_negative_rates,
                                                      p_positive_rates, p_negative_rates)
    
    unique_stable_fp_array = fp_finder(n, a_tot_value, x_tot_value, y_tot_value,
                                      Kp, Pp, G, H, Q, M_mat, D, N_mat)
    
    possible_steady_states = floor(Int, (n + 2) / 2)
    println(i)
    
    if size(unique_stable_fp_array, 1) == possible_steady_states
        multistable_results = Dict(
            "num_of_stable_states" => size(unique_stable_fp_array, 1),
            "stable_states" => unique_stable_fp_array,
            "total_concentration_values" => [a_tot_value, x_tot_value, y_tot_value],
            "alpha_matrix" => alpha_matrix,
            "beta_matrix" => beta_matrix,
            "k_positive_rates" => k_positive_rates,
            "k_negative_rates" => k_negative_rates,
            "p_positive_rates" => p_positive_rates,
            "p_negative_rates" => p_negative_rates
        )
    end
    
    return multistable_results
end


"""
    simulation(n, simulation_size)

Run parallel simulation to find multistable parameter sets.

# Arguments
- `n::Int`: Number of phosphorylation sites
- `simulation_size::Int`: Number of parameter sets to test
"""
function simulation(n::Int, simulation_size::Int)
    a_tot_value = 1.0
    N = 2^n
    x_tot = 1e-2
    y_tot = 1e-2
    
    old_shape_parameters = [0.123, 4.46e6]
    new_shape_parameters = [1e6, 1e6]
    
    gen_rates = AllParameterGeneration(n, "distributive", "gamma", 
                                      old_shape_parameters, verbose=false)
    
    rate_min, rate_max = 1e-2, 1e3
    
    println("Generating parameters...")
    
    # Generate all parameters
    alpha_matrix_parameter_array = [alpha_parameter_generation(gen_rates) 
                                   for _ in 1:simulation_size]
    beta_matrix_parameter_array = [beta_parameter_generation(gen_rates) 
                                  for _ in 1:simulation_size]
    
    k_positive_parameter_array = Vector{Vector{Float64}}()
    k_negative_parameter_array = Vector{Vector{Float64}}()
    for _ in 1:simulation_size
        kp, kn = k_parameter_generation(gen_rates)
        push!(k_positive_parameter_array, kp)
        push!(k_negative_parameter_array, kn)
    end
    
    p_positive_parameter_array = Vector{Vector{Float64}}()
    p_negative_parameter_array = Vector{Vector{Float64}}()
    for _ in 1:simulation_size
        pp, pn = p_parameter_generation(gen_rates)
        push!(p_positive_parameter_array, pp)
        push!(p_negative_parameter_array, pn)
    end
    
    x_tot_value_parameter_array = [[x_tot] for _ in 1:simulation_size]
    y_tot_value_parameter_array = [[y_tot] for _ in 1:simulation_size]
    
    println("Running parallel simulations...")
    
    # Use threading for parallel execution
    results = Vector{Any}(undef, simulation_size)
    
    @threads for i in 1:simulation_size
        results[i] = process_sample_script(
            i, n, a_tot_value,
            x_tot_value_parameter_array,
            y_tot_value_parameter_array,
            alpha_matrix_parameter_array,
            beta_matrix_parameter_array,
            k_positive_parameter_array,
            k_negative_parameter_array,
            p_positive_parameter_array,
            p_negative_parameter_array
        )
    end
    
    # Filter out nothing results
    multistable_results_list = filter(!isnothing, results)
    
    # Save results
    multifile = "multistability_parameters_$(simulation_size)_$(n).jld2"
    
    @save multifile multistable_results_list
    
    println("Saved $(length(multistable_results_list)) multistable results to $multifile")
    
    return multistable_results_list
end


"""
    main()

Main entry point for the simulation.
"""
function main()
    n = 2
    simulation(n, 500)
end


# Run main if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end