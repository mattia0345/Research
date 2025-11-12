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

# Include the functions from the previous files
include("mattia_phos_functions.jl")


"""
    mattia_full_ode_simple!(du, u, p, t)

Simple ODE function wrapper for DifferentialEquations.jl (no events)
"""
function mattia_full_ode_simple!(du, u, p, t)
    n, a_tot, x_tot, y_tot = p[1:4]
    Kp, Pp, G, H, Q, M_mat, D, N_mat = p[5:12]
    
    result = MATTIA_FULL(t, u, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)
    du .= result
end


"""
    phosphorylation_system_solver_plot(parameters_tuple, initial_states_array, final_t)

Solve and plot the phosphorylation ODE system.

# Arguments
- `parameters_tuple`: Tuple of (n, alpha_matrix, beta_matrix, k_positive_rates, 
                      k_negative_rates, p_positive_rates, p_negative_rates, 
                      a_tot, x_tot, y_tot)
- `initial_states_array::Vector{Float64}`: Initial conditions
- `final_t::Float64`: Final time for integration

# Returns
- `ODESolution`: The solution object from DifferentialEquations.jl
"""
function phosphorylation_system_solver_plot(parameters_tuple, 
                                           initial_states_array::Vector{Float64}, 
                                           final_t::Float64)
    
    n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, 
    p_positive_rates, p_negative_rates, a_tot, x_tot, y_tot = parameters_tuple
    
    N = 2^n
    
    @assert all(initial_states_array .>= 0) "All initial states must be non-negative"
    
    initial_t = 0.0
    tspan = (initial_t, final_t)
    
    # Get all matrices
    Kp, Pp, G, H, Q, M_mat, D, N_mat = MATRIX_FINDER(n, alpha_matrix, beta_matrix,
                                                      k_positive_rates, k_negative_rates,
                                                      p_positive_rates, p_negative_rates)
    
    # Tolerances matching Python
    abstol = 1e-20
    reltol = 1e-5
    
    mattia_full_parameter_tuple = (n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)
    
    # Set up and solve ODE problem
    prob = ODEProblem(mattia_full_ode_simple!, initial_states_array, tspan, 
                     mattia_full_parameter_tuple)
    
    sol = solve(prob, QNDF(autodiff=false), 
                abstol=abstol, reltol=reltol,
                maxiters=1e7)
    
    # Plot the solution
    plot_full_dynamics(sol, n, a_tot, tspan)
    
    return sol
end


"""
    plot_full_dynamics(sol, n, a_tot, tspan)

Plot the phosphorylation dynamics with all species.

# Arguments
- `sol`: ODE solution object
- `n::Int`: Number of phosphorylation sites
- `a_tot::Float64`: Total concentration of species A
- `tspan::Tuple`: Time span (t_initial, t_final)
"""
function plot_full_dynamics(sol, n::Int, a_tot::Float64, tspan::Tuple)
    N = 2^n
    
    # Extract solution components
    t = sol.t
    
    # Extract a, b, c components from solution
    a_solution = [sol.u[i][1:N-1] for i in 1:length(sol)]
    b_solution = [sol.u[i][N:2*N-2] for i in 1:length(sol)]
    c_solution = [sol.u[i][2*N-1:3*N-3] for i in 1:length(sol)]
    
    # Convert to matrices for plotting
    a_mat = hcat(a_solution...)'
    b_mat = hcat(b_solution...)'
    c_mat = hcat(c_solution...)'
    
    # Calculate A_N using conservation law
    a_N = a_tot .- sum(a_mat, dims=2) .- sum(b_mat, dims=2) .- sum(c_mat, dims=2)
    
    # Get color palette (similar to matplotlib's Accent)
    colors = palette(:tab10)
    
    # Create the plot
    p = plot(
        title="Plotting reduced phosphorylation dynamics for n = $n",
        xlabel="Time", 
        ylabel="Concentration",
        legend=:outerright, 
        size=(1000, 700),
        xlims=(tspan[1] - 0.1, tspan[2] + 0.1),
        ylims=(-0.05, 1.1),
        grid=true,
        minorgrid=true,
        framestyle=:box
    )
    
    # Plot each A_i species (reduced coordinates)
    for i in 1:N-1
        color_idx = mod1(i, length(colors))
        plot!(p, t, a_mat[:, i], 
              label="\$[A_{$(i-1)}]\$",  # 0-indexed like Python
              lw=4, 
              alpha=0.9,
              color=colors[color_idx])
    end
    
    # Plot A_N (the last species from conservation)
    color_N_idx = mod1(N, length(colors))
    plot!(p, t, vec(a_N), 
          label="\$[A_{$(N-1)}]\$",
          lw=4, 
          alpha=0.9,
          color=colors[color_N_idx])
    
    display(p)
    
    return p
end


"""
    plotter(index, final_t, file_name)

Load multistability results and plot trajectories from various initial conditions.

# Arguments
- `index::Int`: Index of the parameter set to plot (1-indexed)
- `final_t::Float64`: Final time for integration
- `file_name::String`: Name of the JLD2 file containing results
"""
function plotter(index::Int, final_t::Float64, file_name::String)
    # Extract n from filename (assumes format: "multistability_parameters_SIZE_N.jld2")
    # Extract the second-to-last number before .jld2
    parts = split(replace(file_name, ".jld2" => ""), "_")
    n = parse(Int, parts[end])
    N = 2^n
    
    # Load the multistability results
    @load file_name multistable_results_list
    
    println("\n=== Available Parameter Sets ===")
    for i in 1:length(multistable_results_list)
        num_states = multistable_results_list[i]["num_of_stable_states"]
        println("Index $i: # of stable states: $num_states")
        
        # Print each stable state
        stable_states = multistable_results_list[i]["stable_states"]
        for j in 1:size(stable_states, 1)
            println("  State $j: $(stable_states[j, :])")
        end
    end
    println("================================\n")
    
    # Get parameters for the selected index
    results = multistable_results_list[index]
    
    a_stable_states = results["stable_states"]
    a_tot_parameter = results["total_concentration_values"][1]
    x_tot_parameter = results["total_concentration_values"][2]
    y_tot_parameter = results["total_concentration_values"][3]
    alpha_matrix = results["alpha_matrix"]
    beta_matrix = results["beta_matrix"]
    k_positive_rates = results["k_positive_rates"]
    k_negative_rates = results["k_negative_rates"]
    p_positive_rates = results["p_positive_rates"]
    p_negative_rates = results["p_negative_rates"]
    
    println("Plotting for index $index")
    println("Number of stable states: $(size(a_stable_states, 1))")
    println("Total concentrations: a=$a_tot_parameter, x=$x_tot_parameter, y=$y_tot_parameter")
    
    attempt_total_num = floor(Int, 1.75 * N)
    
    initial_conditions_list = Vector{Vector{Float64}}()
    
    # Generate initial conditions similar to Python
    for i in 1:attempt_total_num
        # Type 1: zero at position 1, random at position 2, rest zeros
        push!(initial_conditions_list, 
              vcat([0.0], [rand()], zeros(3*N - 5)))
        
        # Type 2: random at position 1, rest zeros
        push!(initial_conditions_list, 
              vcat([rand()], zeros(3*N - 4)))
        
        # Type 3: random guess from generator
        push!(initial_conditions_list, 
              guess_generator(n, a_tot_parameter, x_tot_parameter, y_tot_parameter))
    end
    
    parameters_tuple = (n, alpha_matrix, beta_matrix, 
                       k_positive_rates, k_negative_rates, 
                       p_positive_rates, p_negative_rates,
                       a_tot_parameter, x_tot_parameter, y_tot_parameter)
    
    println("\nGenerating plots from $(length(initial_conditions_list)) initial conditions...")
    
    # Plot from each initial condition
    for (idx, guess) in enumerate(initial_conditions_list)
        println("Plotting trajectory $idx/$(length(initial_conditions_list))...")
        
        try
            sol = phosphorylation_system_solver_plot(parameters_tuple, guess, final_t)
            
            # Optional: save each plot
            # savefig("trajectory_$(index)_$(idx).png")
            
        catch e
            println("  Warning: Failed to solve for initial condition $idx")
            println("  Error: $e")
            continue
        end
    end
    
    println("\nPlotting complete!")
end


"""
    load_and_inspect(file_name)

Load and inspect the contents of a multistability results file.

# Arguments
- `file_name::String`: Name of the JLD2 file
"""
function load_and_inspect(file_name::String)
    if !isfile(file_name)
        println("Error: File '$file_name' not found!")
        return
    end
    
    @load file_name multistable_results_list
    
    println("\n=== Multistability Results Summary ===")
    println("Total parameter sets: $(length(multistable_results_list))")
    println("\nDetailed breakdown:")
    
    for (i, result) in enumerate(multistable_results_list)
        println("\n--- Index $i ---")
        println("  Number of stable states: $(result["num_of_stable_states"])")
        println("  Total concentrations: $(result["total_concentration_values"])")
        println("  Stable states shape: $(size(result["stable_states"]))")
        
        # Show first few elements of each stable state
        stable_states = result["stable_states"]
        for j in 1:min(size(stable_states, 1), 3)  # Show up to 3 states
            println("    State $j (first 5 elements): $(stable_states[j, 1:min(5, end)])")
        end
    end
    
    println("\n=====================================\n")
end


"""
    plot_stable_states_comparison(index, file_name)

Plot all stable states for a given parameter set for visual comparison.

# Arguments
- `index::Int`: Index of the parameter set
- `file_name::String`: Name of the JLD2 file
"""
function plot_stable_states_comparison(index::Int, file_name::String)
    @load file_name multistable_results_list
    
    result = multistable_results_list[index]
    stable_states = result["stable_states"]
    
    # Extract n from filename
    parts = split(replace(file_name, ".jld2" => ""), "_")
    n = parse(Int, parts[end])
    N = 2^n
    
    # Create bar plot comparing stable states
    num_states = size(stable_states, 1)
    
    p = plot(
        title="Stable States Comparison (Index $index)",
        xlabel="Species Index",
        ylabel="Concentration",
        legend=:outerright,
        size=(1000, 600)
    )
    
    x_labels = ["A$(i-1)" for i in 1:N-1]
    append!(x_labels, ["B$(i-1)" for i in 1:N-1])
    append!(x_labels, ["C$(i)" for i in 1:N-1])
    
    for i in 1:num_states
        bar!(p, 1:length(stable_states[i, :]), stable_states[i, :],
             label="State $i",
             alpha=0.7,
             bar_width=0.8/num_states,
             offset=(i-1)*0.8/num_states - 0.4 + 0.4/num_states)
    end
    
    display(p)
    
    return p
end


"""
    main()

Main entry point - load results and generate plots.
"""
function main()
    # Adjust these parameters as needed
    file_name = "multistability_parameters_500_2.jld2"
    index = 2
    final_t = 1e7  # 10 million in scientific notation
    
    # First, inspect the file
    if isfile(file_name)
        println("Loading and inspecting results file...")
        load_and_inspect(file_name)
        
        # Then plot trajectories
        println("\nStarting trajectory plots...")
        plotter(index, final_t, file_name)
        
        # Also plot stable states comparison
        println("\nPlotting stable states comparison...")
        plot_stable_states_comparison(index, file_name)
    else
        println("Error: File '$file_name' not found!")
        println("Please run the simulation first to generate the results file.")
        println("Expected format: multistability_parameters_SIZE_N.jld2")
    end
end


# Run main if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end