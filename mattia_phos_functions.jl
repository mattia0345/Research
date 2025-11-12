import Pkg
for pkg in ["Distributions"]
    if !haskey(Pkg.project().dependencies, pkg)
        println("Installing $pkg...")
        Pkg.add(pkg)
    end
end

using LinearAlgebra
using Random
using Distributions
using Printf

"""
    MATRIX_FINDER(n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, 
                  p_positive_rates, p_negative_rates)

Construct matrices for the phosphorylation model ODE system.

# Arguments
- `n::Int`: Number of phosphorylation sites
- `alpha_matrix::Matrix{Float64}`: Alpha transition rate matrix
- `beta_matrix::Matrix{Float64}`: Beta transition rate matrix
- `k_positive_rates::Vector{Float64}`: Positive k rates
- `k_negative_rates::Vector{Float64}`: Negative k rates
- `p_positive_rates::Vector{Float64}`: Positive p rates
- `p_negative_rates::Vector{Float64}`: Negative p rates

# Returns
- Tuple of matrices: (Kp, Pp, G, H, Q, M_mat, D, N_mat)
"""
function MATRIX_FINDER(n::Int, 
                       alpha_matrix::Matrix{Float64}, 
                       beta_matrix::Matrix{Float64},
                       k_positive_rates::Vector{Float64}, 
                       k_negative_rates::Vector{Float64},
                       p_positive_rates::Vector{Float64}, 
                       p_negative_rates::Vector{Float64})
    
    N = 2^n
    ones_vec = ones(N - 1)
    
    # Construct Kp and Km matrices
    Kp = diagm(vcat(k_positive_rates, 0.0))
    Km = vcat(diagm(k_negative_rates), zeros(1, length(k_negative_rates)))
    
    @assert size(Kp) == (N, N) "Kp shape must be ($N, $N)"
    @assert size(Km) == (N, N-1) "Km shape must be ($N, $(N-1))"
    
    # Construct Pp and Pm matrices
    Pp = diagm(vcat(0.0, p_positive_rates))
    Pm = vcat(zeros(1, length(p_negative_rates)), diagm(p_negative_rates))
    
    @assert size(Pp) == (N, N) "Pp shape must be ($N, $N)"
    @assert size(Pm) == (N, N-1) "Pm shape must be ($N, $(N-1))"
    
    # Adjusted alpha and beta matrices (remove last/first row)
    adjusted_alpha_mat = alpha_matrix[1:end-1, :]
    adjusted_beta_mat = beta_matrix[2:end, :]
    
    @assert size(adjusted_alpha_mat) == (N-1, N) "adjusted_alpha_mat shape must be ($(N-1), $N)"
    @assert size(adjusted_beta_mat) == (N-1, N) "adjusted_beta_mat shape must be ($(N-1), $N)"
    
    # Diagonal matrices Da and Db
    Da = diagm(alpha_matrix[1:end-1, 2:end] * ones_vec)
    Db = diagm(beta_matrix[2:end, 1:end-1] * ones_vec)
    
    @assert size(Da) == (N-1, N-1) "Da shape must be ($(N-1), $(N-1))"
    @assert size(Db) == (N-1, N-1) "Db shape must be ($(N-1), $(N-1))"
    
    # U and I matrices
    U = diagm(k_negative_rates)
    I_mat = diagm(p_negative_rates)
    
    @assert size(U) == (N-1, N-1) "U shape must be ($(N-1), $(N-1))"
    @assert size(I_mat) == (N-1, N-1) "I shape must be ($(N-1), $(N-1))"
    
    # Q and D matrices
    Q = Kp[1:end-1, :]
    D = Pp[2:end, :]
    
    @assert size(Q) == (N-1, N) "Q shape must be ($(N-1), $N)"
    @assert size(D) == (N-1, N) "D shape must be ($(N-1), $N)"
    
    # M and N matrices
    M_mat = U + Da
    N_mat = I_mat + Db
    
    # G and H matrices
    G = Km + transpose(adjusted_alpha_mat)
    H = Pm + transpose(adjusted_beta_mat)
    
    @assert size(G) == (N, N-1) "G shape must be ($N, $(N-1))"
    @assert size(H) == (N, N-1) "H shape must be ($N, $(N-1))"
    
    return Kp, Pp, G, H, Q, M_mat, D, N_mat
end


"""
    MATTIA_FULL(t, state_array, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)

ODE system for the phosphorylation model with conservation laws.

# Arguments
- `t::Float64`: Time (not used, but required for ODE solvers)
- `state_array::Vector{Float64}`: State vector [a_red; b; c]
- `n::Int`: Number of sites
- `a_tot::Float64`: Total a concentration
- `x_tot::Float64`: Total x concentration
- `y_tot::Float64`: Total y concentration
- Matrices from MATRIX_FINDER

# Returns
- `Vector{Float64}`: Time derivatives [a_dot_red; b_dot; c_dot]
"""
function MATTIA_FULL(t::Float64, 
                     state_array::Vector{Float64}, 
                     n::Int,
                     a_tot::Float64, 
                     x_tot::Float64, 
                     y_tot::Float64,
                     Kp::Matrix{Float64}, 
                     Pp::Matrix{Float64}, 
                     G::Matrix{Float64}, 
                     H::Matrix{Float64},
                     Q::Matrix{Float64}, 
                     M_mat::Matrix{Float64}, 
                     D::Matrix{Float64}, 
                     N_mat::Matrix{Float64})
    
    N = 2^n
    
    @assert length(state_array) == 3*N - 3 "state_array length must be $(3*N - 3)"
    
    # Extract state components
    a_red = state_array[1:N-1]
    b = state_array[N:2*N-2]
    c = state_array[2*N-1:3*N-3]
    
    # Reconstruct full state using conservation laws
    a = vcat(a_red, [a_tot - sum(a_red) - sum(b) - sum(c)])
    x = x_tot - sum(b)
    y = y_tot - sum(c)
    
    # Calculate derivatives
    a_dot = (G * b) + (H * c) - x * (Kp * a) - y * (Pp * a)
    b_dot = x * (Q * a) - (M_mat * b)
    c_dot = y * (D * a) - (N_mat * c)
    
    # Extract reduced a_dot
    a_dot_red = a_dot[1:N-1]
    
    # Apply non-negativity constraints
    zero_threshold = 1e-15
    
    for i in 1:length(a_red)
        if a_red[i] < zero_threshold
            a_dot_red[i] = max(a_dot_red[i], 0.0)
        end
    end
    
    for i in 1:length(b)
        if b[i] < zero_threshold
            b_dot[i] = max(b_dot[i], 0.0)
        end
    end
    
    for i in 1:length(c)
        if c[i] < zero_threshold
            c_dot[i] = max(c_dot[i], 0.0)
        end
    end
    
    return vcat(a_dot_red, b_dot, c_dot)
end


"""
    AllParameterGeneration

Generate state transitions and random parameters for an n-site phosphorylation model.

# Fields
- `n::Int`: Number of sites
- `N::Int`: Number of states (2^n)
- `distribution::String`: Distribution name ("gamma" or "levy")
- `params::Vector{Float64}`: Distribution parameters [shape, scale]
- `reaction_types::String`: Type of reactions
- `verbose::Bool`: Print debug information
- `rng::AbstractRNG`: Random number generator
"""
mutable struct AllParameterGeneration
    n::Int
    N::Int
    distribution::String
    params::Vector{Float64}
    reaction_types::String
    verbose::Bool
    rng::AbstractRNG
    
    function AllParameterGeneration(n::Int, 
                                    reaction_types::String,
                                    distribution::String, 
                                    distribution_parameters::Vector{Float64};
                                    verbose::Bool=false)
        N = 2^n
        rng = MersenneTwister()
        new(n, N, distribution, distribution_parameters, reaction_types, verbose, rng)
    end
end


"""
    padded_binary(i::Int, n::Int) -> String

Convert integer to binary string with padding.
"""
function padded_binary(i::Int, n::Int)::String
    return string(i, base=2, pad=n)
end


"""
    binary_string_to_array(s::String) -> Vector{Int}

Convert binary string to integer array.
"""
function binary_string_to_array(s::String)::Vector{Int}
    return [parse(Int, c) for c in s]
end


"""
    calculate_valid_transitions(gen::AllParameterGeneration)

Calculate valid phosphorylation and dephosphorylation transitions.

# Returns
- `Tuple`: (valid_X_reactions, valid_Y_reactions)
"""
function calculate_valid_transitions(gen::AllParameterGeneration)
    num_states = gen.N
    all_states = [padded_binary(i, gen.n) for i in 0:num_states-1]
    
    valid_difference_vectors = Set{Vector{Int}}()
    valid_X_reactions = Vector{Vector{Any}}()
    valid_Y_reactions = Vector{Vector{Any}}()
    
    for i in 0:num_states-1
        arr_i = binary_string_to_array(all_states[i+1])
        
        for j in 0:num_states-1
            if i == j
                continue
            end
            
            arr_j = binary_string_to_array(all_states[j+1])
            diff = arr_j - arr_i
            hamming_weight = sum(abs.(diff))
            
            if hamming_weight == 1
                # +1 -> phosphorylation (E), -1 -> dephosphorylation (F)
                element = any(diff .== 1) ? "E" : "F"
                
                if element == "E"
                    if gen.verbose
                        println("$(all_states[i+1]) --> $(all_states[j+1]) (E), $i, $j")
                    end
                    push!(valid_X_reactions, [all_states[i+1], all_states[j+1], i, j, element])
                else
                    if gen.verbose
                        println("$(all_states[i+1]) --> $(all_states[j+1]) (F), $i, $j")
                    end
                    push!(valid_Y_reactions, [all_states[i+1], all_states[j+1], i, j, element])
                end
                
                push!(valid_difference_vectors, diff)
            end
        end
    end
    
    return valid_X_reactions, valid_Y_reactions
end


"""
    alpha_parameter_generation(gen::AllParameterGeneration) -> Matrix{Float64}

Generate alpha transition rate matrix.
"""
function alpha_parameter_generation(gen::AllParameterGeneration)::Matrix{Float64}
    valid_X_reactions, _ = calculate_valid_transitions(gen)
    
    shape, scale = gen.params
    alpha_matrix = zeros(gen.N, gen.N)
    
    for reaction in valid_X_reactions
        i, j = reaction[3], reaction[4]
        
        if gen.distribution == "gamma"
            alpha_matrix[i+1, j+1] = rand(gen.rng, Gamma(shape, scale))
        elseif gen.distribution == "levy"
            # Lévy distribution in Julia
            d = Levy(shape, scale)
            alpha_matrix[i+1, j+1] = rand(gen.rng, d)
        else
            error("Unsupported distribution: $(gen.distribution)")
        end
    end
    
    @assert size(alpha_matrix) == (gen.N, gen.N) "alpha_matrix shape must be ($(gen.N), $(gen.N))"
    
    return alpha_matrix
end


"""
    beta_parameter_generation(gen::AllParameterGeneration) -> Matrix{Float64}

Generate beta transition rate matrix.
"""
function beta_parameter_generation(gen::AllParameterGeneration)::Matrix{Float64}
    _, valid_Y_reactions = calculate_valid_transitions(gen)
    
    shape, scale = gen.params
    beta_matrix = zeros(gen.N, gen.N)
    
    for reaction in valid_Y_reactions
        i, j = reaction[3], reaction[4]
        
        if gen.distribution == "gamma"
            beta_matrix[i+1, j+1] = rand(gen.rng, Gamma(shape, scale))
        elseif gen.distribution == "levy"
            d = Levy(shape, scale)
            beta_matrix[i+1, j+1] = rand(gen.rng, d)
        else
            error("Unsupported distribution: $(gen.distribution)")
        end
    end
    
    @assert size(beta_matrix) == (gen.N, gen.N) "beta_matrix shape must be ($(gen.N), $(gen.N))"
    
    return beta_matrix
end


"""
    k_parameter_generation(gen::AllParameterGeneration) -> Tuple{Vector{Float64}, Vector{Float64}}

Generate k positive and negative rate parameters.
"""
function k_parameter_generation(gen::AllParameterGeneration)
    shape, scale = gen.params
    
    if gen.distribution == "gamma"
        k_positive_rates = rand(gen.rng, Gamma(shape, scale), gen.N - 1)
        k_negative_rates = rand(gen.rng, Gamma(shape, scale), gen.N - 1)
    elseif gen.distribution == "levy"
        d = Levy(shape, scale)
        k_positive_rates = rand(gen.rng, d, gen.N - 1)
        k_negative_rates = rand(gen.rng, d, gen.N - 1)
    else
        error("Unsupported distribution: $(gen.distribution)")
    end
    
    return k_positive_rates, k_negative_rates
end


"""
    p_parameter_generation(gen::AllParameterGeneration) -> Tuple{Vector{Float64}, Vector{Float64}}

Generate p positive and negative rate parameters.
"""
function p_parameter_generation(gen::AllParameterGeneration)
    shape, scale = gen.params
    
    if gen.distribution == "gamma"
        p_positive_rates = rand(gen.rng, Gamma(shape, scale), gen.N - 1)
        p_negative_rates = rand(gen.rng, Gamma(shape, scale), gen.N - 1)
    elseif gen.distribution == "levy"
        d = Levy(shape, scale)
        p_positive_rates = rand(gen.rng, d, gen.N - 1)
        p_negative_rates = rand(gen.rng, d, gen.N - 1)
    else
        error("Unsupported distribution: $(gen.distribution)")
    end
    
    return p_positive_rates, p_negative_rates
end


"""
    total_concentration_generation(gen::AllParameterGeneration) -> Tuple{Float64, Float64}

Generate total x and y concentrations.
"""
function total_concentration_generation(gen::AllParameterGeneration)
    shape, scale = gen.params
    
    if gen.distribution == "gamma"
        x_tot_concentration = rand(gen.rng, Gamma(shape, scale))
        y_tot_concentration = rand(gen.rng, Gamma(shape, scale))
    else
        error("Only gamma distribution supported for total concentrations")
    end
    
    return x_tot_concentration, y_tot_concentration
end


"""
    guess_generator(n::Int, a_tot::Float64, x_tot::Float64, y_tot::Float64) -> Vector{Float64}

Generate random initial guess for ODE solver.
"""
function guess_generator(n::Int, a_tot::Float64, x_tot::Float64, y_tot::Float64)::Vector{Float64}
    N = 2^n
    
    # Generate b and x splits
    bx_splits = sort([rand() * x_tot for _ in 1:N-1])
    bx_values = vcat([bx_splits[1]], 
                     [bx_splits[i] - bx_splits[i-1] for i in 2:N-1], 
                     [x_tot - bx_splits[end]])
    b_guess = bx_values[1:N-1]
    
    # Generate c and y splits
    cy_splits = sort([rand() * y_tot for _ in 1:N-1])
    cy_values = vcat([cy_splits[1]], 
                     [cy_splits[i] - cy_splits[i-1] for i in 2:N-1], 
                     [y_tot - cy_splits[end]])
    c_guess = cy_values[1:N-1]
    
    # Generate a splits
    total = a_tot - sum(b_guess) - sum(c_guess)
    a_splits = sort([rand() * total for _ in 1:N-1])
    a_values = vcat([a_splits[1]], 
                    [a_splits[i] - a_splits[i-1] for i in 2:N-1], 
                    [total - a_splits[end]])
    a_guess = a_values[1:N-1]
    
    return vcat(a_guess, b_guess, c_guess)
end


"""
    is_duplicate(candidate::Vector{Float64}, collection::Vector{Vector{Float64}}, 
                 euclidean_distance_tol::Float64) -> Bool

Check if candidate vector is close to any vector in collection.
"""
function is_duplicate(candidate::Vector{Float64}, 
                     collection::Vector{Vector{Float64}}, 
                     euclidean_distance_tol::Float64)::Bool
    if isempty(collection)
        return false
    end
    
    for v in collection
        if (norm(v - candidate) < euclidean_distance_tol) && 
           isapprox(v, candidate, atol=1e-10)
            return true
        end
    end
    
    return false
end


"""
    matrix_clip(matrix::Matrix{Float64}, rate_min::Float64, rate_max::Float64) -> Matrix{Float64}

Clip non-zero matrix elements to [rate_min, rate_max].
"""
function matrix_clip(matrix::Matrix{Float64}, rate_min::Float64, rate_max::Float64)::Matrix{Float64}
    clipped = copy(matrix)
    mask = clipped .!= 0.0
    clipped[mask] .= clamp.(clipped[mask], rate_min, rate_max)
    return clipped
end

function main()
    n = 2
    new_shape_parameters = [1e6, 1e6]
    gen_rates = AllParameterGeneration(n, "distributive", "gamma", new_shape_parameters, verbose=false)
    alpha_matrix = alpha_parameter_generation(gen_rates)
    println("Alpha Matrix:")
    display(alpha_matrix)
    println()
end


# Run main if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end