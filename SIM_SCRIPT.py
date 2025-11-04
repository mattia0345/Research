import numpy as np
import sympy as sp
from typing import List, Tuple, Dict, Any, Set
from scipy.stats import levy
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
import pickle
from PHOS_FUNCTIONS import MATTIA_FULL, MATTIA_REDUCED, MATTIA_REDUCED_JACOBIAN, MATRIX_FINDER
from PHOS_FUNCTIONS import ALL_GUESSES, duplicate, stability_calculator, matrix_clip, all_parameter_generation

def stable_event(t, state_array, n, x_tot, y_tot, L1, L2, W1, W2):
    a_dot = MATTIA_REDUCED(t, state_array, n, x_tot, y_tot, L1, L2, W1, W2)
    # Avoid division by zero using a small epsilon
    denom = np.maximum(np.abs(state_array), 1e-12)
    rel_change = np.max(np.abs(a_dot / denom))
    return rel_change - 1e-4  # event when relative change < 1e-6

stable_event.terminal = True
stable_event.direction = 0

def phosphorylation_system_solver(parameters_tuple, initial_states_array, final_t, full_model, plot):

    n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates, a_tot, x_tot, y_tot = parameters_tuple
    N = n + 1

    assert np.all(initial_states_array <= (a_tot + 1e-12))
    assert np.all(initial_states_array >= 0)
    assert np.all(initial_states_array <= a_tot)

    #### SCALING TIME
    initial_t = 0
    # final_t = 2000
    t_span = (initial_t, final_t)

    ###### OBTAINING ALL MATRICES
    Kp, Pp, G, H, Q, M_mat, D, N_mat, L1, L2, W1, W2 = MATRIX_FINDER(n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates)
    cmap = plt.get_cmap('tab10')
    def color_for_species(idx):
        return cmap(idx % cmap.N)
    
    # default
    abstol = 1e-20
    reltol = 1e-5

    if full_model == False:

        assert len(initial_states_array) == N

        mattia_reduced_parameter_tuple = (n, x_tot, y_tot, L1, L2, W1, W2)

        # ‘DOP853’ is recommended for solving with high precision (low values of rtol and atol).
        # method = 'DOP853'
        # if stiff, you should use ‘Radau’ or ‘BDF’. ‘LSODA’ can also be a good universal choice, 
        # but it might be somewhat less convenient to work with as it wraps old Fortran code.
        # 'LSODA' does not accept array-like jacobian

        method = 'LSODA'
        sol = solve_ivp(MATTIA_REDUCED, t_span = t_span, y0=np.asarray(initial_states_array, dtype=float),
                    args = mattia_reduced_parameter_tuple, method = method, jac = MATTIA_REDUCED_JACOBIAN, 
                    atol = abstol, rtol = reltol)
        print(sol.message)
        # sol = solve_ivp(MATTIA_REDUCED, t_span = t_span, y0=np.asarray(initial_states_array, dtype=float),
        #             args = mattia_reduced_parameter_tuple, method = 'LSODA', jac = MATTIA_REDUCED_JACOBIAN, 
        #             events = stable_event, atol = abstol, rtol = reltol)
        a_solution_stack = np.stack([sol.y[i] for i in range(0, N)]) / a_tot

        if plot == True:
            plt.figure(figsize = (8, 6))
            plt.style.use('seaborn-v0_8-whitegrid')
            for i in range(a_solution_stack.shape[0]):
                color = color_for_species(i)
                plt.plot(sol.t, a_solution_stack[i], color=color, label = f"[$A_{i}]$", lw=4, alpha = 0.4)
                print(f"final A_{i} = {a_solution_stack[i][-1]}")
                plt.title(f"Plotting reduced phosphorylation dynamics for n = {n}")

    # if full_model == True:
    #     assert len(initial_states_array) == 3*n + 3
    #     # assert np.all(initial_states_array >= 1e-15)
    #     assert np.all(initial_states_array >= 0)

    #     mattia_parameter_tuple = (n, Kp, Pp, G, H, Q, M_mat, D, N_mat)
    #     # t_array = np.linspace(t_span[0], t_span[1], 500)
    #     sol = solve_ivp(MATTIA_FULL, t_span = t_span, y0=np.asarray(initial_states_array, dtype=float), 
    #                     args = mattia_parameter_tuple, method = 'LSODA', atol = abstol, rtol = reltol)

    #     a_solution_stack = np.stack([sol.y[i] for i in range(0, N)]) / a_tot



        # b_solution_stack = np.stack([sol.y[i] for i in range(N, 2*N - 1)]) 
        # c_solution_stack = np.stack([sol.y[i] for i in range(2*N - 1, 3*N - 2)]) 
        # x_solution = sol.y[-2]
        # y_solution = sol.y[-1]
        # for i in range(b_solution_stack.shape[0]):
        #     color = color_for_species(i + a_solution_stack.shape[0])
        #     plt.plot(sol.t, b_solution_stack[i], color=color, label = f"$B_{i}$", lw=1.5, linestyle='-', alpha = 0.75)
        #     print(f"final B_{i} = {b_solution_stack[i][-1]}")

        # for i in range(c_solution_stack.shape[0]):
        #     color = color_for_species(i + a_solution_stack.shape[0] + b_solution_stack.shape[0] - 1)
        #     plt.plot(sol.t, c_solution_stack[i], color=color, label=f"$C_{i+1}$", lw=1, linestyle='--', alpha = 1)
        #     print(f"final C_{i+1} = {c_solution_stack[i][-1]}")

        # print(f"final X = {x_solution[-1]}")
        # print(f"final Y = {y_solution[-1]}")
        # print(a_solution_stack)

        for i in range(a_solution_stack.shape[0]):
            color = color_for_species(i)
            plt.plot(sol.t, a_solution_stack[i], color=color, label = f"$[A_{i}]$", lw=4, alpha = 0.4)
            print(f"final numerical fp of A_{i}: {a_solution_stack[i][-1]}")
            plt.title(f"full system dynamics for n = {n}")
        # plt.plot(sol.t, x_solution, color='black', label="$X$", lw=1.75, alpha = 0.75)
        # plt.plot(sol.t, y_solution, color='gray', label="$Y$", lw=1, alpha = 0.75)
    # plt.figure(figsize = (8, 6))

    if plot == True:
        plt.ylabel("concentration")
        plt.xlabel("time")
        plt.minorticks_on()
        plt.tight_layout()
        plt.xlim(t_span[0] - 0.1, t_span[1] + 0.1)
        plt.ylim(-0.05, 1.1)
        plt.legend(frameon=False)
        plt.show()
    
    return sol.y.T[-1]

def fp_finder(n, a_tot_value, x_tot_value, y_tot_value,
                      alpha_matrix, beta_matrix,
                      k_positive_rates, k_negative_rates,
                      p_positive_rates, p_negative_rates):
    N = n + 1
    a_tot_value = 1
    Kp, Pp, G, H, Q, M_mat, D, N_mat, L1, L2, W1, W2 = MATRIX_FINDER(n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates)

    norm_tol = 1e-8
    final_t = 100000
    grid_num = 8
    stable_points = []
    plot = False
    full_system = False

    attempt_total_num = 10
    guesses = []
    guesses = ALL_GUESSES(N, guesses)
    for i in range(attempt_total_num):
        guess = np.random.rand(N)
        guesses.append(guess / np.sum(guess))

    for guess in guesses:
        # rand_guess = np.random.rand(N)
        # rand_guess = rand_guess / np.sum(rand_guess)
        # rand_guess[-1] = 1 - np.sum(rand_guess[0:])

        # small numerical safety:

        # clip tiny negative to zero and renormalize if necessary
        initial_states_array = guess
        parameters_tuple = (n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates, a_tot_value, x_tot_value, y_tot_value)            
        full_sol = phosphorylation_system_solver(parameters_tuple, initial_states_array, final_t, full_system, plot)

        residuals = MATTIA_REDUCED(1, full_sol, n, x_tot_value, y_tot_value, L1, L2, W1, W2)
        if np.max(np.abs(residuals)) > 1e-8:
            print("Point was not found to be a valid zero.")
            continue

        if duplicate(full_sol, stable_points, norm_tol) == True:
            print("Duplicate found, skipping")
            continue

        stable_points.append(full_sol)

    if len(stable_points) == 0:
        return np.array([])
        
    print(f"# of stable points found is {len(stable_points)}")

    return np.array(stable_points)

def process_sample_script(i,
                   n,
                   a_tot_value,
                   x_tot_value_parameter_array,
                   y_tot_value_parameter_array,
                   alpha_matrix_parameter_array,
                   beta_matrix_parameter_array,
                   k_positive_parameter_array,
                   k_negative_parameter_array,
                   p_positive_parameter_array,
                   p_negative_parameter_array):
    
    alpha_matrix = alpha_matrix_parameter_array[i]
    beta_matrix = beta_matrix_parameter_array[i]
    k_positive_rates = k_positive_parameter_array[i]
    k_negative_rates = k_negative_parameter_array[i]
    p_positive_rates = p_positive_parameter_array[i]
    p_negative_rates = p_negative_parameter_array[i]
    x_tot_value = x_tot_value_parameter_array[i][0]
    y_tot_value = y_tot_value_parameter_array[i][0]
    
    rate_min, rate_max = 1e-1, 1e7
    k_positive_rates = k_positive_rates / np.mean(k_positive_rates)
    p_positive_rates = p_positive_rates / np.mean(p_positive_rates)
    k_positive_rates = np.clip(k_positive_rates, rate_min, rate_max)
    p_positive_rates = np.clip(p_positive_rates, rate_min, rate_max)
    k_negative_rates = k_positive_rates
    p_negative_rates = p_positive_rates

    # normalize
    alpha_matrix = alpha_matrix / np.mean(alpha_matrix_parameter_array[i])
    beta_matrix = alpha_matrix / np.mean(beta_matrix_parameter_array[i])
    
    # clipping
    alpha_matrix = matrix_clip(alpha_matrix); beta_matrix = matrix_clip(beta_matrix)

    multistable_results = None
    
    unique_stable_fp_array = fp_finder(n, a_tot_value, x_tot_value, y_tot_value,
                                                     alpha_matrix, beta_matrix, k_positive_rates,
                                                     k_negative_rates, p_positive_rates, p_negative_rates)
    print(unique_stable_fp_array)
    possible_steady_states = np.floor((n + 2) / 2).astype(int)

    if len(unique_stable_fp_array) == 2:
        multistable_results = {
        "num_of_stable_states": len(unique_stable_fp_array),
        "stable_states": unique_stable_fp_array,
        "total_concentration_values": np.array([a_tot_value, x_tot_value, y_tot_value]),
        "alpha_matrix": alpha_matrix,
        "beta_matrix": beta_matrix,
        "k_positive_rates": k_positive_rates,
        "k_negative_rates": k_negative_rates,
        "p_positive_rates": p_positive_rates,
        "p_negative_rates": p_negative_rates,
        # "alpha_matrix": alpha_matrix_parameter_array[i],
        # "beta_matrix": beta_matrix_parameter_array[i],
        # "k_positive": k_positive_parameter_array[i],
        # "k_negative": k_negative_parameter_array[i],
        # "p_positive": p_positive_parameter_array[i],
        # "p_negative": p_negative_parameter_array[i],
        }

    return multistable_results

def simulation_root(n, simulation_size):
    a_tot_value = 1
    N = n + 1

    gen_rates = all_parameter_generation(n, "distributive", "gamma", (0.123, 4.46e6), verbose = False)
    rate_min, rate_max = 1e-1, 1e7
    alpha_matrix_parameter_array = np.array([gen_rates.alpha_parameter_generation() for _ in range(simulation_size)]) # CANNOT CLIP
    beta_matrix_parameter_array = np.array([gen_rates.beta_parameter_generation() for _ in range(simulation_size)]) # CANNOT CLIP
    # k_positive_parameter_array = np.array([np.ones(N - 1) for _ in range(simulation_size)])
    # k_negative_parameter_array = np.array([np.ones(N - 1) for _ in range(simulation_size)])
    # p_positive_parameter_array = np.array([np.ones(N - 1) for _ in range(simulation_size)])
    # p_negative_parameter_array = np.array([np.ones(N - 1) for _ in range(simulation_size)])
    k_positive_parameter_array = np.array([gen_rates.k_parameter_generation()[0] for _ in range(simulation_size)])
    k_negative_parameter_array = np.array([gen_rates.k_parameter_generation()[1] for _ in range(simulation_size)])
    p_positive_parameter_array = np.array([gen_rates.p_parameter_generation()[0] for _ in range(simulation_size)])
    p_negative_parameter_array = np.array([gen_rates.p_parameter_generation()[1] for _ in range(simulation_size)])
    gen_concentrations = all_parameter_generation(n, "distributive", "gamma", (0.40637, 0.035587), verbose = False)
    concentration_min, concentration_max = 1e-4, 1e-1
    x_tot_value_parameter_array = np.array([np.clip(gen_concentrations.total_concentration_generation()[0], concentration_min, concentration_max) for _ in range(simulation_size)])
    y_tot_value_parameter_array = np.array([np.clip(gen_concentrations.total_concentration_generation()[1], concentration_min, concentration_max) for _ in range(simulation_size)])

    results = Parallel(n_jobs=-2, backend="loky")(
        delayed(process_sample_script)(
            i,
            n,
            a_tot_value,
            x_tot_value_parameter_array,
            y_tot_value_parameter_array,
            alpha_matrix_parameter_array,
            beta_matrix_parameter_array,
            k_positive_parameter_array,
            k_negative_parameter_array,
            p_positive_parameter_array,
            p_negative_parameter_array
        ) for i in range(simulation_size)
    )
    
    multistable_results_list = []

    for multi_res in results:
        if multi_res is not None:
            multistable_results_list.append(multi_res)
    multifile = f"multistability_parameters_{simulation_size}_{n}.pkl"

    with open(multifile, "wb") as f:
        pickle.dump(multistable_results_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    return

def main():
    n = 2
    simulation_root(n, 20000)
    
if __name__ == "__main__":
    main()