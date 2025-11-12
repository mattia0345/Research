import numpy as np
import sympy as sp
from typing import List, Tuple, Dict, Any, Set
from scipy.stats import levy
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
import pickle
import random
from scipy.optimize import approx_fprime
from scipy.differentiate import jacobian
from scipy.stats import reciprocal
from numpy import linalg as LA
from sklearn.cluster import DBSCAN
from PHOS_FUNCTIONS import MATRIX_FINDER, MATTIA_FULL, guess_generator, matrix_clip, matrix_normalize, matrix_sample_reciprocal
from PHOS_FUNCTIONS import duplicate, all_parameter_generation

def stable_event(t, state_array, *args):
    n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat = args

    state_dot = MATTIA_FULL(t, state_array, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)

    eps = 1e-10           
    threshold = 1e-10 

    # eps = 1e-12           
    # threshold = 1e-12 

    state_norm = np.linalg.norm(state_array)
    dot_norm = np.linalg.norm(state_dot)

    denom = max(state_norm, eps)
    rel_change = dot_norm / denom

    value = rel_change - threshold

    return value

stable_event.terminal = True
stable_event.direction = 0

def phosphorylation_system_solver(parameters_tuple, initial_states_array, final_t, plot):

    n, a_tot, _, _, _, _, _, _, _, _, _, _ = parameters_tuple
    # N = n + 1
    N = 2**n
    assert np.all(initial_states_array >= 0)

    initial_t = 0
    t_span = (initial_t, final_t)
    
    # default
    # abstol = 1e-20
    # reltol = 1e-5
    abstol = 1e-6
    reltol = 1e-3

    assert len(initial_states_array) == 3*N - 3

    method = 'LSODA'
    try:
        sol = solve_ivp(MATTIA_FULL, t_span = t_span, y0=np.asarray(initial_states_array, dtype=float),
            events = stable_event, args = parameters_tuple, method = method, 
            atol = abstol, rtol = reltol)
    except ValueError as e:
        sol = solve_ivp(MATTIA_FULL, t_span = t_span, y0=np.asarray(initial_states_array, dtype=float),
            args = parameters_tuple, method = method, 
            atol = abstol, rtol = reltol)

    if plot == True:
        cmap = plt.get_cmap('Accent')
        def color_for_species(idx):
            return cmap(idx / (3 * N - 3))  # smooth gradient of distinct colors
        a_solution_stack = np.stack([sol.y[i] for i in range(0, N - 1)])
        b_solution_stack = np.stack([sol.y[i] for i in range(N-1, 2*N - 2)]) 
        c_solution_stack = np.stack([sol.y[i] for i in range(2*N - 2, 3*N - 3)])
        a_N_array = a_tot - np.sum(a_solution_stack, axis=0) - np.sum(b_solution_stack, axis = 0) - np.sum(c_solution_stack, axis=0)

        plt.figure(figsize = (8, 6))
        plt.style.use('seaborn-v0_8-whitegrid')
        for i in range(a_solution_stack.shape[0]):
            color = color_for_species(i)
            plt.plot(sol.t, a_solution_stack[i], color=color, label = f"[$A_{i}]$", lw=4, alpha = 0.9)
            # print(f"final A_{i} = {a_solution_stack[i][-1]}")
            plt.title(f"Plotting reduced phosphorylation dynamics for n = {n}")

        color_N = color_for_species(N-1)
        plt.plot(sol.t, a_N_array, color=color_N, label=f"$[A_{{{N-1}}}]$", lw=4, alpha=0.9)
        plt.ylabel("concentration")
        plt.xlabel("time")
        plt.minorticks_on()
        plt.tight_layout()
        if sol.status == 1:
            plt.xlim(t_span[0] - 0.1, sol.t_events[0][0] + 0.1)
        else:
            plt.xlim(t_span[0] - 0.1, t_span[0] + 0.1)
        plt.ylim(-0.05, 1.1)
        plt.legend(frameon=False)
        plt.show()

    return sol.y.T[-1]

def fp_finder(n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat):

    N = 2**n

    # norm_tol = 0.00001
    final_t = 100000000
    stable_points = []
    plot = False

    attempt_total_num = 8
    guesses = []
    # guesses.append(1e-14*np.ones(3*N - 3))
    # guesses.append(np.concatenate([np.array([1-1e-14]), np.zeros(3*N-4)]))
    guesses.append(np.zeros(3*N - 3))
    guesses.append(1e-5*np.ones(3*N - 3))
    guesses.append(np.concatenate([np.array([1]), np.zeros(3*N-4)]))

    for i in range(5):
        t = np.random.rand(N)
        t /= t.sum()
        t = t[:-1]
        guesses.append(np.concatenate([t, np.zeros(2*N-2)]))

    for i in range(attempt_total_num):
        guesses.append(np.concatenate([np.array([random.random()]), np.zeros(3*N-4)]))
        guesses.append(guess_generator(n, a_tot, x_tot, y_tot))


    for guess in guesses:

        initial_states_array = np.array(guess)
        parameters_tuple = (n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)      
        reduced_sol = phosphorylation_system_solver(parameters_tuple, initial_states_array, final_t, plot)

        residuals = MATTIA_FULL(0, reduced_sol, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)
    
        if not np.all(np.isfinite(residuals)):
            print(f"Residuals contain NaN or inf values, skipping.")
            continue

        if np.max(np.abs(residuals)) > 1e-8:
            print(f"Point {np.max(np.abs(residuals))} was not found to be a valid zero.")
            continue

        zero_tol = 1e-8
        if np.any(reduced_sol < -zero_tol):
            # print(reduced_sol)
            print("Non-physical fixed point.")
            continue

        def mattia_full_wrapper(state):
            return MATTIA_FULL(0, state, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)

        # jacobian_mattia_full = np.array([
        #     approx_fprime(reduced_sol, lambda s: mattia_full_wrapper(s)[i], epsilon=1e-8)
        #     for i in range(len(reduced_sol))])
        
        # eigenvalues = LA.eigvals(jacobian_mattia_full)
        # eigenvalues_real = np.real(eigenvalues)
        # print(f"Eigenvalues at fixed point: {eigenvalues_real}")

        # if duplicate(reduced_sol, stable_points, norm_tol):
        #     # print("Duplicate found, skipping")
        #     continue

        stable_points.append(reduced_sol)

    stable_points = np.array(stable_points)  # shape (num_trials, n_variables)

    db = DBSCAN(eps=1e-5, min_samples=1).fit(stable_points)
    labels = db.labels_

    # Extract representative points
    unique_points = np.array([stable_points[labels == i][0] for i in np.unique(labels)])
    print(f"Found {len(unique_points)} unique steady states.")

    if len(unique_points) == 0:
        return np.array([])
            
    return unique_points

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
    
    N = 2**n
    alpha_matrix = alpha_matrix_parameter_array[i]
    beta_matrix = beta_matrix_parameter_array[i]
    k_positive_rates = k_positive_parameter_array[i]
    k_negative_rates = k_negative_parameter_array[i]
    p_positive_rates = p_positive_parameter_array[i]
    p_negative_rates = p_negative_parameter_array[i]
    x_tot_value = x_tot_value_parameter_array[i][0]
    y_tot_value = y_tot_value_parameter_array[i][0]

    rate_min, rate_max = 1e-1, 1e7

    # k_positive_rates = k_positive_rates / np.mean(k_positive_rates)
    # k_negative_rates = k_negative_rates / np.mean(k_negative_rates)
    # p_positive_rates = p_positive_rates / np.mean(p_positive_rates)
    # p_negative_rates = p_negative_rates / np.mean(p_negative_rates)
    # alpha_matrix = matrix_normalize(alpha_matrix)
    # beta_matrix = matrix_normalize(beta_matrix)

    # k_positive_rates = np.clip(k_positive_rates, rate_min, rate_max)
    # k_negative_rates = np.clip(k_negative_rates, rate_min, rate_max)
    # p_positive_rates = np.clip(p_positive_rates, rate_min, rate_max)
    # p_negative_rates = np.clip(p_negative_rates, rate_min, rate_max)
    # alpha_matrix = matrix_clip(alpha_matrix, rate_min, rate_max)
    # beta_matrix = matrix_clip(beta_matrix, rate_min, rate_max)
    # print("k+:")
    # print(k_positive_rates)
    # print("alpha:")
    # print(alpha_matrix)
    k_positive_rates = reciprocal(a=rate_min, b=rate_max).rvs(size=N-1)
    k_negative_rates = reciprocal(a=rate_min, b=rate_max).rvs(size=N-1)
    p_positive_rates = reciprocal(a=rate_min, b=rate_max).rvs(size=N-1)
    p_negative_rates = reciprocal(a=rate_min, b=rate_max).rvs(size=N-1)
    alpha_matrix = matrix_sample_reciprocal(alpha_matrix, rate_min, rate_max)
    beta_matrix = matrix_sample_reciprocal(beta_matrix, rate_min, rate_max)
    # print("k+:")
    # print(k_positive_rates)
    # print("alpha:")
    # print(alpha_matrix)
    multistable_results = None

    Kp, Pp, G, H, Q, M_mat, D, N_mat = MATRIX_FINDER(n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates)

    unique_stable_fp_array = fp_finder(n, a_tot_value, x_tot_value, y_tot_value,
                                                    Kp, Pp, G, H, Q, M_mat, D, N_mat)


    possible_steady_states = np.floor((n + 2) / 2).astype(int)
    print(i)
    # if 8 >= len(unique_stable_fp_array) == possible_steady_states:
    # if len(unique_stable_fp_array) == possible_steady_states:
    if len(unique_stable_fp_array) == 2 or len(unique_stable_fp_array) == 3:

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
        }

    return multistable_results

def simulation(n, simulation_size):
    a_tot_value = 1
    # N = n + 1
    N = 2**n
    x_tot = 1e-3
    y_tot = 1e-3
    old_shape_parameters = (0.123, 4.46e6)
    shape = 1e-1
    new_shape_parameters = (shape, 1 / shape)
    NEWEST_shape_parameters = (1, 10)
    gen_rates = all_parameter_generation(n, "distributive", "gamma", NEWEST_shape_parameters, verbose = False)
    # rate_min, rate_max = 1e-6, 1e3
    alpha_matrix_parameter_array = np.array([gen_rates.alpha_parameter_generation() for _ in range(simulation_size)]) # CANNOT CLIP
    beta_matrix_parameter_array = np.array([gen_rates.beta_parameter_generation() for _ in range(simulation_size)]) # CANNOT CLIP
    k_positive_parameter_array = np.array([gen_rates.k_parameter_generation()[0] for _ in range(simulation_size)])
    k_negative_parameter_array = np.array([gen_rates.k_parameter_generation()[1] for _ in range(simulation_size)])
    p_positive_parameter_array = np.array([gen_rates.p_parameter_generation()[0] for _ in range(simulation_size)])
    p_negative_parameter_array = np.array([gen_rates.p_parameter_generation()[1] for _ in range(simulation_size)])

    # gen_concentrations = all_parameter_generation(n, "distributive", "gamma", (0.40637, 0.035587), verbose = False)
    # concentration_min, concentration_max = 1e-4, 1e-1
    x_tot_value_parameter_array = np.array([x_tot*np.ones(1) for _ in range(simulation_size)])
    y_tot_value_parameter_array = np.array([y_tot*np.ones(1)  for _ in range(simulation_size)])

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
    n = 4
    simulation(n, 1000)
    
if __name__ == "__main__":
    main()