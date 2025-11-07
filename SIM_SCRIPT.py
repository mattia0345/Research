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
from PHOS_FUNCTIONS import MATRIX_FINDER, MATTIA_FULL, guess_generator
from PHOS_FUNCTIONS import duplicate, all_parameter_generation

def stable_event(t, state_array, *args):
    n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat = args
    a_dot = MATTIA_FULL(t, state_array, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)
    # Avoid division by zero using a small epsilon
    denom = np.maximum(np.abs(state_array), 1e-12)
    rel_change = np.max(np.abs(a_dot / denom))
    return rel_change - 1e-12  

stable_event.terminal = True
stable_event.direction = 0

def phosphorylation_system_solver(parameters_tuple, initial_states_array, final_t, plot):

    n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates, a_tot, x_tot, y_tot = parameters_tuple
    N = n + 1

    assert np.all(initial_states_array >= 0)
    assert np.all(initial_states_array <= a_tot)

    initial_t = 0
    t_span = (initial_t, final_t)

    ###### OBTAINING ALL MATRICES
    Kp, Pp, G, H, Q, M_mat, D, N_mat, L1, L2, W1, W2 = MATRIX_FINDER(n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates)
    
    # default
    # abstol = 1e-20
    # reltol = 1e-5
    abstol = 1e-8
    reltol = 1e-4

    assert len(initial_states_array) == 3*N - 3

    mattia_full_parameter_tuple = (n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)

    method = 'LSODA'
    sol = solve_ivp(MATTIA_FULL, t_span = t_span, y0=np.asarray(initial_states_array, dtype=float),
                args = mattia_full_parameter_tuple, method = method, atol = abstol, rtol = reltol)
    return sol.y.T[-1]

def fp_finder(n, a_tot, x_tot, y_tot,
                      alpha_matrix, beta_matrix,
                      k_positive_rates, k_negative_rates,
                      p_positive_rates, p_negative_rates):
    N = n + 1

    Kp, Pp, G, H, Q, M_mat, D, N_mat, L1, L2, W1, W2 = MATRIX_FINDER(n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates)

    norm_tol = 0.1
    final_t = 1000000
    stable_points = []
    plot = False

    attempt_total_num = int(1.75*N)
    guesses = []

    # guesses.append(np.zeros(3*N-3))
    # guesses.append(np.concatenate([np.array([0.98]), np.zeros(3*N-4)]))

    for i in range(attempt_total_num):
        # guesses.append(np.concatenate([np.array([random.random()]), np.zeros(3*N-4)]))
        guesses.append(np.concatenate([np.array([random.random()]), np.zeros(3*N-4)]))
        guesses.append(guess_generator(n, a_tot, x_tot, y_tot))

    for guess in guesses:

        initial_states_array = np.array(guess)
        parameters_tuple = (n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates, a_tot, x_tot, y_tot)            
        reduced_sol = phosphorylation_system_solver(parameters_tuple, initial_states_array, final_t, plot)

        residuals = MATTIA_FULL(0, reduced_sol, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)
        if np.max(np.abs(residuals)) > 1e-8:
            print("Point was not found to be a valid zero.")
            continue

        def mattia_full_wrapper(state):
            return MATTIA_FULL(0, state, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)

        jacobian_mattia_full = np.array([
            approx_fprime(reduced_sol, lambda s: mattia_full_wrapper(s)[i], epsilon=1e-8)
            for i in range(len(reduced_sol))])
        
        eigenvalues = LA.eigvals(jacobian_mattia_full)
        eigenvalues_real = np.real(eigenvalues)
        # print(f"Eigenvalues at fixed point: {eigenvalues_real}")

        if duplicate(reduced_sol, stable_points, norm_tol) == False:
            # print("Duplicate found, skipping")
            stable_points.append(reduced_sol)
            continue

    # def remove_near_duplicates(data, tol=1e-10):
    #     """Remove near-duplicate elements from a list of lists or arrays."""
    #     unique = []
    #     for a in data:
    #         a = np.asarray(a, dtype=float)  # ensures consistent numeric comparison
    #         if not any(np.allclose(a, np.asarray(b, dtype=float), atol=tol, rtol=0) for b in unique):
    #             unique.append(a.tolist())  # store as plain lists for easier printing
    #     return unique
    
    # stable_points = remove_near_duplicates(stable_points)

    print(f"# of stable points found is {len(stable_points)}")

    if len(stable_points) == 0:
        return np.array([])
        
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
    N = n + 1
    alpha_matrix = alpha_matrix_parameter_array[i]
    beta_matrix = beta_matrix_parameter_array[i]
    k_positive_rates = k_positive_parameter_array[i]
    k_negative_rates = k_negative_parameter_array[i]
    p_positive_rates = p_positive_parameter_array[i]
    p_negative_rates = p_negative_parameter_array[i]
    x_tot_value = x_tot_value_parameter_array[i][0]
    y_tot_value = y_tot_value_parameter_array[i][0]

    rate_min, rate_max = 1e-1, 1e2

    # k_positive_rates = k_positive_rates / np.mean(k_positive_rates)
    # k_positive_rates = np.clip(k_positive_rates, rate_min, rate_max)
    # k_negative_rates = k_negative_rates / np.mean(k_negative_rates)
    # k_negative_rates = np.clip(k_negative_rates, rate_min, rate_max)

    # p_positive_rates = p_positive_rates / np.mean(p_positive_rates)
    # p_positive_rates = np.clip(p_positive_rates, rate_min, rate_max)
    # p_negative_rates = p_negative_rates / np.mean(p_negative_rates)
    # p_negative_rates = np.clip(p_negative_rates, rate_min, rate_max)

    # alpha_matrix = alpha_matrix / np.mean(alpha_matrix)
    # alpha_matrix = matrix_clip(alpha_matrix, rate_min, rate_max)
    # beta_matrix = beta_matrix / np.mean(beta_matrix)
    # beta_matrix = matrix_clip(beta_matrix, rate_min, rate_max)

    multistable_results = None

    # k_positive_rates = np.array([26.804, 49.1102])
    # k_negative_rates = np.array([0.21392, 0.03212])
    # p_positive_rates = np.array([2.0276, 92.0147])
    # p_negative_rates = np.array([1.4041390815765653, 0.14205077577290087])
    # alpha_matrix = np.array([[0, 0.0110415, 0],
    #                          [0, 0, 7.828938],
    #                          [0, 0, 0]])
    # beta_matrix = np.array([[0, 0, 0],
    #                         [0.117683, 0, 0],
    #                         [0, 1.84386, 0]])
    
    # k_positive_rates = np.array([3.2511345282947435, 5.0660064823006525])
    # k_negative_rates = np.array([0.013168070763133404, 1.9918361184817839])
    # p_positive_rates = np.array([5.544441039779104, 63.48549752309308])
    # p_negative_rates = np.array([1.4041390815765653, 0.14205077577290087])
    # alpha_matrix = np.array([[0, 0.03522122035395812, 0],
    #                          [0, 0, 0.08029528726005944],
    #                          [0, 0, 0]])
    # beta_matrix = np.array([[0, 0, 0],
    #                         [16.037025320671066, 0, 0],
    #                         [0, 0.020117127980135444, 0]])
    
    # a_tot_value = 1
    # x_tot_value = 1e-3  # Add this
    # y_tot_value = 1e-3  # Add this

    k_positive_rates = reciprocal(a=rate_min, b=rate_max).rvs(size=N-1)
    k_negative_rates = reciprocal(a=rate_min, b=rate_max).rvs(size=N-1)    
    p_positive_rates = reciprocal(a=rate_min, b=rate_max).rvs(size=N-1)
    p_negative_rates = reciprocal(a=rate_min, b=rate_max).rvs(size=N-1)

    alpha_matrix = np.diag(reciprocal(a=rate_min, b=rate_max).rvs(size=N-1), k = 1)
    beta_matrix = np.diag(reciprocal(a=rate_min, b=rate_max).rvs(size=N-1), k = -1)
    unique_stable_fp_array = fp_finder(n, a_tot_value, x_tot_value, y_tot_value,
                                                     alpha_matrix, beta_matrix, k_positive_rates,
                                                     k_negative_rates, p_positive_rates, p_negative_rates)
    # print(unique_stable_fp_array)
    possible_steady_states = np.floor((n + 2) / 2).astype(int)
    print(i)
    if len(unique_stable_fp_array) == possible_steady_states:
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
        # "k_positive_rates": k_positive_parameter_array[i],
        # "k_negative_rates": k_negative_parameter_array[i],
        # "p_positive_rates": p_positive_parameter_array[i],
        # "p_negative_rates": p_negative_parameter_array[i],
        }

    return multistable_results

def simulation(n, simulation_size):
    a_tot_value = 1
    N = n + 1
    x_tot = 1e-3
    y_tot = 1e-3
    gen_rates = all_parameter_generation(n, "distributive", "gamma", (0.123, 4.46e6), verbose = False)
    rate_min, rate_max = 1e-1, 1e7
    alpha_matrix_parameter_array = np.array([gen_rates.alpha_parameter_generation() for _ in range(simulation_size)]) # CANNOT CLIP
    beta_matrix_parameter_array = np.array([gen_rates.beta_parameter_generation() for _ in range(simulation_size)]) # CANNOT CLIP
    k_positive_parameter_array = np.array([gen_rates.k_parameter_generation()[0] for _ in range(simulation_size)])
    k_negative_parameter_array = np.array([gen_rates.k_parameter_generation()[1] for _ in range(simulation_size)])
    p_positive_parameter_array = np.array([gen_rates.p_parameter_generation()[0] for _ in range(simulation_size)])
    p_negative_parameter_array = np.array([gen_rates.p_parameter_generation()[1] for _ in range(simulation_size)])
    gen_concentrations = all_parameter_generation(n, "distributive", "gamma", (0.40637, 0.035587), verbose = False)
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
    n = 2
    simulation(n, 500)
    
if __name__ == "__main__":
    main()