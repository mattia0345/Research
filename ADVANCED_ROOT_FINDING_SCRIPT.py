import numpy as np
from typing import List, Tuple, Dict, Any, Set
from scipy.stats import levy
import matplotlib.pyplot as plt
from scipy.optimize import root
from joblib import Parallel, delayed
import pickle
import random as random
from numpy import linalg as LA
import numpy as np
from scipy.stats import reciprocal
from scipy.optimize import root
from scipy.optimize import approx_fprime
from PHOS_FUNCTIONS import MATTIA_FULL
from PHOS_FUNCTIONS import MATRIX_FINDER, guess_generator, matrix_sample_reciprocal
from PHOS_FUNCTIONS import duplicate, matrix_clip, all_parameter_generation

def MATTIA_FULL_ROOT_JACOBIAN(state_array, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat):
    """Compute the Jacobian of MATTIA_FULL_ROOT numerically"""
    def mattia_full_wrapper(state):
        return MATTIA_FULL_ROOT(state, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)
    
    jacobian_mattia_full = np.array([
        approx_fprime(state_array, lambda s: mattia_full_wrapper(s)[i], epsilon=1e-8)
        for i in range(len(state_array))])
    
    return jacobian_mattia_full

def MATTIA_FULL_ROOT(state_array, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat):

    # including conservation laws
    # state_array = [a0, a1, b0, b1, c1, c2]

    N = 2**n

    assert len(state_array) == 3*N - 3
    a_red = state_array[0: N - 1]
    b = state_array[N - 1: 2*N - 2]
    c = state_array[2*N - 2: 3*N - 3]

    a = np.concatenate([a_red, [a_tot - np.sum(a_red) - np.sum(b) - np.sum(c)]])
    x = x_tot - np.sum(b)
    y = y_tot - np.sum(c)

    a_dot = (G @ b) + (H @ c) - x * (Kp @ a) - y * (Pp @ a)  
    b_dot = x * (Q @ a) - (M_mat @ b)
    c_dot = y * (D @ a) - (N_mat @ c)

    # x_dot = -1*np.sum(b_dot)
    # y_dot = -1*np.sum(c_dot)

    a_dot_red = a_dot[0: N-1]

    # zero_threshold = 1e-15
    # # below_mask = a_red < zero_threshold
    # a_dot_red = np.where(a_red < zero_threshold, np.maximum(a_dot_red, 0), a_dot_red)
    # b_dot = np.where(b < zero_threshold, np.maximum(b_dot, 0), b_dot)
    # c_dot = np.where(c < zero_threshold, np.maximum(c_dot, 0), c_dot)

    return np.concatenate([a_dot_red, b_dot, c_dot])

def deflated_function(vec, roots, params):
    alpha = 1e-2; p = 2;
    M = 1.0
    for r in roots:
        M *= 1.0 / (np.linalg.norm(vec - r)**p + alpha)
    return M * MATTIA_FULL_ROOT(vec, *params)

def root_finder(n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat):

    red_sol_list_stable = []
    red_sol_list_unstable = []
    N = 2**n

    sol_list = []

    guesses = []
    red_eigenvalues_real_list = []
    guesses.append(1e-14*np.ones(3*N - 3))
    guesses.append(np.concatenate([np.array([1-1e-14]), np.zeros(3*N-4)]))
    attempt_total_num = 8

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

    root_tol = 1e-6
    residual_tol = 1e-8
    euclidian_distance_tol = 1e-6
    eigen_value_tol = 1e-8
    
    params = (n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)
    for guess in guesses:
        # assert np.any(np.array(guess) <= 0), f"Guess {guess} is non-physical"
        assert len(guess) == 3*N - 3, f"Guess {guess} is incorrect length"

        try:
            reduced_sol = root(
                lambda v: deflated_function(v, sol_list, params),
                x0=guess,
                # args=(sol_list, params),
                # jac=MATTIA_FULL_ROOT_JACOBIAN,
                method='hybr',
                tol=root_tol,
                options={'maxfev': 5000})
        except TypeError as e:
            continue

        # if all(np.linalg.norm(reduced_sol - r) > 1e-6 for r in reduced_sol):
        #     sol_list.append(reduced_sol)

        # if not reduced_sol.success:
        #     continue

        reduced_sol = reduced_sol.x

        residuals = MATTIA_FULL_ROOT(reduced_sol, *params)
        if np.max(np.abs(residuals)) > residual_tol:
            # reject if residual too large
            continue

        # Reject if non-physical (strong negative tolerated by some tol)
        zero_tol = 1e-10
        if np.any(reduced_sol < -zero_tol):
            # non-physical
            continue

        # Duplicate detection: compare reduced_sol to previously found solutions
        is_duplicate = False
        for r in sol_list:
            if np.linalg.norm(reduced_sol - r) <= euclidian_distance_tol:
                is_duplicate = True
                break
        if is_duplicate:
            continue

        # Accept this root
        sol_list.append(reduced_sol)
        # residuals = MATTIA_FULL_ROOT(reduced_sol, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)

        # if np.max(np.abs(residuals)) > residual_tol:
        #     # print(f"Residuals {residuals} were not found to obey the tolerance {residual_tol}.")
        #     continue

        # zero_tol = 1e-8
        # if np.any(reduced_sol < -zero_tol):
        #     # print(reduced_sol)
        #     print("Non-physical fixed point.")
        #     continue

    for sol in sol_list:

        def mattia_full_wrapper(v):
            return MATTIA_FULL_ROOT(v, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)

        jacobian_mattia_full = np.array([
            approx_fprime(sol, lambda s: mattia_full_wrapper(s)[i], epsilon=1e-8)
            for i in range(len(sol))])
        
        if not np.isfinite(jacobian_mattia_full).all():
            print("infinite detected")
            continue

        red_eigenvalues = LA.eigvals(jacobian_mattia_full)
        red_eigenvalues_real = np.real(red_eigenvalues)

        # print(f"Eigenvalues at fixed point: {red_eigenvalues_real}")
        if np.max(red_eigenvalues_real) > eigen_value_tol:
            print(f"UNSTABLE EIGENVALUES: {red_eigenvalues_real}")
            red_sol_list_unstable.append(sol)
            red_eigenvalues_real_list.append(red_eigenvalues_real)
        elif np.all(red_eigenvalues_real < -eigen_value_tol):
            print(f"STABLE EIGENVALUES: {red_eigenvalues_real}")
            red_sol_list_stable.append(sol)
            red_eigenvalues_real_list.append(red_eigenvalues_real)
        else:
            print(f" -> Marginally stable")

    print(f"\n# of stable points: {len(red_sol_list_stable)}, # of unstable points: {len(red_sol_list_unstable)}")
    # print(f"\n# of fixed points: {len(sol_list)}")
    
    if (len(red_sol_list_stable) == 0) and (len(red_sol_list_unstable) == 0):
        return np.array([]), np.array([]), np.array([])
    
    return np.array(red_sol_list_stable), np.array(red_sol_list_unstable), np.array(red_eigenvalues_real_list)

def process_sample_script(i,
                   n,
                   a_tot_value,
                   x_tot_value,
                   y_tot_value,
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
    rate_min, rate_max = 1e-1, 1e7
    
    k_positive_rates = reciprocal(a=rate_min, b=rate_max).rvs(size=N-1)
    k_negative_rates = reciprocal(a=rate_min, b=rate_max).rvs(size=N-1)
    p_positive_rates = reciprocal(a=rate_min, b=rate_max).rvs(size=N-1)
    p_negative_rates = reciprocal(a=rate_min, b=rate_max).rvs(size=N-1)
    alpha_matrix = matrix_sample_reciprocal(alpha_matrix, rate_min, rate_max)
    beta_matrix = matrix_sample_reciprocal(beta_matrix, rate_min, rate_max)

    multistable_results = None
    
    Kp, Pp, G, H, Q, M_mat, D, N_mat = MATRIX_FINDER(n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates)

    unique_stable_fp_array, unique_unstable_fp_array, eigenvalues_array = root_finder(n, a_tot_value, x_tot_value, y_tot_value,
                                                     Kp, Pp, G, H, Q, M_mat, D, N_mat)
    
    possible_steady_states = np.floor((n + 2) / 2).astype(int)

    # if len(unique_unstable_fp_array) == 1:
    if len(unique_stable_fp_array) == possible_steady_states:
        multistable_results = {
        "num_of_stable_states": len(unique_stable_fp_array),
        "num_of_unstable_states": len(unique_unstable_fp_array),
        "stable_states": np.array([unique_stable_fp_array[k] for k in range(len(unique_stable_fp_array))]),
        "unstable_states": np.array([unique_unstable_fp_array[k] for k in range(len(unique_unstable_fp_array))]),
        "total_concentration_values": np.array([a_tot_value, x_tot_value, y_tot_value]),
        "alpha_matrix": alpha_matrix,
        "beta_matrix": beta_matrix,
        "k_positive_rates": k_positive_rates,
        "k_negative_rates": k_negative_rates,
        "p_positive_rates": p_positive_rates,
        "p_negative_rates": p_negative_rates,
        "eigenvalues": eigenvalues_array
        }

    return multistable_results

def simulation_root(n, simulation_size):
    a_tot = 1
    x_tot = 1e-3
    y_tot = 1e-3
    old_best_gamma_parameters = (0.123, 4.46e6)
    new_gamma_parameters = (1, 10)
    gen_rates = all_parameter_generation(n, "distributive", "gamma", new_gamma_parameters, verbose = False)
    alpha_matrix_parameter_array = np.array([gen_rates.alpha_parameter_generation() for _ in range(simulation_size)]) 
    beta_matrix_parameter_array = np.array([gen_rates.beta_parameter_generation() for _ in range(simulation_size)]) 
    k_positive_parameter_array = np.array([gen_rates.k_parameter_generation()[0] for _ in range(simulation_size)])
    k_negative_parameter_array = np.array([gen_rates.k_parameter_generation()[1] for _ in range(simulation_size)])
    p_positive_parameter_array = np.array([gen_rates.p_parameter_generation()[0] for _ in range(simulation_size)])
    p_negative_parameter_array = np.array([gen_rates.p_parameter_generation()[1] for _ in range(simulation_size)])

    # gen_concentrations = all_parameter_generation(n, "distributive", "gamma", (0.40637, 0.035587), verbose = False)
    x_tot_value_parameter_array = np.array([x_tot*np.ones(1) for _ in range(simulation_size)])
    y_tot_value_parameter_array = np.array([y_tot*np.ones(1)  for _ in range(simulation_size)])

    results = Parallel(n_jobs=-2, backend="loky")(
        delayed(process_sample_script)(
            i,
            n,
            a_tot,
            x_tot,
            y_tot,
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
    simulation_root(n, 100)
    
if __name__ == "__main__":
    main()