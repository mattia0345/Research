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
from scipy.optimize import root, least_squares
from PHOS_FUNCTIONS import MATTIA_FULL, MATTIA_FULL_ROOT_JACOBIAN
from PHOS_FUNCTIONS import MATRIX_FINDER, matrix_sample_reciprocal
from PHOS_FUNCTIONS import all_parameter_generation
import argparse
import json

def deflated_function(vec, roots, params):
    alpha = 1; p = 2
    M = 1
    for r in roots:
        M *= 1 / (np.linalg.norm(vec - r)**p + alpha)
    return M * MATTIA_FULL_ROOT(vec, *params)

def deflated_jacobian(x, roots, params):

    alpha = 1; p = 2

    x = np.asarray(x)
    n = x.size

    Jf = MATTIA_FULL_ROOT_JACOBIAN(x, *params)  # shape (n,n)
    fx = MATTIA_FULL_ROOT(x, *params)           # shape (n,)

    # if no roots, it's trivial
    if len(roots) == 0:
        return Jf

    M = 1.0
    s_list = []
    d_list = []
    delta_list = []
    g_list = []
    for r in roots:
        delta = x - r
        s = np.linalg.norm(delta)
        d = s**p
        g = (1.0 / d) + alpha  # Form A
        s_list.append(s)
        d_list.append(d)
        delta_list.append(delta)
        g_list.append(g)
        M *= g

    # compute gradM = M * sum_i ( - p * s_i^(p-2) * delta_i / ( d_i * (1 + alpha * d_i) ) )
    grad_sum = np.zeros(n, dtype=float)
    for s, d, delta, g in zip(s_list, d_list, delta_list, g_list):
        if s == 0.0:
            continue
        denom = d * (1.0 + alpha * d)
        coeff = -p * (s**(p-2)) / denom
        grad_sum += coeff * delta
    gradM = M * grad_sum  # length n

    outer = np.outer(fx, gradM)  # shape (n,n)
    JF = M * Jf + outer
    return JF

def jac_for_root(x, roots, params):
    return deflated_jacobian(x, roots, params)

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

    a_dot_red = a_dot[0: N-1]

    return np.concatenate([a_dot_red, b_dot, c_dot])

def root_finder(n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat):

    red_sol_list_stable = []
    red_sol_list_unstable = []
    N = 2**n

    sol_list = []

    guesses = []
    red_eigenvalues_real_list = []
    # guesses.append(1e-14*np.ones(3*N - 3))
    # guesses.append(np.concatenate([np.array([1-1e-14]), np.zeros(3*N-4)]))
    attempt_total_num = 20

    guesses.append(np.zeros(3*N - 3))
    guesses.append(1e-5*np.ones(3*N - 3))
    guesses.append(np.concatenate([np.array([a_tot]), np.zeros(3*N-4)]))

    # THIS SEEMS TO REALLY HELP THE ROOT FINDER FIND ROOTS
    for i in range(20):
        t = np.random.rand(N)
        t /= t.sum()
        t = t[:-1]
        guesses.append(np.concatenate([t, np.zeros(2*N-2)]))


    a_0_ic = np.linspace(1e-10, a_tot - 1e-10, attempt_total_num)
    for i in a_0_ic:
        guesses.append(np.concatenate([np.array([i]), np.zeros(3*N-4)]))

    root_tol = 1e-8
    residual_tol = 1e-8
    euclidian_distance_tol = 1e-1
    eigen_value_tol = 1e-12

    params = (n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)
    for guess in guesses:
        assert len(guess) == 3*N - 3, f"Guess {guess} is incorrect length"

        try:
            reduced_sol = root(
                deflated_function,
                x0=guess,
                args=(sol_list, params),
                jac=jac_for_root,
                method='hybr',
                tol=root_tol,
                options={'xtol': root_tol, 'maxfev': 5000})
                

        except ValueError as e:
            continue

        reduced_sol = reduced_sol.x

        residuals = MATTIA_FULL_ROOT(reduced_sol, *params)
        if np.max(np.abs(residuals)) > residual_tol:
            continue

        zero_tol = 1e-10
        if np.any(reduced_sol < -zero_tol):
            continue

        is_duplicate = False

        for r in red_sol_list_stable + red_sol_list_unstable:
            if np.linalg.norm(reduced_sol - r) <= euclidian_distance_tol:
                is_duplicate = True
                break
        if is_duplicate:
            continue

        jacobian_mattia_full = MATTIA_FULL_ROOT_JACOBIAN(reduced_sol, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)
        
        if not np.isfinite(jacobian_mattia_full).all():
            print("infinite detected")
            continue

        red_eigenvalues = LA.eigvals(jacobian_mattia_full)
        red_eigenvalues_real = np.real(red_eigenvalues)

        if np.max(red_eigenvalues_real) > eigen_value_tol:
            # print(f"UNSTABLE EIGENVALUES: {red_eigenvalues_real}")
            red_sol_list_unstable.append(reduced_sol)
            red_eigenvalues_real_list.append(red_eigenvalues_real)
        elif np.all(red_eigenvalues_real < -eigen_value_tol):
            # print(f"STABLE EIGENVALUES: {red_eigenvalues_real}")
            red_sol_list_stable.append(reduced_sol)
            red_eigenvalues_real_list.append(red_eigenvalues_real)

    # print(f"\n# of stable points: {len(red_sol_list_stable)}, # of unstable points: {len(red_sol_list_unstable)}")
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
                   p_negative_parameter_array, 
                   rate_min, rate_max):
    
    N = 2**n
    alpha_matrix = alpha_matrix_parameter_array[i]
    beta_matrix = beta_matrix_parameter_array[i]
    k_positive_rates = k_positive_parameter_array[i]
    k_negative_rates = k_negative_parameter_array[i]
    p_positive_rates = p_positive_parameter_array[i]
    p_negative_rates = p_negative_parameter_array[i]

    k_positive_rates = np.ones(N-1)
    k_negative_rates = np.ones(N-1)
    p_positive_rates = np.ones(N-1)
    p_negative_rates = np.ones(N-1)
    # k_positive_rates = reciprocal(a=rate_min, b=rate_max).rvs(size=N-1)
    # k_negative_rates = reciprocal(a=rate_min, b=rate_max).rvs(size=N-1)
    # p_positive_rates = reciprocal(a=rate_min, b=rate_max).rvs(size=N-1)
    # p_negative_rates = reciprocal(a=rate_min, b=rate_max).rvs(size=N-1)
    alpha_matrix = matrix_sample_reciprocal(alpha_matrix, rate_min, rate_max)
    beta_matrix = matrix_sample_reciprocal(beta_matrix, rate_min, rate_max)

    multistable_results = None
    
    Kp, Pp, G, H, Q, M_mat, D, N_mat = MATRIX_FINDER(n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates)

    unique_stable_fp_array, unique_unstable_fp_array, eigenvalues_array = root_finder(n, a_tot_value, x_tot_value, y_tot_value,
                                                     Kp, Pp, G, H, Q, M_mat, D, N_mat)
    
    possible_steady_states = np.floor((n + 2) / 2).astype(int)

    #    (len(unique_unstable_fp_array) > 1):
    # if (len(unique_stable_fp_array) >= possible_steady_states - 1) or \
    # (len(unique_unstable_fp_array) > 1):
    # print(len(unique_stable_fp_array), possible_steady_states + 1)
    # if (len(unique_stable_fp_array) != 1) and (1 < len(unique_stable_fp_array) < possible_steady_states + 1):

    results_dict = {
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
    "eigenvalues": eigenvalues_array,
    # "mean": mean,
    # "variance": variance
    }

    return results_dict

def simulation_root(n, simulation_size, rate_min, rate_max, master_pickle_file):
    a_tot = 1000
    x_tot = 1
    y_tot = 1
    old_best_gamma_parameters = (0.123, 4.46e6)
    new_gamma_parameters = (1, 10)
    gen_rates = all_parameter_generation(n, "distributive", "gamma", new_gamma_parameters, verbose = False)
    alpha_matrix_parameter_array = np.array([gen_rates.alpha_parameter_generation() for _ in range(simulation_size)]) 
    beta_matrix_parameter_array = np.array([gen_rates.beta_parameter_generation() for _ in range(simulation_size)]) 
    k_positive_parameter_array = np.array([gen_rates.k_parameter_generation()[0] for _ in range(simulation_size)])
    k_negative_parameter_array = np.array([gen_rates.k_parameter_generation()[1] for _ in range(simulation_size)])
    p_positive_parameter_array = np.array([gen_rates.p_parameter_generation()[0] for _ in range(simulation_size)])
    p_negative_parameter_array = np.array([gen_rates.p_parameter_generation()[1] for _ in range(simulation_size)])

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
            p_negative_parameter_array, 
            rate_min, rate_max
        ) for i in range(simulation_size)
    )
    
    results_list = [res for res in results if res is not None]

    # update master pickle (append new results)
    try:
        with open(master_pickle_file, "rb") as f:
            master_results = pickle.load(f)
    except FileNotFoundError:
        master_results = []
    
    master_results.extend(results_list)
    
    with open(master_pickle_file, "wb") as f:
        pickle.dump(master_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    # compute statistics
    num_evaluated = len(results_list)
    # systems with at least one stable state
    num_with_any_stable = sum(1 for r in results_list if r.get("num_of_stable_states", 0) > 0)
    # systems with >1 stable state (true multistability)
    num_truly_multistable = sum(1 for r in results_list if r.get("num_of_stable_states", 0) > 1)

    # average number of stable states among truly multistable systems (or 0)
    stable_counts_for_multistable = [r["num_of_stable_states"] for r in results_list if r.get("num_of_stable_states", 0) > 1]
    avg_stable = float(np.mean(stable_counts_for_multistable)) if len(stable_counts_for_multistable) > 0 else 0.0

    results_dict = {
        'num_systems_evaluated': num_evaluated,
        'num_with_any_stable': num_with_any_stable,
        'num_truly_multistable': num_truly_multistable,
        'num_multistable_systems': num_truly_multistable,  # alias kept for compatibility
        'master_pickle_file': master_pickle_file,
        'total_results_in_master': len(master_results),
        'n': n,
        'simulation_size': simulation_size,
        'a': rate_min,
        'b': rate_max,
        'average_num_stable_states': avg_stable
    }

    # print JSON to stdout for parameter_space_script to parse
    print(json.dumps(results_dict))

    return results_dict

def main(a, b, n, sims, master_pickle_file):
    simulation_root(n, sims, a, b, master_pickle_file)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run root finding simulation with specified parameters')
    parser.add_argument('--a', type=float, required=True, help='a')
    parser.add_argument('--b', type=float, required=True, help='b')
    parser.add_argument('--n', type=int, required=True, help='n')
    parser.add_argument('--sims', type=int, required=True, help='sims')
    parser.add_argument('--master-pickle', type=str, required=True, help='Master pickle file path')
    args = parser.parse_args()
    
    main(args.a, args.b, args.n, args.sims, args.master_pickle)