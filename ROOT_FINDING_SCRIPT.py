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
from PHOS_FUNCTIONS import MATTIA_REDUCED, MATTIA_REDUCED_JACOBIAN, MATRIX_FINDER, guess_generator
from PHOS_FUNCTIONS import ALL_GUESSES, duplicate, stability_calculator, matrix_clip, all_parameter_generation

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

    N = n + 1
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

    x_dot = -1*np.sum(b_dot)
    y_dot = -1*np.sum(c_dot)

    a_dot_red = a_dot[0: N-1]

    return np.concatenate([a_dot_red, b_dot, c_dot])

def root_finder(n, a_tot, x_tot, y_tot,
                alpha_matrix, beta_matrix,
                k_positive_rates, k_negative_rates,
                p_positive_rates, p_negative_rates):

    red_sol_list_stable = []
    red_sol_list_unstable = []
    N = n + 1

    guesses = []
    red_eigenvalues_real_list = []

    attempt_total_num = 8
    guesses = []

    # guesses.append(np.ones(3*N-3)) 
    # guesses.append(np.zeros(3*N-3))    

    # for i in range(attempt_total_num):
    #     guesses.append(np.concatenate([np.array([random.random()]), np.zeros(3*N-4)]))    # attempt_total_num = 3

    #     guesses.append(np.concatenate([np.array([alpha]), np.zeros(3*N-4)]))    # attempt_total_num = 3

    for i in range(attempt_total_num):
        guesses.append(guess_generator(n, a_tot, x_tot, y_tot))

    Kp, Pp, G, H, Q, M_mat, D, N_mat, L1, L2, W1, W2 = MATRIX_FINDER(
        n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates
    )

    root_tol = 1e-8
    residual_tol = 1e-5
    euclidian_distance_tol = 1e-1
    eigen_value_tol = 1e-5
    
    for guess in guesses:
        # assert np.any(np.array(guess) <= 0), f"Guess {guess} is non-physical"
        assert len(guess) == 3*N - 3, f"Guess {guess} is incorrect length"

        try:
            reduced_sol = root(
                MATTIA_FULL_ROOT,
                x0=guess,
                args=(n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat),
                jac=MATTIA_FULL_ROOT_JACOBIAN,
                method='hybr',
                tol=root_tol,
                options={'maxfev': 5000}
            )
        except Exception as e:
            # print(reduced_sol.message)
            continue

        reduced_sol = reduced_sol.x

        residuals = MATTIA_FULL_ROOT(reduced_sol, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)

        jacobian_mattia_full = MATTIA_FULL_ROOT_JACOBIAN
        if np.max(np.abs(residuals)) > residual_tol:
        
            print(f"Residuals {residuals} were not found to obey the tolerance {residual_tol}.")
            continue

        # if np.any(reduced_sol > 0):
        #     # print(reduced_sol)
        #     print("Non-physical fixed point.")
        #     continue

        def mattia_full_wrapper(state):
            return MATTIA_FULL_ROOT(state, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)

        jacobian_mattia_full = np.array([
            approx_fprime(reduced_sol, lambda s: mattia_full_wrapper(s)[i], epsilon=1e-8)
            for i in range(len(reduced_sol))])
        
        if not np.isfinite(jacobian_mattia_full).all():
            print("infinite detected")
            continue

        # duplicate detection
        if duplicate(reduced_sol, red_sol_list_stable, euclidian_distance_tol) or duplicate(reduced_sol, red_sol_list_unstable, euclidian_distance_tol):
            print("duplicate detected")
            continue

        red_eigenvalues = LA.eigvals(jacobian_mattia_full)
        red_eigenvalues_real = np.real(red_eigenvalues)

        # print(f"Eigenvalues at fixed point: {red_eigenvalues_real}")
        if np.max(red_eigenvalues_real) > eigen_value_tol:
            print(f"UNSTABLE EIGENVALUES: {red_eigenvalues_real}")
            red_sol_list_unstable.append(reduced_sol)
            red_eigenvalues_real_list.append(red_eigenvalues_real)
        elif np.all(red_eigenvalues_real < -eigen_value_tol):
            print(f"STABLE EIGENVALUES: {red_eigenvalues_real}")
            red_sol_list_stable.append(reduced_sol)
            red_eigenvalues_real_list.append(red_eigenvalues_real)
        else:
            print(f" -> Marginally stable")

    print(f"\n# of stable points: {len(red_sol_list_stable)}, # of unstable points: {len(red_sol_list_unstable)}")

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
    N = n+1
    alpha_matrix = alpha_matrix_parameter_array[i]
    beta_matrix = beta_matrix_parameter_array[i]
    k_positive_rates = k_positive_parameter_array[i]
    k_negative_rates = k_negative_parameter_array[i]
    p_positive_rates = p_positive_parameter_array[i]
    p_negative_rates = p_negative_parameter_array[i]
    rate_min, rate_max = 1e-2, 1e2

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
    
    # k_positive_rates = k_positive_rates / np.mean(k_positive_rates)
    # k_positive_rates = np.clip(k_positive_rates, rate_min, rate_max)
    # k_negative_rates = k_negative_rates / np.mean(k_negative_rates)
    # k_negative_rates = np.clip(k_negative_rates, rate_min, rate_max)

    # p_positive_rates = p_positive_rates / np.mean(p_positive_rates)
    # p_positive_rates = np.clip(p_positive_rates, rate_min, rate_max)
    # p_negative_rates = p_negative_rates / np.mean(p_negative_rates)
    # p_negative_rates = np.clip(p_negative_rates, rate_min, rate_max)

    # alpha_matrix = alpha_matrix_parameter_array[i] / np.mean(alpha_matrix_parameter_array[i])
    # alpha_matrix = matrix_clip(alpha_matrix, rate_min, rate_max)
    # beta_matrix = beta_matrix_parameter_array[i] / np.mean(beta_matrix_parameter_array[i])
    # beta_matrix = matrix_clip(beta_matrix, rate_min, rate_max)

    k_positive_rates = reciprocal(a=rate_min, b=rate_max).rvs(size=N-1)
    k_negative_rates = reciprocal(a=rate_min, b=rate_max).rvs(size=N-1)    
    p_positive_rates = reciprocal(a=rate_min, b=rate_max).rvs(size=N-1)
    p_negative_rates = reciprocal(a=rate_min, b=rate_max).rvs(size=N-1)

    alpha_matrix = np.diag(reciprocal(a=rate_min, b=rate_max).rvs(size=N-1), k = 1)
    beta_matrix = np.diag(reciprocal(a=rate_min, b=rate_max).rvs(size=N-1), k = -1)

    multistable_results = None
    
    unique_stable_fp_array, unique_unstable_fp_array, eigenvalues_array = root_finder(n, a_tot_value, x_tot_value, y_tot_value,
                                                     alpha_matrix, beta_matrix, k_positive_rates,
                                                     k_negative_rates, p_positive_rates, p_negative_rates)
    
    possible_steady_states = np.floor((n + 2) / 2).astype(int)

    # if len(unique_unstable_fp_array) == 1:
    if len(unique_stable_fp_array) == 2 and len(unique_unstable_fp_array) == 1:
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
        # "alpha_matrix": alpha_matrix_parameter_array[i],
        # "beta_matrix": beta_matrix_parameter_array[i],
        # "k_positive_rates": k_positive_parameter_array[i],
        # "k_negative_rates": k_negative_parameter_array[i],
        # "p_positive_rates": p_positive_parameter_array[i],
        # "p_negative_rates": p_negative_parameter_array[i],
        "eigenvalues": eigenvalues_array
        }

    return multistable_results

def simulation_root(n, simulation_size):
    a_tot = 1
    N = n + 1
    N = n + 1
    x_tot = 1e-3
    y_tot = 1e-3
    old_best_gamma_parameters = (0.123, 4.46e6)
    new_gamma_parameters = (1e8, 1e8)
    gen_rates = all_parameter_generation(n, "distributive", "gamma", new_gamma_parameters, verbose = False)
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
    # concentration_min, concentration_max = 1e-4, 1e-1
    # x_tot_value_parameter_array = np.array([x_tot*np.ones(1) for _ in range(simulation_size)])
    # y_tot_value_parameter_array = np.array([y_tot*np.ones(1)  for _ in range(simulation_size)])

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
    simulation_root(n, 20000)
    
if __name__ == "__main__":
    main()