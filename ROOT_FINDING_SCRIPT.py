import numpy as np
from typing import List, Tuple, Dict, Any, Set
from scipy.stats import levy
import matplotlib.pyplot as plt
from scipy.optimize import root
from joblib import Parallel, delayed
import pickle
from numpy import linalg as LA
import numpy as np
from scipy.optimize import root
# from PHOS_FUNCTIONS import MATTIA_FULL
from PHOS_FUNCTIONS import MATTIA_REDUCED, MATTIA_REDUCED_JACOBIAN, MATRIX_FINDER 
from PHOS_FUNCTIONS import ALL_GUESSES, duplicate, stability_calculator, matrix_clip, all_parameter_generation

def MATTIA_REDUCED_ROOT(state_array, n, x_tot, y_tot, L1, L2, W1, W2):
    N = n + 1
    # N = state_array.size

    ones_vec = np.ones(W1.shape[0])

    a_red = np.asarray(state_array, dtype=float).ravel()
    # assert a_red.size == N - 1, f"expected b of length {N-1}, got {a_red.size}"

    # Create full state vector (1D array, not column vector)
    a = np.concatenate([a_red, [1 - np.sum(a_red)]])
    
    # Compute denominator scalars first
    denom1 = 1 + ones_vec @ (W1 @ a)
    denom2 = 1 + ones_vec @ (W2 @ a)
    
    # Compute a_dot as 1D array
    a_dot = ((x_tot * (L1 @ a)) / denom1) + ((y_tot * (L2 @ a)) / denom2)

    return a_dot[:N-1]

def jacobian_reduced_root(state_array, n, x_tot, y_tot, L1, L2, W1, W2):
    N = n + 1  # Full state size (not the reduced state size!)
    
    a_red = np.asarray(state_array, dtype=float).ravel()
    assert a_red.size == N - 1, f"expected state of length {N-1}, got {a_red.size}"

    # Create full state vector as 1D array
    a_fixed_points = np.concatenate([a_red, [1 - np.sum(a_red)]])
    
    ones_vec_j = np.ones(W1.shape[0])  # 1D array of length N-1

    onesW1 = ones_vec_j @ W1  # shape (N,)
    onesW2 = ones_vec_j @ W2  # shape (N,)

    L1a = L1 @ a_fixed_points  # shape (N,)
    L2a = L2 @ a_fixed_points  # shape (N,)
    
    # Compute denominators as scalars
    denom1 = 1 + ones_vec_j @ (W1 @ a_fixed_points)
    denom2 = 1 + ones_vec_j @ (W2 @ a_fixed_points)

    # Compute terms - outer product for the second part
    term1 = (L1 / denom1) - (np.outer(L1a, onesW1) / (denom1**2))   # (N,N)
    term2 = (L2 / denom2) - (np.outer(L2a, onesW2) / (denom2**2))   # (N,N)

    J = (x_tot * term1) + (y_tot * term2)

    return J[:N-1, :N-1]

def root_finder(n, a_tot_value, x_tot_value, y_tot_value,
                alpha_matrix, beta_matrix,
                k_positive_rates, k_negative_rates,
                p_positive_rates, p_negative_rates):

    final_sol_list_stable = []
    final_sol_list_unstable = []
    N = n + 1

    guesses = []
    eigenvalues_real_list = []

    # build guesses for reduced problem (length N-1)
    guesses = ALL_GUESSES(N-1, guesses)  # UNCOMMENT THIS!

    attempt_total_num = 10
    for i in range(attempt_total_num):
        guess = np.random.rand(N-1)
        guesses.append(guess / np.sum(guess))

    Kp, Pp, G, H, Q, M_mat, D, N_mat, L1, L2, W1, W2 = MATRIX_FINDER(
        n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates
    )

    root_tol = 1e-12
    residual_tol = 1e-8
    euclidian_distance_tol = 1e-6
    eigen_value_tol = 1e-8
    zero_tol = 1e-9  # Changed from 1e-16
    
    for guess in guesses:
        try:
            sol = root(
                MATTIA_REDUCED_ROOT,
                x0=guess,
                args=(n, x_tot_value, y_tot_value, L1, L2, W1, W2),
                jac=jacobian_reduced_root,
                method='hybr',
                tol=root_tol,
                options={'maxfev': 5000}
            )
        except Exception as e:
            # print("root() exception:", e)
            continue

        full_sol = np.concatenate([sol.x, [1 - np.sum(sol.x)]])
        
        residual = MATTIA_REDUCED_ROOT(sol.x, n, x_tot_value, y_tot_value, L1, L2, W1, W2)
        if np.max(np.abs(residual)) > residual_tol:  # Use abs for residual check
            # print(f"Max residual {np.max(np.abs(residual))} > {residual_tol}: REJECTED")
            continue

        clipping_tolerance = 1e-8
        if np.any((full_sol < -clipping_tolerance) | (full_sol > 1 + clipping_tolerance)):
            # print("clipping error, full_sol out of bounds:", full_sol)
            continue

        full_sol = np.clip(full_sol, 0.0, 1.0)

        # compute Jacobian / stability
        J = stability_calculator(full_sol, x_tot_value, y_tot_value, L1, L2, W1, W2)
        if not np.isfinite(J).all():
            # print("stability_calculator returned non-finite J; skipping")
            continue

        eigenvalues = LA.eigvals(J)
        eigenvalues_real = np.real(eigenvalues)

        # Filter out eigenvalues from conservation constraint
        eigenvalues_nonzero = eigenvalues_real[np.abs(eigenvalues_real) > zero_tol]

        # duplicate detection
        if duplicate(full_sol, final_sol_list_stable, euclidian_distance_tol) or \
           duplicate(full_sol, final_sol_list_unstable, euclidian_distance_tol):
            continue

        # classify by NON-ZERO eigenvalues only
        if len(eigenvalues_nonzero) == 0:
            print(f"All eigenvalues near zero - degenerate case")
            continue
        elif np.max(eigenvalues_nonzero) > eigen_value_tol:
            print(f"Unstable: max eigenvalue {np.max(eigenvalues_nonzero):.6e} > {eigen_value_tol}")
            print(f"  Fixed point: {full_sol}")
            print(f"  All eigenvalues: {eigenvalues_real}")
            final_sol_list_unstable.append(full_sol)
            eigenvalues_real_list.append(eigenvalues_real)
        elif np.all(eigenvalues_nonzero < -eigen_value_tol):
            print(f"Stable: all non-zero eigenvalues negative")
            print(f"  Fixed point: {full_sol}")
            final_sol_list_stable.append(full_sol)
            eigenvalues_real_list.append(eigenvalues_real)
        else:
            print(f"Marginally stable eigenvalues: {eigenvalues_nonzero}")

    print(f"# of stable points found is {len(final_sol_list_stable)}, and # of unstable states found is {len(final_sol_list_unstable)}")

    if (len(final_sol_list_stable) == 0) and (len(final_sol_list_unstable) == 0):
        return np.array([]), np.array([]), np.array([])
    
    return np.array(final_sol_list_stable), np.array(final_sol_list_unstable), np.array(eigenvalues_real_list)

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
    alpha_matrix = alpha_matrix_parameter_array[i] / np.mean(alpha_matrix_parameter_array[i])
    beta_matrix = beta_matrix_parameter_array[i] / np.mean(beta_matrix_parameter_array[i])

    # clipping
    alpha_matrix = matrix_clip(alpha_matrix); beta_matrix = matrix_clip(beta_matrix)

    multistable_results = None
    
    unique_stable_fp_array, unique_unstable_fp_array, eigenvalues_array = root_finder(n, a_tot_value, x_tot_value, y_tot_value,
                                                     alpha_matrix, beta_matrix, k_positive_rates,
                                                     k_negative_rates, p_positive_rates, p_negative_rates)
    
    possible_steady_states = np.floor((n + 2) / 2).astype(int)
    # if len(unique_stable_fp_array) == 1:
    if len(unique_stable_fp_array) == 2 and len(unique_unstable_fp_array) == 1:
        multistable_results = {
        "num_of_stable_states": len(unique_stable_fp_array),
        "num_of_unstable_states": len(unique_unstable_fp_array),
        "stable_states": np.array([unique_stable_fp_array[k] for k in range(len(unique_stable_fp_array))]),
        "unstable_states": np.array([unique_unstable_fp_array[k] for k in range(len(unique_unstable_fp_array))]),
        "total_concentration_values": np.array([a_tot_value, x_tot_value, y_tot_value]),
        "alpha_matrix": alpha_matrix_parameter_array[i],
        "beta_matrix": beta_matrix_parameter_array[i],
        "k_positive_rates": k_positive_parameter_array[i],
        "k_negative_rates": k_negative_parameter_array[i],
        "p_positive_rates": p_positive_parameter_array[i],
        "p_negative_rates": p_negative_parameter_array[i],
        "eigenvalues": eigenvalues_array
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
    simulation_root(n, 200)
    
if __name__ == "__main__":
    main()