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
from parameter_search_script import all_parameter_generation, stability_calculator
from phos_simulation_script import MATRIX_FINDER

def MATTIA_REDUCED_ROOT(state_array, n, x_tot, y_tot, L1, L2, W1, W2):

    N = n + 1
    # N = n**2
    ones_vec = np.ones(N - 1)
    a_red = np.asarray(state_array, dtype = float)
    assert a_red.size == N - 1, f"expected b of length {N-1}, got {a_red.size}"

    # assert len(state_array) == N

    a = np.concatenate([a_red, [1 - np.sum(a_red)]])

    a_dot = ((x_tot * L1 @ a) / (1 + ones_vec @ W1 @ a)) + ((y_tot * L2 @ a) / (1 + ones_vec @ W2 @ a)) 
    # a_dot = ((x_tot * L1 @ a) / (1 + ones_vec @ W1 @ a)) + ((y_tot * L2 @ a) / (1 + ones_vec @ W2 @ a)) 

    return a_dot[: N-1]

def jacobian_reduced_root(state_array, n, x_tot, y_tot, L1, L2, W1, W2):

    N = n + 1
    # N = n**2
    a_red = np.asarray(state_array, dtype=float)
    assert a_red.size == N - 1, f"expected b of length {N-1}, got {a_red.size}"

    a_fixed_points = np.concatenate([a_red, [1-np.sum(a_red)]])
    # a_fixed_red = np.asarray(state_array, dtype = float)

    ones_vec_j = np.ones((1, N-1), dtype = float) # shape (1, N-1)

    a_fixed_points = a_fixed_points.reshape((N, 1))  # shape (N, 1)
    L1 = np.array(L1, dtype=float)
    L2 = np.array(L2, dtype=float)
    W1 = np.array(W1, dtype=float)
    W2 = np.array(W2, dtype=float)

    # Compute denominators
    denom1 = float(1 + np.dot(ones_vec_j, (W1 @ a_fixed_points)).item())
    denom2 = float(1 + np.dot(ones_vec_j, (W2 @ a_fixed_points)).item())

    term1 = (L1 / denom1) - (((L1 @ a_fixed_points) @ (ones_vec_j @ W1)) / (denom1**2))
    term2 = (L2 / denom2) - (((L2 @ a_fixed_points) @ (ones_vec_j @ W2)) / (denom2**2))

    p = x_tot / y_tot

    J = (p * term1) + term2
    # return J
    return J[:N-1, :N-1]

def duplicate(candidate, collection, norm_tol):
    for v in collection:
        if np.linalg.norm(np.array(v) - np.array(candidate)) < norm_tol:
            if np.allclose(v, candidate):
                return True
    return False

def root_finder(sites_n, a_tot_value, x_tot_value, y_tot_value,
                      alpha_matrix, beta_matrix,
                      k_positive_rates, k_negative_rates,
                      p_positive_rates, p_negative_rates):

    final_sol_list_stable = []
    final_sol_list_unstable = []
    N = sites_n + 1
    # N = sites_n**2

    attempt_total_num = 10 # should be proportional to the number of sites/dimensions?
    guesses = []

    rng = np.random.default_rng()
    
    # 1. Very sharp corners: one component dominates (alpha << 1)
    for _ in range(N):
        alpha_sharp = np.full(N, 0.1)  # Small alpha pushes to corners
        guess = rng.dirichlet(alpha_sharp)
        guesses.append(guess[:-1])  # Remove last component for reduced system
    
    # 2. Edges: two components share mass (set others to very small alpha)
    edge_samples_per_pair = 2
    for i in range(N):
        for j in range(i+1, N):
            for _ in range(edge_samples_per_pair):
                alpha_edge = np.full(N, 0.05)  # Very small for components not on edge
                alpha_edge[i] = 1.0  # Moderate for edge components
                alpha_edge[j] = 1.0
                guess = rng.dirichlet(alpha_edge)
                guesses.append(guess[:-1])
    
    # 3. Faces: k components share mass equally
    for n_active in range(2, N+1):
        for _ in range(2):  # 2 samples per face
            alpha_face = np.full(N, 0.05)
            alpha_face[:n_active] = 1.0
            guess = rng.dirichlet(alpha_face)
            guesses.append(guess[:-1])
    
    # 4. Uniform-ish distribution (alpha = 1 is uniform)
    for _ in range(3):
        guess = rng.dirichlet(np.ones(N))
        guesses.append(guess[:-1])
    
    # 5. Slightly perturbed corners (exact corners)
    for i in range(N):
        corner = np.zeros(N)
        corner[i] = 1.0
        guesses.append(corner[:-1])

    Kp, Pp, G, H, Q, M_mat, D, N_mat, L1, L2, W1, W2 = MATRIX_FINDER(sites_n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates)

    for guess in guesses:

        root_finder_tol = 1e-12
        # guess = 0.1*np.ones(N-1)
        try: 
            sol = root(MATTIA_REDUCED_ROOT, x0=guess, jac=jacobian_reduced_root, tol = root_finder_tol,
                        method='hybr', args=(sites_n, x_tot_value, y_tot_value, L1, L2, W1, W2))
            # sol = root(MATTIA_REDUCED_ROOT, x0=guess, tol = root_finder_tol,
            #             method='hybr', args=(sites_n, x_tot_value, y_tot_value, L1, L2, W1, W2))
            if not sol.success:
                print("root finding not a success")
                print(sol.message)
                continue

        except Exception as e:
            continue   

        full_sol = np.concatenate([sol.x, [1 - np.sum(sol.x)]])

        clipping_tolerance = 1e-8
        if np.any((full_sol < -clipping_tolerance) | (full_sol > 1+clipping_tolerance)):
            print("clipping error")
            continue

        full_sol = np.clip(full_sol, 0-clipping_tolerance, 1+clipping_tolerance)
        # print("success:", res.success)
        # print("b solution:", res.x)
        # # reconstruct full a from b solution:
        # a_sol = np.concatenate([res.x, [1.0 - np.sum(res.x)]])
        # print("a (full) solution:", a_sol)
        # print("residual (last N-1 eqns):", res.fun)


        J = stability_calculator(full_sol, x_tot_value / y_tot_value, L1, L2, W1, W2)
        # if not np.all(np.isfinite(J)) or np.isnan(J).any():
        if not np.isfinite(J).all():
            print("infinite detected")
            continue  # skip this guess if J contains NaN or inf

        eigenvalues = LA.eigvals(J)
        eigenvalues_real = np.real(eigenvalues)  

        # norm_tol = root_finder_tol 
        norm_tol = 1e-4
        eigen_value_tol = 1e-16
        if np.max(eigenvalues_real) > eigen_value_tol:
            if duplicate(full_sol, final_sol_list_unstable, norm_tol) == False:

        # if duplicate(full_sol, final_sol_list_unstable, norm_tol) == False:
                final_sol_list_unstable.append(full_sol)
        
        if np.all(eigenvalues_real < -eigen_value_tol):
            if duplicate(full_sol, final_sol_list_stable, norm_tol) == False:
        # if duplicate(full_sol, final_sol_list_stable, norm_tol) == False:
                final_sol_list_stable.append(full_sol) 
    

    print(f"# of stable points found is {len(final_sol_list_stable)}, and # of unsteady states found is {len(final_sol_list_unstable)}")
    # final_sol_list_stable = remove_close_subarrays(final_sol_list_stable)
    # final_sol_list_unstable = remove_close_subarrays(final_sol_list_unstable)

    if (len(final_sol_list_stable) == 0) and (len(final_sol_list_unstable) == 0):
        # print("Found no solutions.")
        return np.array([]), np.array([])  # failed
    return np.array(final_sol_list_stable), np.array(final_sol_list_unstable) # an array of arrays (a matrix)

def process_sample_script(i,
                   sites_n,
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

    # normalize
    alpha_matrix = alpha_matrix_parameter_array[i] / np.mean(alpha_matrix_parameter_array[i])
    beta_matrix = beta_matrix_parameter_array[i] / np.mean(beta_matrix_parameter_array[i])

    def matrix_clip(matrix):
        clipped = matrix.copy()
        mask = clipped != 0
        clipped[mask] = np.clip(clipped[mask], rate_min, rate_max)
        return clipped
    
    # clipping
    alpha_matrix = matrix_clip(alpha_matrix); beta_matrix = matrix_clip(beta_matrix)

    multistable_results = None
    
    unique_stable_fp_array, unique_unstable_fp_array = root_finder(sites_n, a_tot_value, x_tot_value, y_tot_value,
                                                     alpha_matrix, beta_matrix, k_positive_rates,
                                                     k_negative_rates, p_positive_rates, p_negative_rates)
    
    possible_steady_states = np.floor((sites_n + 2) / 2).astype(int)

    if len(unique_stable_fp_array) == 1 and len(unique_unstable_fp_array) == 1:
    # if len(unique_stable_fp_array) == possible_steady_states:
        multistable_results = {
        "num_of_stable_states": len(unique_stable_fp_array),
        "num_of_unstable_states": len(unique_unstable_fp_array),
        "stable_states": np.array([unique_stable_fp_array[k] for k in range(len(unique_stable_fp_array))]),
        "unstable_states": np.array([unique_unstable_fp_array[k] for k in range(len(unique_unstable_fp_array))]),
        "total_concentration_values": np.array([a_tot_value, x_tot_value, y_tot_value]),
        "alpha_matrix": alpha_matrix_parameter_array[i],
        "beta_matrix": beta_matrix_parameter_array[i],
        "k_positive": k_positive_parameter_array[i],
        "k_negative": k_negative_parameter_array[i],
        "p_positive": p_positive_parameter_array[i],
        "p_negative": p_negative_parameter_array[i]
        }

    return multistable_results

def simulation_root(sites_n, simulation_size):
    a_tot_value = 1
    N = sites_n + 1
    # N = sites_n**2

    gen_rates = all_parameter_generation(sites_n, "distributive", "gamma", (0.123, 4.46e6), verbose = False)
    rate_min, rate_max = 1e-1, 1e7
    alpha_matrix_parameter_array = np.array([gen_rates.alpha_parameter_generation() for _ in range(simulation_size)]) # CANNOT CLIP
    beta_matrix_parameter_array = np.array([gen_rates.beta_parameter_generation() for _ in range(simulation_size)]) # CANNOT CLIP
    k_positive_parameter_array = np.array([np.ones(N - 1) for _ in range(simulation_size)])
    k_negative_parameter_array = np.array([np.ones(N - 1) for _ in range(simulation_size)])
    p_positive_parameter_array = np.array([np.ones(N - 1) for _ in range(simulation_size)])
    p_negative_parameter_array = np.array([np.ones(N - 1) for _ in range(simulation_size)])
    # k_positive_parameter_array = np.array([np.clip(gen_rates.k_parameter_generation()[0], rate_min, rate_max) for _ in range(simulation_size)])
    # k_negative_parameter_array = np.array([np.clip(gen_rates.k_parameter_generation()[1], rate_min, rate_max) for _ in range(simulation_size)])
    # p_positive_parameter_array = np.array([np.clip(gen_rates.p_parameter_generation()[0], rate_min, rate_max) for _ in range(simulation_size)])
    # p_negative_parameter_array = np.array([np.clip(gen_rates.p_parameter_generation()[1], rate_min, rate_max) for _ in range(simulation_size)])
    gen_concentrations = all_parameter_generation(sites_n, "distributive", "gamma", (0.40637, 0.035587), verbose = False)
    concentration_min, concentration_max = 1e-4, 1e-1
    x_tot_value_parameter_array = np.array([np.clip(gen_concentrations.total_concentration_generation()[0], concentration_min, concentration_max) for _ in range(simulation_size)])
    y_tot_value_parameter_array = np.array([np.clip(gen_concentrations.total_concentration_generation()[1], concentration_min, concentration_max) for _ in range(simulation_size)])

    results = Parallel(n_jobs=-2, backend="loky")(
        delayed(process_sample_script)(
            i,
            sites_n,
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
    multifile = f"multistability_parameters_{simulation_size}_{sites_n}.pkl"

    with open(multifile, "wb") as f:
        pickle.dump(multistable_results_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    return

def main():
    n = 2
    simulation_root(n, 5000)
    
if __name__ == "__main__":
    main()