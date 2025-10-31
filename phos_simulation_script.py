import numpy as np
import sympy as sp
from typing import List, Tuple, Dict, Any, Set
from scipy.stats import levy
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from parameter_search_script import all_parameter_generation

def MATTIA_FULL(t, state_array, n, Kp, Pp, G, H, Q, M_mat, D, N_mat):

    N = n + 1
    ones_vec = np.ones(N - 1)
    
    # assert len(state_array) == 3*N
    # print(state_array)
    a = state_array[0: N]
    b = state_array[N: 2*N - 1]
    c = state_array[2*N - 1: 3*N - 2]
    x = float(state_array[-2])
    y = float(state_array[-1])

    a_dot = (G @ b) + (H @ c) - x * (Kp @ a) - y * (Pp @ a)  
    b_dot = x * (Q @ a) - (M_mat @ b)
    c_dot = y * (D @ a) - (N_mat @ c)

    x_dot = -1*ones_vec.T @ b_dot
    y_dot = -1*ones_vec.T @ c_dot

    return np.concatenate((a_dot, b_dot, c_dot, np.array([x_dot, y_dot])))

# def jacobian_reduced(t, state_array, n, x_tot, y_tot, L1, L2, W1, W2):
#     N = 2**n
#     ones_vec_j = np.ones(N - 1)
#     # a_fixed_points = np.array(a_fixed_points).reshape((N, 1))  # shape (N, 1)
#     a = state_array[0: N].astype(float)

#     # L1 = np.array(L1, dtype=float)
#     # L2 = np.array(L2, dtype=float)
#     # W1 = np.array(W1, dtype=float)
#     # W2 = np.array(W2, dtype=float)

#     # Compute denominators
#     denom1 = 1 + float(ones_vec_j @ (W1 @ a))
#     denom2 = 1 + float(ones_vec_j @ (W2 @ a))

#     # Compute terms
#     term1 = (L1 / denom1) - np.outer(L1 @ a, ones_vec_j @ W1) / (denom1**2)
#     term2 = (L2 / denom2) - np.outer(L2 @ a, ones_vec_j @ W2) / (denom2**2)
#     return (x_tot / y_tot) * term1 + term2

def MATTIA_REDUCED(t, state_array, n, x_tot, y_tot, L1, L2, W1, W2):

    N = n + 1
    ones_vec = np.ones(N - 1)

    # assert len(state_array) == N

    a = state_array[0: N].astype(float)

    a_dot = ((x_tot * L1 @ a) / (1 + ones_vec @ W1 @ a)) + (y_tot * (L2 @ a) / (1 + ones_vec @ W2 @ a)) 

    return a_dot

def MATRIX_FINDER(n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates):

    N = n + 1
    ones_vec = np.ones(N - 1)

    Kp = np.diag(np.append(k_positive_rates, 0))
    Km = np.append(np.diag(k_negative_rates), np.zeros((1, len(k_negative_rates))), axis=0)

    Pp = np.diag(np.insert(p_positive_rates, 0, 0))
    Pm = np.vstack([np.zeros((1, len(p_negative_rates))), np.diag(p_negative_rates)])    # print("a", a)

    adjusted_alpha_mat = np.delete(alpha_matrix, -1, axis = 0)
    adjusted_beta_mat = np.delete(beta_matrix, 0, axis = 0)

    Da = np.diag(alpha_matrix[:-1, 1:] @ ones_vec)
    Db = np.diag(beta_matrix[1:, :-1] @ ones_vec)

    U = np.diag(k_negative_rates)
    I = np.diag(p_negative_rates)
    Q = Kp[:-1, :]
    D = np.delete(Pp, 0, axis=0)

    M_mat = U + Da
    N_mat = I + Db

    G = Km + adjusted_alpha_mat.T
    H = Pm + adjusted_beta_mat.T

    M_inv = np.linalg.inv(M_mat); N_inv = np.linalg.inv(N_mat)

    L1 = G @ M_inv @ Q - Kp; L2 = H @ N_inv @ D - Pp
    W1 = M_inv @ Q; W2 = N_inv @ D

    return Kp, Pp, G, H, Q, M_mat, D, N_mat, L1, L2, W1, W2

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
    
    abstol = 1e-20
    reltol = 1e-5

    if full_model == False:

        assert len(initial_states_array) == N

        mattia_reduced_parameter_tuple = (n, x_tot, y_tot, L1, L2, W1, W2)
        # t_array = np.linspace(t_span[0], t_span[1], 500)
        sol = solve_ivp(MATTIA_REDUCED, t_span = t_span, y0=np.asarray(initial_states_array, dtype=float), 
                        args = mattia_reduced_parameter_tuple, method = 'LSODA', atol = abstol, rtol = reltol)
        # jac = jacobian_reduced
        a_solution_stack = np.stack([sol.y[i] for i in range(0, N)]) / a_tot

        if plot == True:
            plt.figure(figsize = (8, 6))
            plt.style.use('seaborn-v0_8-whitegrid')
            for i in range(a_solution_stack.shape[0]):
                color = color_for_species(i)
                plt.plot(sol.t, a_solution_stack[i], color=color, label = f"[$A_{i}]$", lw=4, alpha = 0.4)
                print(f"final A_{i} = {a_solution_stack[i][-1]}")
                plt.title(f"Plotting reduced phosphorylation dynamics for n = {n}")

    if full_model == True:
        assert len(initial_states_array) == 3*n + 3
        # assert np.all(initial_states_array >= 1e-15)
        assert np.all(initial_states_array >= 0)

        mattia_parameter_tuple = (n, Kp, Pp, G, H, Q, M_mat, D, N_mat)
        # t_array = np.linspace(t_span[0], t_span[1], 500)
        sol = solve_ivp(MATTIA_FULL, t_span = t_span, y0=np.asarray(initial_states_array, dtype=float), 
                        args = mattia_parameter_tuple, method = 'LSODA', atol = abstol, rtol = reltol)

        a_solution_stack = np.stack([sol.y[i] for i in range(0, N)]) / a_tot
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

def remove_close_subarrays(arrays, threshold,):
    # Convert to numpy array for easier computation
    arrays_np = np.array(arrays)
    
    # Keep track of which arrays to keep
    keep_indices = []
    
    for i in range(len(arrays_np)):
        is_far_enough = True
        
        # Check distance to all previously kept arrays
        for j in keep_indices:
            if np.linalg.norm(arrays_np[i] - arrays_np[j]) < threshold:
                is_far_enough = False
                break
        if is_far_enough:
            keep_indices.append(i)
    
    return np.array([arrays[i] for i in keep_indices])

def main():
    sites_n = 2; N = sites_n + 1
    a_tot_value = 1

    rates_best_shape_parameters_non_sequential = (0.123, 4.46e6)
    gen_rates = all_parameter_generation(sites_n, "distributive", "gamma", rates_best_shape_parameters_non_sequential, verbose = False)
    alpha_matrix = gen_rates.alpha_parameter_generation()
    beta_matrix = gen_rates.beta_parameter_generation()
    k_positive_rates, k_negative_rates = gen_rates.k_parameter_generation()
    p_positive_rates, p_negative_rates = gen_rates.p_parameter_generation()

    rate_min, rate_max = 1e-1, 1e7

    # NORMALIZING
    alpha_matrix = alpha_matrix / np.mean(alpha_matrix)
    beta_matrix = beta_matrix / np.mean(beta_matrix)

    concentrations_best_shape_parameters = (0.40637, 0.035587)
    gen_concentrations = all_parameter_generation(sites_n, "distributive", "gamma", concentrations_best_shape_parameters, verbose = False)
    concentration_min, concentration_max = 1e-4, 1e-1
    x_tot_value = gen_concentrations.total_concentration_generation()[0]
    y_tot_value = gen_concentrations.total_concentration_generation()[1]
    Kp, Pp, G, H, Q, M_mat, D, N_mat, L1, L2, W1, W2 = MATRIX_FINDER(sites_n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates)

    final_t = 1000000
    grid_num = 5
    stable_points = []
    for u in range(0, grid_num):
        rand_guess = np.random.rand(N)
        rand_guess = rand_guess / np.sum(rand_guess)
        # rand_guess[-1] = 1 - np.sum(rand_guess[0:])
    # for guess in guesses:
        initial_states_array = rand_guess
        parameters_tuple = (sites_n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates, a_tot_value, x_tot_value, y_tot_value)            
        stable_points.append(phosphorylation_system_solver(parameters_tuple, initial_states_array, final_t, False, True))
    stable_point_array = np.array(stable_points)
    print(f"All points found:")
    print(stable_point_array)

    close_tol = 1e-6
    unique_stable_point_array = np.array(remove_close_subarrays(stable_points, close_tol))
    print(f"All unique points are:")
    print(unique_stable_point_array)
    for fp in unique_stable_point_array:
        residual = MATTIA_REDUCED(1, fp, sites_n, x_tot_value, y_tot_value, L1, L2, W1, W2)
        if np.max(residual) > 1e-6:
            print("Point was not found to be a valid zero.")


if __name__ == "__main__":
    main()