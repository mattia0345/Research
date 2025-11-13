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

# def MATTIA_FULL_ROOT_JACOBIAN(state_array, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat):
    # """Compute the Jacobian of MATTIA_FULL_ROOT numerically"""
    # def mattia_full_wrapper(state):
    #     return MATTIA_FULL_ROOT(state, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)
    
    # jacobian_mattia_full = np.array([
    #     approx_fprime(state_array, lambda s: mattia_full_wrapper(s)[i], epsilon=1e-8)
    #     for i in range(len(state_array))])
    
    # return jacobian_mattia_full

# def MATTIA_FULL_ROOT_JACOBIAN(state_array, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat):
#     N = 2**n

#     assert len(state_array) == 3*N - 3
#     a_red = state_array[0: N - 1]
#     b = state_array[N - 1: 2*N - 2]
#     c = state_array[2*N - 2: 3*N - 3]

#     a = np.concatenate([a_red, [a_tot - np.sum(a_red) - np.sum(b) - np.sum(c)]])
#     x = x_tot - np.sum(b)
#     y = y_tot - np.sum(c)

#     a_dot_partial_a = -x * Kp - y * Pp 
#     a_dot_partial_b = G
#     a_dot_partial_c = H
#     a_dot_partial_x = (- Kp @ a).reshape(-1, 1)
#     a_dot_partial_y = (- Pp @ a).reshape(-1, 1)
#     adot_gradient = np.hstack([a_dot_partial_a, a_dot_partial_b, a_dot_partial_c])
#     adot_reduced_gradient = adot_gradient[:-1, :]
#     # print(adot_reduced_gradient)
#     b_dot_partial_a = x * Q
#     b_dot_partial_b = - M_mat
#     b_dot_partial_c = np.zeros((N-1, N-1))
#     b_dot_partial_x = (Q @ a).reshape(-1, 1)
#     b_dot_partial_y = np.zeros((N-1, 1))
#     bdot_gradient = np.hstack([b_dot_partial_a, b_dot_partial_b, b_dot_partial_c])

#     c_dot_partial_a = y * D
#     c_dot_partial_b = np.zeros((N-1, N-1))
#     c_dot_partial_c = - N_mat
#     c_dot_partial_x = np.zeros((N-1, 1))
#     c_dot_partial_y = (D @ a).reshape(-1, 1)
#     cdot_gradient = np.hstack([c_dot_partial_a, c_dot_partial_b, c_dot_partial_c])

#     red_J = np.vstack([adot_reduced_gradient, bdot_gradient, cdot_gradient])
#     print(red_J.shape)
#     return red_J

def MATTIA_FULL_ROOT_JACOBIAN(state_array, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat):
    N = 2**n
    assert len(state_array) == 3*N - 3
    
    a_red = state_array[0: N - 1]
    b = state_array[N - 1: 2*N - 2]
    c = state_array[2*N - 2: 3*N - 3]

    a = np.concatenate([a_red, [a_tot - np.sum(a_red) - np.sum(b) - np.sum(c)]])
    x = x_tot - np.sum(b)
    y = y_tot - np.sum(c)

    # We need to compute the Jacobian of:
    # a_dot_red = (G @ b + H @ c - x * Kp @ a - y * Pp @ a)[0:N-1]
    # b_dot = x * Q @ a - M_mat @ b
    # c_dot = y * D @ a - N_mat @ c
    
    # First, let's establish how a, x, y depend on the state variables:
    # a[i] = a_red[i] for i < N-1
    # a[N-1] = a_tot - sum(a_red) - sum(b) - sum(c)
    # x = x_tot - sum(b)
    # y = y_tot - sum(c)
    
    # Create derivative matrices for a, x, y w.r.t. state_array
    # da/d(state_array): shape (N, 3N-3)
    da_dstate = np.zeros((N, 3*N - 3))
    # First N-1 rows: da[i]/da_red[j] = delta[i,j]
    da_dstate[0:N-1, 0:N-1] = np.eye(N-1)
    # Last row: da[N-1]/da_red[j] = -1, da[N-1]/db[j] = -1, da[N-1]/dc[j] = -1
    da_dstate[N-1, :] = -1
    
    # dx/d(state_array): shape (1, 3N-3)
    dx_dstate = np.zeros((1, 3*N - 3))
    dx_dstate[0, N-1:2*N-2] = -1  # dx/db[j] = -1
    
    # dy/d(state_array): shape (1, 3N-3)
    dy_dstate = np.zeros((1, 3*N - 3))
    dy_dstate[0, 2*N-2:3*N-3] = -1  # dy/dc[j] = -1
    
    # Now compute the Jacobian for each output block
    
    # === Block 1: d(a_dot_red)/d(state_array) ===
    # a_dot = G @ b + H @ c - x * Kp @ a - y * Pp @ a
    # We need first N-1 rows
    
    # Term 1: d(G @ b)/d(state_array)
    # G @ b depends only on b
    term1 = np.zeros((N, 3*N - 3))
    term1[:, N-1:2*N-2] = G  # d(G @ b)/db = G
    
    # Term 2: d(H @ c)/d(state_array)
    term2 = np.zeros((N, 3*N - 3))
    term2[:, 2*N-2:3*N-3] = H  # d(H @ c)/dc = H
    
    # Term 3: d(-x * Kp @ a)/d(state_array)
    # = -x * Kp @ (da/dstate) - (dx/dstate) * (Kp @ a)^T
    Kp_a = Kp @ a  # shape (N,)
    term3 = -x * Kp @ da_dstate - np.outer(Kp_a, dx_dstate[0, :])
    
    # Term 4: d(-y * Pp @ a)/d(state_array)
    # = -y * Pp @ (da/dstate) - (dy/dstate) * (Pp @ a)^T
    Pp_a = Pp @ a  # shape (N,)
    term4 = -y * Pp @ da_dstate - np.outer(Pp_a, dy_dstate[0, :])
    
    # Combine and take first N-1 rows
    J_a_dot_red = (term1 + term2 + term3 + term4)[0:N-1, :]
    
    # === Block 2: d(b_dot)/d(state_array) ===
    # b_dot = x * Q @ a - M_mat @ b
    
    # Term 1: d(x * Q @ a)/d(state_array)
    # = x * Q @ (da/dstate) + (dx/dstate) * (Q @ a)^T
    Q_a = Q @ a  # shape (N-1,)
    term1_b = x * Q @ da_dstate + np.outer(Q_a, dx_dstate[0, :])
    
    # Term 2: d(-M_mat @ b)/d(state_array)
    term2_b = np.zeros((N-1, 3*N - 3))
    term2_b[:, N-1:2*N-2] = -M_mat
    
    J_b_dot = term1_b + term2_b
    
    # === Block 3: d(c_dot)/d(state_array) ===
    # c_dot = y * D @ a - N_mat @ c
    
    # Term 1: d(y * D @ a)/d(state_array)
    # = y * D @ (da/dstate) + (dy/dstate) * (D @ a)^T
    D_a = D @ a  # shape (N-1,)
    term1_c = y * D @ da_dstate + np.outer(D_a, dy_dstate[0, :])
    
    # Term 2: d(-N_mat @ c)/d(state_array)
    term2_c = np.zeros((N-1, 3*N - 3))
    term2_c[:, 2*N-2:3*N-3] = -N_mat
    
    J_c_dot = term1_c + term2_c
    
    # Combine all blocks
    J = np.vstack([J_a_dot_red, J_b_dot, J_c_dot])
    
    return J

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

    # compute J_f using existing function (expects signature: state_array, n, a_tot, ...)
    # params is passed-through tuple that matches MATTIA_FULL_ROOT_JACOBIAN
    Jf = MATTIA_FULL_ROOT_JACOBIAN(x, *params)  # shape (n,n)
    fx = MATTIA_FULL_ROOT(x, *params)           # shape (n,)

    # if no roots, it's trivial
    if len(roots) == 0:
        return Jf

    # Compute M and auxiliary quantities
    M = 1.0
    # store per-root quantities to reuse
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
        # handle s==0 safely (delta==0 contributes zero)
        if s == 0.0:
            continue
        # denominator: d * (1 + alpha*d)
        denom = d * (1.0 + alpha * d)
        coeff = -p * (s**(p-2)) / denom
        grad_sum += coeff * delta
    gradM = M * grad_sum  # length n

    # Construct Jacobian: JF = M * Jf + outer(fx, gradM)
    outer = np.outer(fx, gradM)  # shape (n,n)
    JF = M * Jf + outer
    return JF

# Adapter to match scipy.optimize.root jac signature jac(x, *args)
def jac_for_root(x, roots, params):
    return deflated_jacobian(x, roots, params)

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
    guesses.append(np.concatenate([np.array([1]), np.zeros(3*N-4)]))

    # THIS SEEMS TO REALLY HELP THE ROOT FINDER
    for i in range(5):
        t = np.random.rand(N)
        t /= t.sum()
        t = t[:-1]
        guesses.append(np.concatenate([t, np.zeros(2*N-2)]))

    for i in range(10):
        guesses.append(guess_generator(n, a_tot, x_tot, y_tot))


    a_0_ic = np.linspace(1e-5, a_tot - 1e-5, attempt_total_num)
    for i in a_0_ic:
        guesses.append(np.concatenate([np.array([i]), np.zeros(3*N-4)]))

    root_tol = 1e-8
    residual_tol = 1e-8
    euclidian_distance_tol = 1e-1
    # euclidian_distance_tol = 1e-2
    eigen_value_tol = 1e-12

    params = (n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)
    for guess in guesses:
        # assert np.any(np.array(guess) <= 0), f"Guess {guess} is non-physical"
        assert len(guess) == 3*N - 3, f"Guess {guess} is incorrect length"

        try:
            reduced_sol = root(
                deflated_function,
                x0=guess,
                args=(sol_list, params),
                jac=jac_for_root,
                method='hybr',
                tol=root_tol,
                options={'xtol': root_tol, 'maxfev': 4000})
                
            print(reduced_sol.message)
        except ValueError as e:
            continue

        reduced_sol = reduced_sol.x

        # J_analytical = MATTIA_FULL_ROOT_JACOBIAN(reduced_sol, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)
        # J_numerical = np.array([
        #     approx_fprime(reduced_sol, lambda s: MATTIA_FULL_ROOT(s, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)[i], epsilon=1e-8)
        #     for i in range(len(reduced_sol))
        # ])
        # print("Max difference:", np.max(np.abs(J_analytical - J_numerical)))
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

        def mattia_full_wrapper(v):
            return MATTIA_FULL_ROOT(v, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)

        jacobian_mattia_full = np.array([
            approx_fprime(reduced_sol, lambda s: mattia_full_wrapper(s)[i], epsilon=1e-8)
            for i in range(len(reduced_sol))])
        
        if not np.isfinite(jacobian_mattia_full).all():
            print("infinite detected")
            continue

        red_eigenvalues = LA.eigvals(jacobian_mattia_full)
        red_eigenvalues_real = np.real(red_eigenvalues)

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

    # k_positive_rates = np.array([8.40651484e+03, 5.48882357e-01, 2.63713975e+01])
    # k_negative_rates = np.array([1.32637580e-01, 9.14787266e-01, 8.89557552e+03])
    # p_positive_rates = np.array([7.41708599e+02, 1.47870389e+02, 5.91384022e-01])
    # p_negative_rates = np.array([13718.10403978, 57.50142864, 138.89310632])
    # alpha_matrix = np.array([[0.00000000e+00, 1.33367372e-01, 4.06884060e+00, 0.00000000e+00],
    #                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.33189760e-01],
    #                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.88500142e+04],
    #                             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])
    # beta_matrix = np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
    #                         [1.60769298e+06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
    #                         [1.53528882e+01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
    #                         [0.00000000e+00, 1.73856212e+02, 3.07486110e+05, 0.00000000e+00]])
    multistable_results = None
    
    Kp, Pp, G, H, Q, M_mat, D, N_mat = MATRIX_FINDER(n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates)

    unique_stable_fp_array, unique_unstable_fp_array, eigenvalues_array = root_finder(n, a_tot_value, x_tot_value, y_tot_value,
                                                     Kp, Pp, G, H, Q, M_mat, D, N_mat)
    
    possible_steady_states = np.floor((n + 2) / 2).astype(int)

    # if len(unique_unstable_fp_array) == 1:
    # if len(unique_stable_fp_array) == 2 and len(unique_unstable_fp_array) == 1:
    # if len(unique_stable_fp_array) == 2:
    # if len(unique_stable_fp_array) == possible_steady_states:
    if len(unique_stable_fp_array) > 1 or len(unique_unstable_fp_array) > 1:
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
    n = 4

    simulation_root(n, 4000)
    
if __name__ == "__main__":
    main()