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
import multiprocessing
import time as time
import threading
from multiprocessing import Process, Queue
from PHOS_FUNCTIONS import MATRIX_FINDER, MATTIA_FULL, guess_generator, matrix_normalize, matrix_sample_reciprocal
from PHOS_FUNCTIONS import duplicate, all_parameter_generation

# def _solver_worker(queue, parameters_tuple, initial_states_array, final_t, 
#                    abstol, reltol, method):
#     """Worker function that runs in a separate process"""
#     try:
#         n, a_tot, _, _, _, _, _, _, _, _, _, _ = parameters_tuple
#         N = 2**n
#         initial_t = 0
#         t_span = (initial_t, final_t)
        
#         # Use stable_event only (no timeout event - that's handled by process termination)
#         sol = solve_ivp(
#             MATTIA_FULL,
#             t_span=t_span,
#             y0=np.asarray(initial_states_array, dtype=float),
#             events=stable_event,
#             args=parameters_tuple,
#             method=method,
#             atol=abstol,
#             rtol=reltol,
#             dense_output=False,  # Don't store dense output to save memory
#             max_step=np.inf  # Allow large steps if needed
#         )
        
#         # Put result in queue
#         result = {
#             'success': True,
#             'y': sol.y,
#             't': sol.t,
#             'message': sol.message,
#             'status': sol.status,
#             't_events': sol.t_events,
#             'final_state': sol.y.T[-1] if sol.y.size > 0 else np.full((3*N - 3,), np.nan)
#         }
#         queue.put(result)
        
#     except Exception as e:
#         # Put error in queue
#         n, a_tot, _, _, _, _, _, _, _, _, _, _ = parameters_tuple
#         N = 2**n
#         result = {
#             'success': False,
#             'error': str(e),
#             'final_state': np.full((3*N - 3,), np.nan)
#         }
#         queue.put(result)

def MATTIA_FULL_ROOT_JACOBIAN(t, state_array, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat):
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

def phosphorylation_system_solver(parameters_tuple,
                                  initial_states_array,
                                  final_t,
                                  plot,
                                  timeout_seconds: float = 15.0):
    """
    Solver with timeout using threading.
    Compatible with joblib's loky backend.
    """
    n, a_tot, _, _, _, _, _, _, _, _, _, _ = parameters_tuple
    N = 2**n
    assert np.all(initial_states_array >= 0)
    assert len(initial_states_array) == 3*N - 3

    initial_t = 0
    t_span = (initial_t, final_t)
    abstol = 1e-6
    reltol = 1e-3
    method = "LSODA"

    # Shared variables for thread communication
    result = {'sol': None, 'error': None, 'finished': False}
    start_time = time.time()
    
    def solver_thread():
        """Thread that runs the solver"""
        try:
            sol = solve_ivp(
                MATTIA_FULL,
                t_span=t_span,
                y0=np.asarray(initial_states_array, dtype=float),
                events=stable_event,
                args=parameters_tuple,
                method=method,
                atol=abstol,
                rtol=reltol,
                dense_output=False,
            )
            result['sol'] = sol
            result['finished'] = True
        except Exception as e:
            result['error'] = str(e)
            result['finished'] = True
    
    # Start solver in separate thread
    thread = threading.Thread(target=solver_thread, daemon=True)
    thread.start()
    
    # Wait for completion or timeout
    thread.join(timeout=timeout_seconds)
    elapsed = time.time() - start_time
    
    # Check if thread is still alive (timeout)
    if thread.is_alive():
        # print(f"Integration TIMEOUT after {elapsed:.2f}s (thread still running).")
        # Note: We can't forcibly stop the thread, but marking it as daemon
        # means it won't prevent program exit
        return np.full((3*N - 3,), np.nan, dtype=float)
    
    # Check for errors
    if result['error'] is not None:
        # print(f"Integration error: {result['error']}")
        return np.full((3*N - 3,), np.nan, dtype=float)
    
    # Check if solver finished
    if not result['finished'] or result['sol'] is None:
        # print("Integration did not complete properly.")
        return np.full((3*N - 3,), np.nan, dtype=float)
    
    sol = result['sol']
    
    # Determine how solver stopped
    stable_hit = False
    if sol.t_events is not None and len(sol.t_events) > 0:
        if sol.t_events[0] is not None and len(sol.t_events[0]) > 0:
            stable_hit = True
    
    # if stable_hit:
    #     print(f"Integration converged (stable_event) in {elapsed:.2f}s.")
    # else:
    #     print(f"Integration completed normally in {elapsed:.2f}s. Status: {sol.message}")
    
    # Return final state
    try:
        return sol.y.T[-1]
    except Exception:
        return np.full((3*N - 3,), np.nan, dtype=float)
    # try:
    #     sol = solve_ivp(
    #         MATTIA_FULL,
    #         t_span=t_span,
    #         y0=np.asarray(initial_states_array, dtype=float),
    #         events=(stable_event, timeout_event),
    #         args=parameters_tuple,
    #         method=method,
    #         atol=abstol,
    #         rtol=reltol,
    #     )
    # except ValueError:
    #     # fallback without events if LSODA/event combination raised ValueError
    #     sol = solve_ivp(
    #         MATTIA_FULL,
    #         t_span=t_span,
    #         y0=np.asarray(initial_states_array, dtype=float),
    #         args=parameters_tuple,
    #         method=method,
    #         atol=abstol,
    #         rtol=reltol,
    #     )

    # # sol.message is informative
    # print(sol.message)

    # # determine how the solver stopped
    # timed_out = False
    # stable_hit = False
    # # sol.t_events is a list corresponding to the order of events passed
    # if sol.t_events is not None:
    #     # sol.t_events[0] corresponds to stable_event, sol.t_events[1] to timeout_event
    #     if len(sol.t_events) >= 2:
    #         if sol.t_events[1] is not None and len(sol.t_events[1]) > 0:
    #             timed_out = True
    #         if sol.t_events[0] is not None and len(sol.t_events[0]) > 0:
    #             stable_hit = True

    # if timed_out:
    #     print(f"Integration stopped because timeout_event fired after ~{timeout_seconds} s.")
    # elif stable_hit:
    #     print("Integration stopped because stable_event fired (converged).")
    # else:
    #     print("Integration ended normally (t reached final_t or solver terminated for other reason).")

    # # plotting code: you can reuse your existing plotting block; use sol.y / sol.t

    # # return last state (same as previously)
    # try:
    #     return sol.y.T[-1]
    # except Exception:
    #     return np.full((3*N - 3,), np.nan, dtype=float)

def fp_finder(n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat):

    N = 2**n

    fp = []
    # norm_tol = 0.00001
    final_t = 500000
    stable_points = []
    plot = False
    euclidian_distance_tol = 1e-1
    attempt_total_num = 20
    guesses = []
    # guesses.append(1e-14*np.ones(3*N - 3))
    # guesses.append(np.concatenate([np.array([1-1e-14]), np.zeros(3*N-4)]))
    # guesses.append(np.zeros(3*N - 3))
    # guesses.append(1e-5*np.ones(3*N - 3))
    # guesses.append(np.concatenate([np.array([1]), np.zeros(3*N-4)]))

    a_0_ic = np.linspace(1e-20, a_tot - 1e-20, attempt_total_num)
    for ic in a_0_ic:
        guesses.append(np.concatenate([np.array([ic]), np.zeros(3*N-4)]))
    # for i in range(5):
    #     t = np.random.rand(N)
    #     t /= t.sum()
    #     t = t[:-1]
    #     guesses.append(np.concatenate([t, np.zeros(2*N-2)]))

    # for i in range(attempt_total_num):
    #     guesses.append(np.concatenate([np.array([random.random()]), np.zeros(3*N-4)]))
    #     guesses.append(guess_generator(n, a_tot, x_tot, y_tot))


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
        if np.any(reduced_sol < -zero_tol) or np.any(reduced_sol > 1 + zero_tol):
            # print(reduced_sol)
            print("Non-physical fixed point.")
            continue

        is_duplicate = False

        for r in fp:
            if np.linalg.norm(reduced_sol - r) <= euclidian_distance_tol:
                is_duplicate = True
                break
        if is_duplicate:
            continue
        # def mattia_full_wrapper(state):
        #     return MATTIA_FULL(0, state, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)

        # jacobian_mattia_full = np.array([
        #     approx_fprime(reduced_sol, lambda s: mattia_full_wrapper(s)[i], epsilon=1e-8)
        #     for i in range(len(reduced_sol))])
        
        # eigenvalues = LA.eigvals(jacobian_mattia_full)
        # eigenvalues_real = np.real(eigenvalues)
        # print(f"Eigenvalues at fixed point: {eigenvalues_real}")

        # if duplicate(reduced_sol, stable_points, norm_tol):
        #     # print("Duplicate found, skipping")
        #     continue

        fp.append(reduced_sol)

        stable_points.append(reduced_sol)

    # if len(stable_points) == 0:
    #     print("No valid steady states found.")
    #     return np.array([])
    
    # stable_points = np.array(stable_points)  # shape (num_trials, n_variables)

    # db = DBSCAN(eps=1e-1, min_samples=1).fit(stable_points)
    # labels = db.labels_

    # # Extract representative points
    # unique_points = np.array([stable_points[labels == i][0] for i in np.unique(labels)])
    # print(f"Found {len(unique_points)} unique steady states.")

    print(f"Found {len(fp)} unique steady states.")

    if len(fp) == 0:
        return np.array([])
            
    return np.array(fp)
    # if len(unique_points) == 0:
    #     return np.array([])
            
    # return unique_points

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

    unique_stable_fp_array = fp_finder(n, a_tot_value, x_tot_value, y_tot_value,
                                                    Kp, Pp, G, H, Q, M_mat, D, N_mat)

    possible_steady_states = np.floor((n + 2) / 2).astype(int)
    print(i)

    if len(unique_stable_fp_array) == possible_steady_states or len(unique_stable_fp_array) == 2:

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
    a_tot_value = 1000
    # N = n + 1
    N = 2**n
    x_tot = 1
    y_tot = 1
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
    simulation(n, 500)
    
if __name__ == "__main__":
    main()