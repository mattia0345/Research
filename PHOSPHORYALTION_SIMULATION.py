import numpy as np
from typing import List, Tuple, Dict, Any, Set
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pickle
import random
from PHOS_FUNCTIONS import MATRIX_FINDER, MATTIA_FULL, guess_generator, matrix_normalize

def sim_full_jacobian(t, state_array, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat):
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

    state_norm = np.linalg.norm(state_array)
    dot_norm = np.linalg.norm(state_dot)

    denom = max(state_norm, eps)
    rel_change = dot_norm / denom

    value = rel_change - threshold

    return value

stable_event.terminal = True
stable_event.direction = 0

def phosphorylation_system_solver(parameters_tuple, initial_states_array, final_t):

    n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates, a_tot, x_tot, y_tot = parameters_tuple
    N = 2**n

    assert np.all(initial_states_array >= 0)

    #### SCALING TIME
    initial_t = 0
    t_span = (initial_t, final_t)

    ###### OBTAINING ALL MATRICES
    Kp, Pp, G, H, Q, M_mat, D, N_mat = MATRIX_FINDER(n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates)
    
    abstol = 1e-15
    reltol = 1e-5

    mattia_full_parameter_tuple = (n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)

    method = 'LSODA'
    sol = solve_ivp(MATTIA_FULL, t_span = t_span, y0=np.asarray(initial_states_array, dtype=float),
                jac = sim_full_jacobian, args = mattia_full_parameter_tuple, method = method, atol = abstol, rtol = reltol)
    print(sol.message)
    
    # Return solution for A_0
    a_0_solution = sol.y[0]
    
    return sol.t, a_0_solution

def plotter(index, final_t, file_name):
    n = int(file_name[-5:-4])
    N = 2**n

    with open(file_name, "rb") as f:
        multistable_results = pickle.load(f)
    for i in range(len(multistable_results)):
        print(f"index {i}:")
        # print(f"stable states:")
        for j in range(len(multistable_results[i]['stable_states'])):
            print(f"stable value of [A_0]: {multistable_results[i]['stable_states'][j][0]}")
        if len(multistable_results[i]['unstable_states']) != 0:
            for k in range(len(multistable_results[i]['unstable_states'])):
                print(f"unstable value of [A_0]: {multistable_results[i]['unstable_states'][k][0]}")
        print()
    a_stable_states = multistable_results[index]["stable_states"]
    a_tot_parameter = multistable_results[index]["total_concentration_values"][0]
    x_tot_parameter = multistable_results[index]["total_concentration_values"][1]
    y_tot_parameter = multistable_results[index]["total_concentration_values"][2]
    alpha_matrix = multistable_results[index]["alpha_matrix"]
    beta_matrix = multistable_results[index]["beta_matrix"]
    k_positive_rates = multistable_results[index]["k_positive_rates"]
    k_negative_rates = multistable_results[index]["k_negative_rates"]
    p_positive_rates = multistable_results[index]["p_positive_rates"]
    p_negative_rates = multistable_results[index]["p_negative_rates"]

    eigenvalues = multistable_results[index]["eigenvalues"]

    # eigen_value_tol = 1e-8
    # for e in eigenvalues:
    #     if np.max(e) > eigen_value_tol:
    #         print(f"UNSTABLE EIGENVALUES: {e}")
    #     elif np.all(e < -eigen_value_tol):
    #         print(f"STABLE EIGENVALUES: {e}")

    attempt_total_num = 25
    initial_conditions_list = []
    # initial_conditions_list.append(np.zeros(3*N - 3))
    # initial_conditions_list.append(1e-5*np.ones(3*N - 3))
    # initial_conditions_list.append(np.concatenate([np.array([1]), np.zeros(3*N-4)]))

    for i in range(8):
        t = np.random.rand(N)
        t /= t.sum()
        t = t[:-1]
        initial_conditions_list.append(np.concatenate([t, np.zeros(2*N-2)]))

    p = 1e-10
    a_0_ic = np.linspace(p, a_tot_parameter - p, attempt_total_num)
    for i in a_0_ic:
        initial_conditions_list.append(np.concatenate([np.array([i]), np.zeros(3*N-4)]))

    # Create single figure
    plt.figure(figsize = (8, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    cmap = plt.get_cmap('plasma')
    
    # Plot A_0 for each initial condition
    for idx, guess in enumerate(initial_conditions_list):
        parameters_tuple = (n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates, a_tot_parameter, x_tot_parameter, y_tot_parameter)
        t_sol, a_0_sol = phosphorylation_system_solver(parameters_tuple, np.array(guess), final_t)
        
        color = cmap(idx / len(initial_conditions_list))
        plt.plot(t_sol, a_0_sol, lw=1, alpha=1, color = color)
    
    plt.ylabel("$[A_0]$")
    plt.xlabel("time")
    # plt.xscale('log')
    # plt.yscale('log')
    plt.title(f"$A_0$ dynamics for n = {n} with different initial conditions")
    plt.minorticks_on()
    # plt.xlim(0 - 0.1, final_t + 0.1)
    # plt.ylim(-0.01, a_tot_parameter + 0.01)
    # plt.ylim(1, 1000)
    plt.yscale('log')
    plt.xscale('log')
    # plt.ylim(8e-7, 0.0006)
    plt.xlim(1e-9, final_t)
    plt.tight_layout()
    # plt.legend(frameon=False, ncol=2, fontsize=8)
    plt.show()

def main():
    file_name = "multistability_parameters_100_2.pkl"

    index = 31

    final_t = 50000
    plotter(index, final_t, file_name)

if __name__ == "__main__":
    main()