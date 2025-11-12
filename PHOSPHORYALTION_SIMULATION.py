import numpy as np
from typing import List, Tuple, Dict, Any, Set
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pickle
import random
from PHOS_FUNCTIONS import MATRIX_FINDER, MATTIA_FULL, guess_generator, matrix_normalize

def phosphorylation_system_solver(parameters_tuple, initial_states_array, final_t):

    n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates, a_tot, x_tot, y_tot = parameters_tuple
    N = 2**n

    assert np.all(initial_states_array >= 0)

    #### SCALING TIME
    initial_t = 0
    t_span = (initial_t, final_t)

    ###### OBTAINING ALL MATRICES
    Kp, Pp, G, H, Q, M_mat, D, N_mat = MATRIX_FINDER(n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates)
    cmap = plt.get_cmap('Accent')
    def color_for_species(idx):
        return cmap(idx / (3 * N - 3))  # smooth gradient of distinct colors
    
    abstol = 1e-20
    reltol = 1e-5

    mattia_full_parameter_tuple = (n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat)

    method = 'LSODA'
    sol = solve_ivp(MATTIA_FULL, t_span = t_span, y0=np.asarray(initial_states_array, dtype=float),
                args = mattia_full_parameter_tuple, method = method, atol = abstol, rtol = reltol)
    
    a_solution_stack = np.stack([sol.y[i] for i in range(0, N - 1)])
    b_solution_stack = np.stack([sol.y[i] for i in range(N-1, 2*N - 2)]) 
    c_solution_stack = np.stack([sol.y[i] for i in range(2*N - 2, 3*N - 3)])
    a_N_array = a_tot - np.sum(a_solution_stack, axis=0) - np.sum(b_solution_stack, axis = 0) - np.sum(c_solution_stack, axis=0)

    plt.figure(figsize = (8, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    for i in range(a_solution_stack.shape[0]):
        color = color_for_species(i)
        plt.plot(sol.t, a_solution_stack[i], color=color, label = f"[$A_{i}]$", lw=4, alpha = 0.9)
        # print(f"final A_{i} = {a_solution_stack[i][-1]}")
        plt.title(f"phosphorylation dynamics for n = {n}")

    color_N = color_for_species(N-1)
    plt.plot(sol.t, a_N_array, color=color_N, label=f"$[A_{{{N-1}}}]$", lw=4, alpha=0.9)
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


    plt.ylabel("concentration")
    plt.xlabel("time")
    plt.minorticks_on()
    plt.tight_layout()
    plt.xlim(t_span[0] - 0.1, t_span[1] + 0.1)
    plt.ylim(-0.05, 1.1)
    plt.legend(frameon=False)
    plt.show()
    
def plotter(index, final_t, file_name):
    n = int(file_name[-5:-4])
    N = 2**n

    with open(file_name, "rb") as f:
        multistable_results = pickle.load(f)
    for i in range(len(multistable_results)):
        print(f"Index {i}: # of stable states:")
        for j in range(len(multistable_results[i]['stable_states'])):
            print(f"{multistable_results[i]['stable_states'][j]}")
    # for i in multistable_results:
    #     # print(f"Index {i}: # of stable states:")
    #     print(i)
        # print(f"{multistable_results[i]['stable_states']}")
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


    # NORMALIZING
    # sum_rates = np.sum(alpha_matrix) + np.sum(beta_matrix) + np.sum(k_positive_rates) + \
    #      np.sum(k_negative_rates) + np.sum(p_positive_rates) + np.sum(p_negative_rates)
    # mean_all_rates = sum_rates / 20
    # alpha_matrix = alpha_matrix / mean_all_rates
    # beta_matrix = beta_matrix / mean_all_rates
    # k_positive_rates = k_positive_rates / mean_all_rates
    # k_negative_rates = k_negative_rates / mean_all_rates
    # p_positive_rates = p_positive_rates / mean_all_rates
    # p_negative_rates = p_negative_rates / mean_all_rates

    attempt_total_num = 8
    initial_conditions_list = []
    # guesses.append(1e-14*np.ones(3*N - 3))
    # guesses.append(np.concatenate([np.array([1-1e-14]), np.zeros(3*N-4)]))
    initial_conditions_list.append(np.zeros(3*N - 3))
    initial_conditions_list.append(1e-5*np.ones(3*N - 3))
    initial_conditions_list.append(np.concatenate([np.array([1]), np.zeros(3*N-4)]))

    for i in range(5):
        t = np.random.rand(N)
        t /= t.sum()
        t = t[:-1]
        initial_conditions_list.append(np.concatenate([t, np.zeros(2*N-2)]))

    for i in range(attempt_total_num):
        initial_conditions_list.append(np.concatenate([np.array([random.random()]), np.zeros(3*N-4)]))
        # initial_conditions_list.append(guess_generator(n, a_tot, x_tot, y_tot))
        initial_conditions_list.append(guess_generator(n, a_tot_parameter, x_tot_parameter, y_tot_parameter))

    for guess in initial_conditions_list:
        # initial_states_array = np.concatenate([np.array([random.random()]), np.zeros(3*N-4)])
        parameters_tuple = (n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates, a_tot_parameter, x_tot_parameter, y_tot_parameter)
        phosphorylation_system_solver(parameters_tuple, np.array(guess), final_t)

def main():
    file_name = "multistability_parameters_1500_2.pkl"

    index = 1

    final_t = 10000000
    plotter(index, final_t, file_name)

if __name__ == "__main__":
    main()