import numpy as np
from typing import List, Tuple, Dict, Any, Set
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pickle
import random
from PHOS_FUNCTIONS import MATTIA_REDUCED, MATTIA_REDUCED_JACOBIAN, MATRIX_FINDER, pertubation_array_creation, matrix_clip, all_parameter_generation

def phosphorylation_system_solver(parameters_tuple, initial_states_array, final_t, full_model):

    n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates, a_tot, x_tot, y_tot = parameters_tuple
    N = n + 1

    assert np.all(initial_states_array <= (a_tot + 1e-12))
    assert np.all(initial_states_array >= 0)
    assert np.all(initial_states_array <= a_tot)

    #### SCALING TIME
    initial_t = 0
    t_span = (initial_t, final_t)

    ###### OBTAINING ALL MATRICES
    Kp, Pp, G, H, Q, M_mat, D, N_mat, L1, L2, W1, W2 = MATRIX_FINDER(n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates)
    cmap = plt.get_cmap('tab10')
    def color_for_species(idx):
        return cmap(idx % cmap.N)
    
    abstol = 1e-20
    reltol = 1e-5

    # if full_model == False:

    assert len(initial_states_array) == N

    mattia_reduced_parameter_tuple = (n, x_tot, y_tot, L1, L2, W1, W2)

    # ‘DOP853’ is recommended for solving with high precision (low values of rtol and atol).
    # method = 'DOP853'
    # if stiff, you should use ‘Radau’ or ‘BDF’. ‘LSODA’ can also be a good universal choice, 
    # but it might be somewhat less convenient to work with as it wraps old Fortran code.
    # 'LSODA' does not accept array-like jacobian

    method = 'LSODA'
    sol = solve_ivp(MATTIA_REDUCED, t_span = t_span, y0=np.asarray(initial_states_array, dtype=float),
                args = mattia_reduced_parameter_tuple, method = method, jac = MATTIA_REDUCED_JACOBIAN, 
                atol = abstol, rtol = reltol)
    
    a_solution_stack = np.stack([sol.y[i] for i in range(0, N)]) / a_tot

    plt.figure(figsize = (8, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    for i in range(a_solution_stack.shape[0]):
        color = color_for_species(i)
        plt.plot(sol.t, a_solution_stack[i], color=color, label = f"[$A_{i}]$", lw=4, alpha = 0.4)
        print(f"final A_{i} = {a_solution_stack[i][-1]}")
        plt.title(f"Plotting reduced phosphorylation dynamics for n = {n}")

    # if full_model == True:
    #     assert len(initial_states_array) == 3*n + 3
    #     # assert np.all(initial_states_array >= 1e-15)
    #     assert np.all(initial_states_array >= 0)

    #     mattia_parameter_tuple = (n, Kp, Pp, G, H, Q, M_mat, D, N_mat)
    #     # t_array = np.linspace(t_span[0], t_span[1], 500)
    #     sol = solve_ivp(MATTIA_FULL, t_span = t_span, y0=np.asarray(initial_states_array, dtype=float), 
    #                     args = mattia_parameter_tuple, method = 'LSODA', atol = abstol, rtol = reltol)

    #     a_solution_stack = np.stack([sol.y[i] for i in range(0, N)]) / a_tot


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

        # for i in range(a_solution_stack.shape[0]):
        #     color = color_for_species(i)
        #     plt.plot(sol.t, a_solution_stack[i], color=color, label = f"$[A_{i}]$", lw=4, alpha = 0.4)
        #     print(f"final numerical fp of A_{i}: {a_solution_stack[i][-1]}")
        #     plt.title(f"full system dynamics for n = {n}")
        # plt.plot(sol.t, x_solution, color='black', label="$X$", lw=1.75, alpha = 0.75)
        # plt.plot(sol.t, y_solution, color='gray', label="$Y$", lw=1, alpha = 0.75)
    # plt.figure(figsize = (8, 6))

    plt.ylabel("concentration")
    plt.xlabel("time")
    plt.minorticks_on()
    plt.tight_layout()
    plt.xlim(t_span[0] - 0.1, t_span[1] + 0.1)
    plt.ylim(-0.05, 1.1)
    plt.legend(frameon=False)
    plt.show()
    
def plotter(full_model_bool, index, final_t, file_name):
    n = int(file_name[-5:-4])
    N = n + 1

    with open(file_name, "rb") as f:
        multistable_results = pickle.load(f)
    print(len(multistable_results))
    for i in range(len(multistable_results)):
        print(f"Index {i}: # of stable states:")
        print(f"{multistable_results[i]['stable_states']}")
    for i in range(len(multistable_results)):
        print(f"Index {i}: # of unstable states:")
        print(f"{multistable_results[i]['unstable_states']}")
    a_stable_states = multistable_results[index]["stable_states"]
    a_unstable_states = multistable_results[index]["unstable_states"]
    num_stable_states = multistable_results[index]["num_of_stable_states"]
    num_unstable_states = multistable_results[index]["num_of_unstable_states"]
    a_tot_parameter = multistable_results[index]["total_concentration_values"][0]
    x_tot_parameter = multistable_results[index]["total_concentration_values"][1]
    y_tot_parameter = multistable_results[index]["total_concentration_values"][2]
    alpha_matrix = multistable_results[index]["alpha_matrix"]
    beta_matrix = multistable_results[index]["beta_matrix"]
    k_positive_rates = multistable_results[index]["k_positive_rates"]
    k_negative_rates = multistable_results[index]["k_negative_rates"]
    p_positive_rates = multistable_results[index]["p_positive_rates"]
    p_negative_rates = multistable_results[index]["p_negative_rates"]
    # eigenvalues = multistable_results[index]["eigenvalues"]

    # a_stable_states = np.ravel(a_stable_states)*a_tot_parameter
    # for i, fp in enumerate(a_stable_states):
    #     print(f"stable state {i}: {fp}")
    # for i, fp in enumerate(a_unstable_states):
    #     print(f"unstable state {i}: {fp}")

    rate_min, rate_max = 1e-1, 1e7

    # clipping
    alpha_matrix = alpha_matrix / np.mean(alpha_matrix)
    beta_matrix = beta_matrix / np.mean(beta_matrix)
    alpha_matrix = matrix_clip(alpha_matrix); beta_matrix = matrix_clip(beta_matrix)

    k_positive_rates = k_positive_rates / np.mean(k_positive_rates)
    p_positive_rates = p_positive_rates / np.mean(p_positive_rates)
    k_positive_rates = np.clip(k_positive_rates, rate_min, rate_max)
    p_positive_rates = np.clip(p_positive_rates, rate_min, rate_max)
    k_negative_rates = k_positive_rates
    p_negative_rates = p_positive_rates

    # pick a random unstable state as initial condition, usually just 1 exists
    # a_init_conditions_reduced = a_unstable_states[np.random.randint(0, num_unstable_states)]
    # a_init_conditions_full = a_init_conditions_reduced
    a_init_conditions_reduced = a_stable_states[np.random.randint(0, num_stable_states)]

    pertubation_parameter = 0.05

    # if full_model_bool == False:

    pertubation_array = pertubation_array_creation(a_init_conditions_reduced, pertubation_parameter)
    initial_states_array_reduced_perturbed = a_init_conditions_reduced + pertubation_array
    print(f"Initial condition is: {initial_states_array_reduced_perturbed}")
    # print(f"The initial condition is: {initial_states_array_reduced}")
    
    # for alpha in [i/10 for i in range(1,10)]:
    #     initial_states_array_reduced_perturbed = np.concatenate([np.array([alpha]), np.zeros(N-2), np.array([1-alpha])])
    #     print(f"Initial condition (reduced) is: {initial_states_array_reduced_perturbed}")
    #     parameters_tuple = (n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates, a_tot_parameter, x_tot_parameter, y_tot_parameter)
    #     phosphorylation_system_solver(parameters_tuple, initial_states_array_reduced_perturbed, final_t, full_model_bool)

    parameters_tuple = (n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates, a_tot_parameter, x_tot_parameter, y_tot_parameter)
    # phosphorylation_system_solver(parameters_tuple, initial_states_array_reduced_perturbed, final_t, full_model_bool)
    phosphorylation_system_solver(parameters_tuple, initial_states_array_reduced_perturbed, final_t, full_model_bool)

    # if full_model_bool == True:
        # Kp, Pp, G, H, Q, M_mat, D, N_mat, L1, L2, W1, W2 = MATRIX_FINDER(n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates)

    #     ones_vec = np.ones((1, N-1), dtype = float)

    #     # # a_init_conditions = a_stable_states
    #     x_init_condition = x_tot_parameter / float(1.0 + np.dot(ones_vec, (W1 @ a_init_conditions_full)))
    #     y_init_condition = y_tot_parameter / float(1.0 + np.dot(ones_vec, (W2 @ a_init_conditions_full)))

    #     b_init_conditions = x_init_condition * (W1 @ a_init_conditions_full)
    #     c_init_conditions = y_init_condition * (W2 @ a_init_conditions_full)
    #     print(b_init_conditions)
    #     print(c_init_conditions)
    #     # if random.random() < 0.5:
    #     #     pertubation_array[0] = -pertubation_parameter; pertubation_array[2] = pertubation_parameter
    #     # else:
    #     #     pertubation_array[0] = pertubation_parameter; pertubation_array[2] = -pertubation_parameter

    #     # print(len(initial_states_array_full_random))
    #     # pertubation_array = np.zeros(3*N); 
    #     if random.random() < 0.5:
    #         pertubation_array = pertubation_array_creation(a_init_conditions_full, pertubation_parameter, True)
    #     else:
    #         pertubation_array = pertubation_array_creation(a_init_conditions_full, pertubation_parameter, False)
    #     initial_states_array_full = np.concatenate([np.ravel(a_init_conditions_full + pertubation_array), np.ravel(b_init_conditions), np.ravel(c_init_conditions), np.ravel(x_init_condition), np.ravel(y_init_condition)]).astype(float)
    #     print(len(initial_states_array_full))
    #     parameters_tuple = (n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates, a_tot_parameter, x_tot_parameter, y_tot_parameter)
    #     phosphorylation_system_solver(parameters_tuple, initial_states_array_full, final_t, full_model_bool)
    # return eigenvalues

def main():
    file_name = "multistability_parameters_200_2.pkl"

    index = 0
    # subindex = 0
    full_model_bool = False
    final_t = 100000
    plotter(full_model_bool, index, final_t, file_name)
    # print(f"Eigenvalues are: {eigenvalues}")
    # plt.hist(eigenvalues)
    # plt.show()
if __name__ == "__main__":
    main()