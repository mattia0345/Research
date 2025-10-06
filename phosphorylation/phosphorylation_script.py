import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Dict, Any, Set
from phosphorylation_functions import all_parameter_generation, generate_phosphorylation_ODES
from matplotlib import cm
from scipy.integrate import solve_ivp

def main(n: int) -> None:

    gen = all_parameter_generation(n, distribution = "gamma", params=(1, 1), verbose=False)

    alpha_out_matrix, alpha_in_matrix, alpha_out, alpha_in = gen.alpha_parameter_generation()
    beta_out_matrix, beta_in_matrix, beta_out, beta_in = gen.beta_parameter_generation()

    k_positive_rates, k_negative_rates = gen.k_parameter_generation()
    p_positive_rates, p_negative_rates = gen.p_parameter_generation()
    distribution_params = [2.0, 2.0]

    ######################
    # k_positive_custom = [8.12e-3, 1.02e-1, 8.12e-3, 1.02e-1] # a_E
    # p_positive_custom = [1.6e-2, 2.04e-1, 1.6e-2, 2.04e-1]   # b_E




    ######################

    params = (n, "gamma", 1, 1, alpha_out_matrix, alpha_in_matrix, alpha_out, alpha_in, beta_out_matrix, beta_in_matrix, beta_out, beta_in, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates)
    initial_states_array = np.random.randint(1, 2, 3 * (2**n))

    # state_array = generate_phosphorylation_ODES(1, initial_states_array, params_list)
    # print(state_array)


    sol = solve_ivp(generate_phosphorylation_ODES, t_span=(0, 500), y0=np.asarray(initial_states_array, dtype=float), args = params)
    # num_phos = sol.shape[1]

    cmap = cm.get_cmap('tab10')  # choose 'tab10' or 'tab20'

    def color_for_species(idx):
        return cmap(idx % cmap.N)
    
    print(type(sol))


    # for i in range(num_phos):
    #     color = color_for_species(i)
    #     label = f"$n_{i+1}$"
    #     plt.plot(sol.t, sol[i], color=color, label=label)
        
    plt.figure(figsize = (10, 10))
    plt.ylabel("Concentration")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main(4)