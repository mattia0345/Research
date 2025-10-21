import numpy as np
import matplotlib.pyplot as plt
import random as random

def flip(p: float) -> bool:
    """Returns True with probability p and False with probability 1-p."""
    return True if random.random() < p else False

def propensity_vector(N_vector: np.ndarray, k_vector: np.ndarray, params: np.ndarray) -> np.ndarray:
    A, B, C = N_vector
    k1, k2, k3 = k_vector
    r_a, r_c, A_tilda, C_tilda, V = params
    # return np.array([k1*A, k1*B, k2*A*B / V, k2*B*(B-1)/2, k3*B, k3*C, r_a*A_tilda, r_a*A, r_c*C_tilda, r_c*C])
    return np.array([k1*A, k1*B, k2*A*B, k2*B*(B-1)/2, k3*B, k3*C, r_a*A_tilda, r_a*A, r_c*C_tilda, r_c*C])

def updated_state(N_vector: np.ndarray, reaction_index: int) -> np.ndarray:
    stoichiometric_matrix = np.array([[-1, 1, 0],  # A -> B
                                      [1, -1, 0],  # B -> A
                                      [-1, 1, 0],  # A + B -> 2B
                                      [1, -1, 0], # 2B -> A + B
                                      [0, -1, 1],  # B -> C
                                      [0, 1, -1],  # C -> B 
                                      [1, 0, 0],
                                      [-1, 0, 0],
                                      [0, 0, 1],
                                      [0, 0, -1]])  # C -> C_tilda
    return N_vector + stoichiometric_matrix[reaction_index]
                                                                       
def main() -> None:
    V = 3  # volume
    A_initial = 1; B_initial = 2; C_initial = 3
    A_state = [A_initial]; B_state = [B_initial]; C_state = [C_initial]
    initial_state = np.array([A_initial, B_initial, C_initial])
    A_tilda_initial = 2; C_tilda_initial = 4; r_a = 1; r_c = 1; param_array = np.array([r_a, r_c, A_tilda_initial, C_tilda_initial, V])

    t = float(1e-2); time = [t]

    k1 = 1; k2 = 1; k3 = 1; k_vector = np.array([k1, k2, k3])

    negative_count_flag = False

    while t < 1000: 
        A_vector = propensity_vector(initial_state, k_vector, param_array)
        A_sum = np.sum(A_vector)
        if not np.isfinite(A_sum) or A_sum <= 1e-300:
            break
        r1, r2 = np.random.uniform(0, 1, 2)
        tau = np.log(1 / r1) / A_sum
        
        if A_sum == 0:
            break
        i = np.searchsorted(np.cumsum(A_vector), r2*A_sum, side="right")
        
        initial_state = updated_state(initial_state, i)
        if np.any(initial_state < 0):
            raise RuntimeError("Negative count after update; check stoichiometry/propensities.")

        t += tau
        time.append(t)

        A_state.append(initial_state[0])
        B_state.append(initial_state[1])
        C_state.append(initial_state[2])
        

    

    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12,5))
    axes[0].plot(time, np.array(A_state) / V, label='[A]', color='blue')
    axes[0].set_xscale('log')
    axes[0].legend()
    axes[1].plot(time, np.array(B_state) / V, label='[B]', color='red')
    axes[1].set_xscale('log')
    axes[1].legend()
    axes[2].plot(time, np.array(C_state) / V, label='[C]', color='green')
    axes[2].set_xscale('log')
    axes[2].legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()