import numpy as np
import sympy as sp
from typing import List, Tuple, Dict, Any, Set

class all_parameter_generation:
    """
    Generate state transitions and random parameters (a, b, c, enzyme) for an n-site phosphorylation model.

    Args:
        n: number of sites (int)
        distribution: distribution name ("gamma" supported)
        params: parameters for the distribution (for gamma: [shape, scale])
        verbose: if True, prints transitions and matrices
    """
    def __init__(self, n: int, distribution: str = "gamma", params: List[float] = [1.0, 1.0],
                 verbose: bool = False):
        self.n = n
        self.num_states = 2 ** n
        self.distribution = distribution
        self.params = params
        self.verbose = verbose
        self.rng = np.random.default_rng()

    @staticmethod
    def padded_binary(i: int, n: int) -> str:
        return bin(i)[2:].zfill(n)

    @staticmethod
    def binary_string_to_array(string: str) -> np.ndarray:
        return np.array([int(i) for i in string], dtype=int)

    def calculate_valid_transitions(self) -> Tuple[List[List[Any]], List[List[Any]]]:
        """
        Returns:
            valid_E_reactions: list of [state_i_str, state_j_str, i, j, "E"]
            valid_F_reactions: list of [state_i_str, state_j_str, i, j, "F"]
        """
        all_states = [self.padded_binary(i, self.n) for i in range(self.num_states)]

        valid_difference_vectors: Set[Tuple[int, ...]] = set()
        valid_E_reactions: List[List[Any]] = []
        valid_F_reactions: List[List[Any]] = []

        for i in range(self.num_states):
            arr_i = self.binary_string_to_array(all_states[i])
            for j in range(self.num_states):
                if i == j:
                    continue
                arr_j = self.binary_string_to_array(all_states[j])
                diff = arr_j - arr_i
                if np.sum(np.abs(diff)) == 1:
                    # +1 -> phosphorylation (E), -1 -> dephosphorylation (F)
                    element = "E" if np.any(diff == 1) else "F"
                    if element == "E":
                        if self.verbose:
                            print(f"{all_states[i]} --> {all_states[j]} (E), {i}, {j}")
                        valid_E_reactions.append([all_states[i], all_states[j], i, j, element])
                    else:
                        if self.verbose:
                            print(f"{all_states[i]} --> {all_states[j]} (F), {i}, {j}")
                        valid_F_reactions.append([all_states[i], all_states[j], i, j, element])
                    valid_difference_vectors.add(tuple(diff))

        return valid_E_reactions, valid_F_reactions

    def alpha_parameter_generation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                            Dict[int, List[int]], Dict[int, List[int]],
                                            Dict[int, List[int]], Dict[int, List[int]]]:
        
        valid_E_reactions, valid_F_reactions = self.calculate_valid_transitions()

        shape, scale = self.params
        # matrices are num_states x num_states

        alpha_out_matrix = np.zeros((self.num_states, self.num_states))
        alpha_in_matrix = np.zeros((self.num_states, self.num_states))

        alpha_out: Dict[int, List[int]] = {}
        alpha_in: Dict[int, List[int]] = {}
        for _, _, i, j, _ in valid_E_reactions:
            alpha_out.setdefault(i, []).append(j)
            alpha_in.setdefault(j, []).append(i)
            alpha_out_matrix[i, j] = self.rng.gamma(shape, scale)
            alpha_in_matrix[j, i] = self.rng.gamma(shape, scale)

        return alpha_out_matrix, alpha_in_matrix, alpha_out, alpha_in
        # return (c_E_out_matrix, c_F_out_matrix, c_E_in_matrix, c_F_in_matrix,
        #         cE_out, cF_out, cE_in, cF_in)

    def beta_parameter_generation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                            Dict[int, List[int]], Dict[int, List[int]],
                                            Dict[int, List[int]], Dict[int, List[int]]]:
        
        valid_E_reactions, valid_F_reactions = self.calculate_valid_transitions()

        shape, scale = self.params

        beta_out_matrix = np.zeros((self.num_states, self.num_states))
        beta_in_matrix = np.zeros((self.num_states, self.num_states))

        beta_out: Dict[int, List[int]] = {}
        beta_in: Dict[int, List[int]] = {}
        for _, _, i, j, _ in valid_F_reactions:
            beta_out.setdefault(i, []).append(j)
            beta_in.setdefault(j, []).append(i)
            beta_out_matrix[i, j] = self.rng.gamma(shape, scale)
            beta_in_matrix[j, i] = self.rng.gamma(shape, scale)

        return beta_out_matrix, beta_in_matrix, beta_out, beta_in

    def k_parameter_generation(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.distribution != "gamma":
            raise NotImplementedError("Only 'gamma' distribution implemented for a_parameter_generation")
        shape, scale = self.params
        k_positive_rates = self.rng.gamma(shape, scale, self.num_states)
        k_negative_rates = self.rng.gamma(shape, scale, self.num_states)
        return k_positive_rates, k_negative_rates

    def p_parameter_generation(self) -> Tuple[np.ndarray, np.ndarray]:
        
        if self.distribution != "gamma":
            raise NotImplementedError("Only 'gamma' distribution implemented for b_parameter_generation")
        shape, scale = self.params
        p_positive_rates = self.rng.gamma(shape, scale, self.num_states)
        p_negative_rates = self.rng.gamma(shape, scale, self.num_states)
        return p_positive_rates, p_negative_rates


import numpy as np
from typing import List

def generate_phosphorylation_ODES(t, state_array, n, distribution_type, dist_par_1, dist_par_2, alpha_out_matrix, alpha_in_matrix, alpha_out, alpha_in,
                                  beta_out_matrix, beta_in_matrix, beta_out, beta_in, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates):
    
    # print()
    # print(k_positive_rates, k_negative_rates)
    # print(p_positive_rates, p_negative_rates)
    # print()
    a_array = state_array[0: 2**n] # A = S
    # b_array = state_array[2**n: 2**n + n] # B = XA = ES
    b_array = state_array[2**n: 2 * 2**n]
    c_array = state_array[2 * 2**n: 3 * 2**n]
    # c_array = state_array[2**n + n: 2**n + 2*n] # C = YA = FS

    x_array = state_array[-2] # X = E, kinase
    y_array = state_array[-1] # Y = F, phosphotase

    a_dot = np.zeros_like(a_array, dtype=float)
    b_dot = np.zeros_like(b_array, dtype=float)
    c_dot = np.zeros_like(c_array, dtype=float)

    for i in range(len(b_dot)):
        sum_b = np.sum(alpha_out_matrix[i, :])
        b_dot[i] = k_positive_rates[i] * x_array * a_array[i] - k_negative_rates[i] * b_array[i] - sum_b * b_array[i]

    for i in range(len(c_dot)):
        sum_c = np.sum(beta_out_matrix[i, :])
        c_dot[i] = p_positive_rates[i] * y_array * a_array[i] - p_negative_rates[i] * c_array[i] - sum_c * c_array[i]
    
    for i in range(len(a_dot)):
        E_term = 0
        E_term += k_negative_rates[i] * b_array[i]
        E_term -= k_positive_rates[i] * x_array * a_array[i]
        if 0 <= i < len(range(len(a_dot))):
            a_dot[i] = k_negative_rates[i] * b_array[i] - k_positive_rates[i] * x_array * a_array[i]
        else:
            E_term += 0

        F_term = 0
        if 0 < i <= len(range(len(a_dot))):
            F_term += p_negative_rates[i] * c_array[i]
            F_term -= p_positive_rates[i] * y_array * a_array[i]

        sum_cE = sum(alpha_in_matrix[k, i] * b_array[k] for k in alpha_in.get(i, [])) if i in alpha_in else 0
        sum_cF = sum(beta_in_matrix[k, i] * c_array[k] for k in beta_in.get(i, [])) if i in beta_in else 0

        a_dot[i] = E_term + F_term + sum_cE + sum_cF

    x_dot = sum(-k_positive_rates[j] * x_array * a_array[j] + (k_negative_rates[j] + np.sum(alpha_out_matrix[j, :])) * b_array[j] for j in range(len(b_array))); 

    y_dot = sum(-p_positive_rates[j] * y_array * a_array[j] + (p_negative_rates[j] + np.sum(beta_out_matrix[j, :])) * c_array[j] for j in range(len(c_array))); 

    # if (np.sum(a_dot) + np.sum(b_dot) + np.sum(c_dot)) <= 1e-5:
    #     print("Conservation of a is respected.")

    # if (np.sum(b_dot) + x_dot) <= 1e-5:
    #     print("Conservation of b is respected.")

    # if (np.sum(c_dot) + y_dot) <= 1e-5:
    #     print("Conservation of c is respected.")

    # print()
    # print("a_dot_array: ", a_dot)
    # print("b_dot_array: ", b_dot)
    # print("c_dot_array: ", c_dot)
    # print("x_dot_array: ", x_dot)
    # print("y_dot_array: ", y_dot)
    # print()

    
    final_state_array = np.concatenate((a_dot, b_dot, c_dot))
    return final_state_array