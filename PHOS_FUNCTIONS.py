import numpy as np
from typing import List, Tuple, Dict, Any, Set
import random as rnd
from scipy.stats import levy
import random as random

def MATRIX_FINDER(n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates):

    N = n + 1
    ones_vec = np.ones(N - 1)
    # ones_vec = np.ones((1, N - 1))
    # ones_vec = np.ones((1, N-1), dtype = float) # shape (1, N-1)

    Kp = np.diag(np.append(k_positive_rates, 0))
    Km = np.append(np.diag(k_negative_rates), np.zeros((1, len(k_negative_rates))), axis=0)
    assert Kp.shape == (N, N), f"Kp shape must be ({N}, {N})"
    assert Km.shape == (N, N-1), f"Km shape must be ({N}, {N-1})"
    
    Pp = np.diag(np.insert(p_positive_rates, 0, 0))
    Pm = np.vstack([np.zeros((1, len(p_negative_rates))), np.diag(p_negative_rates)])    # print("a", a)
    assert Pp.shape == (N, N), f"Pp shape must be ({N}, {N})"
    assert Pm.shape == (N, N-1), f"Pm shape must be ({N}, {N-1})"
    # print("Kp")
    # print(np.asarray(Kp, dtype = float))
    # print("Pp")
    # print(Pp)
    adjusted_alpha_mat = np.delete(alpha_matrix, -1, axis = 0)
    # print("adjusted_alpha_mat.T")
    # print(adjusted_alpha_mat.T)
    adjusted_beta_mat = np.delete(beta_matrix, 0, axis = 0)
    # print("adjusted_beta_mat.T")
    # print(adjusted_beta_mat.T)
    assert adjusted_alpha_mat.shape == (N-1, N), f"adjusted_alpha_mat shape must be ({N-1}, {N})"
    assert adjusted_beta_mat.shape == (N-1, N), f"adjusted_beta_mat shape must be ({N-1}, {N})"

    Da = np.diag(alpha_matrix[:-1, 1:] @ ones_vec)
    Db = np.diag(beta_matrix[1:, :-1] @ ones_vec)
    assert Da.shape == (N-1, N-1), f"Da shape must be ({N-1}, {N-1})"
    assert Db.shape == (N-1, N-1), f"Db shape must be ({N-1}, {N-1})"

    U = np.diag(k_negative_rates)
    I = np.diag(p_negative_rates)
    assert U.shape == (N-1, N-1), f"U shape must be ({N-1}, {N-1})"
    assert I.shape == (N-1, N-1), f"I shape must be ({N-1}, {N-1})"
    
    Q = Kp[:-1, :]
    D = np.delete(Pp, 0, axis=0)
    assert Q.shape == (N-1, N), f"Q shape must be ({N-1}, {N})"
    assert D.shape == (N-1, N), f"D shape must be ({N-1}, {N})"

    M_mat = U + Da
    N_mat = I + Db

    G = Km + adjusted_alpha_mat.T

    H = Pm + adjusted_beta_mat.T

    assert G.shape == (N, N-1), f"G shape must be ({N}, {N-1})"
    assert H.shape == (N, N-1), f"H shape must be ({N}, {N-1})"
    M_inv = np.linalg.inv(M_mat); N_inv = np.linalg.inv(N_mat)

    L1 = G @ M_inv @ Q - Kp; L2 = H @ N_inv @ D - Pp
    assert L1.shape == (N, N), f"L1 shape must be ({N}, {N})"
    assert L2.shape == (N, N), f"L2 shape must be ({N}, {N})"
    W1 = M_inv @ Q; W2 = N_inv @ D
    assert W1.shape == (N-1, N), f"W1 shape must be ({N-1}, {N})"
    assert W2.shape == (N-1, N), f"W2 shape must be ({N-1}, {N})"

    return Kp, Pp, G, H, Q, M_mat, D, N_mat, L1, L2, W1, W2

def MATTIA_FULL(t, state_array, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat):

    # including conservation laws
    # state_array = [a0, a1, b0, b1, c1, c2]

    N = n + 1
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

    x_dot = -1*np.sum(b_dot)
    y_dot = -1*np.sum(c_dot)

    a_dot_red = a_dot[0: N-1]

    return np.concatenate([a_dot_red, b_dot, c_dot])

class all_parameter_generation:
    """
    Generate state transitions and random parameters (a, b, c, enzyme) for an n-site phosphorylation model.

    Args:
        n: number of sites (int)
        distribution: distribution name ("gamma" supported)
        params: parameters for the distribution (for gamma: [shape, scale])
        verbose: if True, prints transitions and matrices
    """
    def __init__(self, n: int, reaction_types: str, distribution: str, distribution_paramaters: List[float], verbose: bool = False):
        self.n = n
        self.N = n + 1
        self.distribution = distribution
        self.params = distribution_paramaters
        self.reaction_types = reaction_types
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
            valid_X_reactions: list of [state_i_str, state_j_str, i, j, "E"]
            valid_Y_reactions: list of [state_i_str, state_j_str, i, j, "F"]
        """
        all_states = [self.padded_binary(i, self.n) for i in range(self.N)]

        valid_difference_vectors: Set[Tuple[int, ...]] = set()
        valid_X_reactions: List[List[Any]] = []
        valid_Y_reactions: List[List[Any]] = []

        for i in range(self.N):
            arr_i = self.binary_string_to_array(all_states[i])
            for j in range(self.N):
                if i == j:
                    continue
                arr_j = self.binary_string_to_array(all_states[j])
                diff = arr_j - arr_i
                # if self.reaction_types == "distributive":
                    
                hamming_weight = np.sum(np.abs(diff))

                if hamming_weight == 1:
                    # +1 -> phosphorylation (E), -1 -> dephosphorylation (F)
                    element = "E" if np.any(diff == 1) else "F"
                    if element == "E":
                        if self.verbose:
                            print(f"{all_states[i]} --> {all_states[j]} (E), {i}, {j}")
                        valid_X_reactions.append([all_states[i], all_states[j], i, j, element])
                    else:
                        if self.verbose:
                            print(f"{all_states[i]} --> {all_states[j]} (F), {i}, {j}")
                        valid_Y_reactions.append([all_states[i], all_states[j], i, j, element])
                    valid_difference_vectors.add(tuple(diff))

        return valid_X_reactions, valid_Y_reactions
    
    def alpha_parameter_generation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                            Dict[int, List[int]], Dict[int, List[int]],
                                            Dict[int, List[int]], Dict[int, List[int]]]:
        
        # valid_X_reactions, valid_Y_reactions = self.calculate_valid_transitions()

        shape, scale = self.params

        # alpha_matrix = np.zeros((self.N, self.N))
        alpha_array = np.array([self.rng.gamma(shape, scale) for i in range(self.n)])
        alpha_matrix = np.diag(alpha_array, 1)
        
        # alpha_matrix = np.diag([self.rng.gamma(shape, scale)]*(min(self.N, self.N - 1)), 1)[:self.N, :self.N]
        # for _, _, i, j, _ in valid_X_reactions:

        #     alpha_matrix[i][j] = self.rng.gamma(shape, scale)

        assert alpha_matrix.shape == (self.N, self.N), f"alpha_matrix shape must be ({self.N}, {self.N})"

        return alpha_matrix


    def beta_parameter_generation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                            Dict[int, List[int]], Dict[int, List[int]],
                                            Dict[int, List[int]], Dict[int, List[int]]]:
        
        # valid_X_reactions, valid_Y_reactions = self.calculate_valid_transitions()

        shape, scale = self.params
        beta_array = np.array([self.rng.gamma(shape, scale) for i in range(self.n)])
        
        beta_matrix = np.diag(beta_array, -1)
        # beta_matrix = np.zeros((self.N, self.N))
        
        # for _, _, i, j, _ in valid_Y_reactions:

        #     beta_matrix[i][j] = self.rng.gamma(shape, scale)

        # beta_matrix = np.diag([self.rng.gamma(shape, scale)]*(min(self.N-1, self.N)), -1)[:self.N, :self.N]
        assert beta_matrix.shape == (self.N, self.N), f"alpha_matrix shape must be ({self.N}, {self.N})"

        return beta_matrix
    
    def k_parameter_generation(self) -> Tuple[np.ndarray, np.ndarray]:
        # if self.distribution != "gamma":
        #     raise NotImplementedError("Only 'gamma' distribution implemented for a_parameter_generation")
        shape, scale = self.params
        if self.distribution == "gamma":
            k_positive_rates = self.rng.gamma(shape, scale, self.N - 1)
            k_negative_rates = self.rng.gamma(shape, scale, self.N - 1)
        if self.distribution == "levy":
            k_positive_rates = levy.rvs(loc=shape, scale=scale, size=self.N - 1, random_state=self.rng)
            k_negative_rates = levy.rvs(loc=shape, scale=scale, size=self.N - 1, random_state=self.rng)

        return k_positive_rates, k_negative_rates

    def p_parameter_generation(self) -> Tuple[np.ndarray, np.ndarray]:
        
        # if self.distribution != "gamma":
        #     raise NotImplementedError("Only 'gamma' distribution implemented for b_parameter_generation")
        shape, scale = self.params
        if self.distribution == "gamma":
            p_positive_rates = self.rng.gamma(shape, scale, self.N - 1)
            p_negative_rates = self.rng.gamma(shape, scale, self.N - 1)
        if self.distribution == "levy":
            p_positive_rates = levy.rvs(loc=shape, scale=scale, size=self.N - 1, random_state=self.rng)
            p_negative_rates = levy.rvs(loc=shape, scale=scale, size=self.N - 1, random_state=self.rng)

        return p_positive_rates, p_negative_rates
    
    def total_concentration_generation(self) -> Tuple[np.ndarray, np.ndarray]:
        
        # if self.distribution != "gamma":
        #     raise NotImplementedError("Only 'gamma' distribution implemented for b_parameter_generation")
        shape, scale = self.params
        if self.distribution == "gamma":
            x_tot_concentration = self.rng.gamma(shape, scale, 1)
            y_tot_concentration = self.rng.gamma(shape, scale, 1)

        return x_tot_concentration, y_tot_concentration


def guess_generator(n, a_tot, x_tot, y_tot):
    N = n + 1

    bx_splits = sorted([random.uniform(0, x_tot) for _ in range(N - 1)])
    bx_values = [bx_splits[0]] + [bx_splits[i] - bx_splits[i-1] for i in range(1, N-1)] + [x_tot - bx_splits[-1]]
    b_guess = bx_values[0:N-1]
    x_guess = bx_values[-1]
    # print(b_guess)
    # print(x_guess)

    cy_splits = sorted([random.uniform(0, y_tot) for _ in range(N - 1)])
    cy_values = [cy_splits[0]] + [cy_splits[i] - cy_splits[i-1] for i in range(1, N-1)] + [x_tot - cy_splits[-1]]
    c_guess = cy_values[0:N-1]
    y_guess = cy_values[-1]
    # print(c_guess)
    # print(y_guess)

    total = a_tot - sum(b_guess) - sum(c_guess)
    a_splits = sorted([random.uniform(0, total) for _ in range(N - 1)])
    a_values = [a_splits[0]] + [a_splits[i] - a_splits[i-1] for i in range(1, N-1)] + [total - a_splits[-1]]
    # print(a_values)
    a_guess = a_values[0:N-1]
    # print(a_guess)
    a_guess.extend(b_guess)
    a_guess.extend(c_guess)
    guess_list = a_guess
    return guess_list

def duplicate(candidate, collection, euclidian_distance_tol):
    """Check if candidate is close to any vector in collection"""
    if len(collection) == 0:
        return False
    
    for v in collection:
        # if (np.linalg.norm(np.array(v) - np.array(candidate)) < euclidian_distance_tol):
        if (np.linalg.norm(np.array(v) - np.array(candidate)) < euclidian_distance_tol) and (np.allclose(v, candidate, atol = 1e-10)):
            return True
    return False

def matrix_clip(matrix, rate_min, rate_max):

    clipped = matrix.copy()
    mask = clipped != 0
    clipped[mask] = np.clip(clipped[mask], rate_min, rate_max)
    return clipped

def pertubation_array_creation(ic_array, pertubation_parameter):
    # find the 2 indices with the largest elements, create a pertubation array
    largest_index = np.argsort(ic_array)[-2:][-2]
    second_largest_index = np.argsort(ic_array)[-2:][-1]
    if rnd.random() > 0.5:
        return np.array([-pertubation_parameter if i == largest_index else pertubation_parameter if i == second_largest_index else 0 for i in range(len(ic_array))])
    else: 
        return np.array([pertubation_parameter if i == largest_index else -pertubation_parameter if i == second_largest_index else 0 for i in range(len(ic_array))])

def main():
    n = 2


if __name__ == "__main__":
    main()