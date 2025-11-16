import numpy as np
from typing import List, Tuple, Dict, Any, Set
import random as rnd
from scipy.stats import levy
import random as random
from scipy.stats import reciprocal

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

def MATRIX_FINDER(n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates):

    N = 2**n
    ones_vec = np.ones(N - 1)

    Kp = np.diag(np.append(k_positive_rates, 0))
    Km = np.append(np.diag(k_negative_rates), np.zeros((1, len(k_negative_rates))), axis=0)
    assert Kp.shape == (N, N), f"Kp shape must be ({N}, {N})"
    assert Km.shape == (N, N-1), f"Km shape must be ({N}, {N-1})"
    
    Pp = np.diag(np.insert(p_positive_rates, 0, 0))
    Pm = np.vstack([np.zeros((1, len(p_negative_rates))), np.diag(p_negative_rates)])    # print("a", a)
    assert Pp.shape == (N, N), f"Pp shape must be ({N}, {N})"
    assert Pm.shape == (N, N-1), f"Pm shape must be ({N}, {N-1})"

    adjusted_alpha_mat = np.delete(alpha_matrix, -1, axis = 0)

    adjusted_beta_mat = np.delete(beta_matrix, 0, axis = 0)

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
    # lu, piv = lu_factor(M_mat)
    # M_inv = lu_solve((lu, piv), np.eye(M_mat.shape[0]))
    # lu, piv = lu_factor(N_mat)
    # N_inv = lu_solve((lu, piv), np.eye(N_mat.shape[0]))
    # M_inv = np.linalg.pinv(M_mat)
    # N_inv = np.linalg.pinv(N_mat)
    # M_inv = np.linalg.inv(M_mat); N_inv = np.linalg.inv(N_mat)

    # L1 = G @ M_inv @ Q - Kp; L2 = H @ N_inv @ D - Pp
    # assert L1.shape == (N, N), f"L1 shape must be ({N}, {N})"
    # assert L2.shape == (N, N), f"L2 shape must be ({N}, {N})"
    # W1 = M_inv @ Q; W2 = N_inv @ D
    # assert W1.shape == (N-1, N), f"W1 shape must be ({N-1}, {N})"
    # assert W2.shape == (N-1, N), f"W2 shape must be ({N-1}, {N})"

    # return Kp, Pp, G, H, Q, M_mat, D, N_mat, L1, L2, W1, W2

    return Kp, Pp, G, H, Q, M_mat, D, N_mat

def MATTIA_FULL(t, state_array, n, a_tot, x_tot, y_tot, Kp, Pp, G, H, Q, M_mat, D, N_mat):

    # including conservation laws
    # state_array = [a0, a1, b0, b1, c1, c2]

    # N = n + 1
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

    # x_dot = -1*np.sum(b_dot)
    # y_dot = -1*np.sum(c_dot)

    a_dot_red = a_dot[0: N-1]

    zero_threshold = 1e-15
    # below_mask = a_red < zero_threshold
    a_dot_red = np.where(a_red < zero_threshold, np.maximum(a_dot_red, 0), a_dot_red)
    b_dot = np.where(b < zero_threshold, np.maximum(b_dot, 0), b_dot)
    c_dot = np.where(c < zero_threshold, np.maximum(c_dot, 0), c_dot)

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
        self.N = 2**n
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
        num_states = self.N
        all_states = [self.padded_binary(i, self.n) for i in range(num_states)]

        valid_difference_vectors: Set[Tuple[int, ...]] = set()
        valid_X_reactions: List[List[Any]] = []
        valid_Y_reactions: List[List[Any]] = []

        for i in range(num_states):
            arr_i = self.binary_string_to_array(all_states[i])
            # print(all_states[i])
            for j in range(num_states):
                if i == j:
                    continue

                arr_j = self.binary_string_to_array(all_states[j])

                diff = arr_j - arr_i
                    
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
        
        valid_X_reactions, valid_Y_reactions = self.calculate_valid_transitions()

        shape, scale = self.params

        alpha_matrix = np.zeros((self.N, self.N))
        # alpha_array = np.array([self.rng.gamma(shape, scale) for i in range(self.n)])
        # alpha_matrix = np.diag(alpha_array, 1)
        
        # alpha_matrix = np.diag([self.rng.gamma(shape, scale)]*(min(self.N, self.N - 1)), 1)[:self.N, :self.N]
        for _, _, i, j, _ in valid_X_reactions:

            alpha_matrix[i][j] = self.rng.gamma(shape, scale)

        assert alpha_matrix.shape == (self.N, self.N), f"alpha_matrix shape must be ({self.N}, {self.N})"

        return alpha_matrix


    def beta_parameter_generation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                            Dict[int, List[int]], Dict[int, List[int]],
                                            Dict[int, List[int]], Dict[int, List[int]]]:
        
        valid_X_reactions, valid_Y_reactions = self.calculate_valid_transitions()

        shape, scale = self.params

        # beta_array = np.array([self.rng.gamma(shape, scale) for i in range(self.n)])
        # beta_matrix = np.diag(beta_array, -1)

        beta_matrix = np.zeros((self.N, self.N))
        
        for _, _, i, j, _ in valid_Y_reactions:

            beta_matrix[i][j] = self.rng.gamma(shape, scale)

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

    N = 2**n

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

def matrix_sample_reciprocal(matrix, rate_min, rate_max):

    sampled_matrix = matrix.copy()
    mask = sampled_matrix != 0
    num_samples = np.sum(mask)

    new_samples = reciprocal.rvs(
        a=rate_min, 
        b=rate_max, 
        size=num_samples
    )
    sampled_matrix[mask] = new_samples
    return sampled_matrix

def matrix_normalize(matrix: np.ndarray) -> np.ndarray:
    normalized = matrix.copy()
    non_zero_mask = normalized != 0
    non_zero_elements = normalized[non_zero_mask]
    mean_non_zero = non_zero_elements.mean()
    # normalized[non_zero_mask] = normalized[non_zero_mask] / mean_non_zero
    return normalized

def main():
    n = 2
    # n = 14
    # new_shape_parameters = (1e6, 1e6)
    # gen_rates = all_parameter_generation(n, "distributive", "gamma", new_shape_parameters, verbose = False)
    # alpha_matrix = gen_rates.alpha_parameter_generation()
    # print(alpha_matrix)


if __name__ == "__main__":
    main()