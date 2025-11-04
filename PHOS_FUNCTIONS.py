import numpy as np
from typing import List, Tuple, Dict, Any, Set
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
import pickle
import random as rnd
from scipy.stats import levy

def MATRIX_FINDER(n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates):

    N = n + 1
    ones_vec = np.ones(N - 1)
    # ones_vec = np.ones((1, N-1), dtype = float) # shape (1, N-1)

    Kp = np.diag(np.append(k_positive_rates, 0))
    Km = np.append(np.diag(k_negative_rates), np.zeros((1, len(k_negative_rates))), axis=0)
    assert Kp.shape == (N, N), f"Kp shape must be ({N}, {N})"
    assert Km.shape == (N, N-1), f"Km shape must be ({N}, {N-1})"
    # print(Kp)
    # print(Km)
    
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
    M_inv = np.linalg.inv(M_mat); N_inv = np.linalg.inv(N_mat)

    L1 = G @ M_inv @ Q - Kp; L2 = H @ N_inv @ D - Pp
    assert L1.shape == (N, N), f"L1 shape must be ({N}, {N})"
    assert L2.shape == (N, N), f"L2 shape must be ({N}, {N})"
    W1 = M_inv @ Q; W2 = N_inv @ D
    assert W1.shape == (N-1, N), f"W1 shape must be ({N-1}, {N})"
    assert W2.shape == (N-1, N), f"W2 shape must be ({N-1}, {N})"

    return Kp, Pp, G, H, Q, M_mat, D, N_mat, L1, L2, W1, W2

# def MATTIA_FULL(t, state_array, n, Kp, Pp, G, H, Q, M_mat, D, N_mat):

#     N = n + 1
#     # N = n**2
#     # ones_vec = np.ones(N - 1)
#     ones_vec = np.ones((1, N-1), dtype = float) # shape (1, N-1)

#     # assert len(state_array) == 3*N
#     # print(state_array)
#     a = state_array[0: N]
#     b = state_array[N: 2*N - 1]
#     c = state_array[2*N - 1: 3*N - 2]
#     x = float(state_array[-2])
#     y = float(state_array[-1])

#     a_dot = (G @ b) + (H @ c) - x * (Kp @ a) - y * (Pp @ a)  
#     b_dot = x * (Q @ a) - (M_mat @ b)
#     c_dot = y * (D @ a) - (N_mat @ c)

#     x_dot = -1*ones_vec.T @ b_dot
#     y_dot = -1*ones_vec.T @ c_dot

#     return np.concatenate((a_dot, b_dot, c_dot, np.array([x_dot, y_dot])))

def MATTIA_REDUCED(t, state_array, n, x_tot, y_tot, L1, L2, W1, W2):
    N = state_array.size
    ones_vec = np.ones(W1.shape[0])   # length N-1

    a = np.asarray(state_array, dtype=float).ravel()
    assert len(a) == N

    # Compute denominator scalars
    denom1 = 1 + ones_vec @ (W1 @ a)
    denom2 = 1 + ones_vec @ (W2 @ a)
    
    # Compute a_dot as 1D array
    a_dot = ((x_tot * (L1 @ a)) / denom1) + ((y_tot * (L2 @ a)) / denom2)

    return a_dot

def MATTIA_REDUCED_JACOBIAN(t, state_array, n, x_tot, y_tot, L1, L2, W1, W2):
    N = state_array.size
    state_array = np.asarray(state_array, dtype=float).ravel()
    assert state_array.size == N, f"expected state length {N}, got {state_array.size}"

    ones_vec_j = np.ones(W1.shape[0])  # length N-1

    onesW1 = (ones_vec_j.T @ W1).ravel()  # shape (N,)
    onesW2 = (ones_vec_j.T @ W2).ravel()  # shape (N,)

    L1a = L1 @ state_array  # shape (N,)
    L2a = L2 @ state_array  # shape (N,)

    # Compute denominators as scalars
    denom1 = 1 + ones_vec_j @ (W1 @ state_array)
    denom2 = 1 + ones_vec_j @ (W2 @ state_array)

    # Use outer product to get (N, N) matrices
    term1 = (L1 / denom1) - (np.outer(L1a, onesW1) / (denom1**2))   # (N,N)
    term2 = (L2 / denom2) - (np.outer(L2a, onesW2) / (denom2**2))   # (N,N)

    J = (x_tot * term1) + (y_tot * term2)

    return J

def stability_calculator(a_fixed_points, x_tot, y_tot, L1, L2, W1, W2):
    N = a_fixed_points.size
    state_array = np.asarray(a_fixed_points, dtype=float).ravel()
    assert state_array.size == N, f"expected state length {N}, got {state_array.size}"

    ones_vec_j = np.ones(W1.shape[0])  # 1D array of length N-1

    onesW1 = ones_vec_j @ W1  # shape (N,)
    onesW2 = ones_vec_j @ W2  # shape (N,)

    L1a = L1 @ state_array  # shape (N,)
    L2a = L2 @ state_array  # shape (N,)

    # Compute denominators as scalars
    denom1 = 1 + ones_vec_j @ (W1 @ state_array)
    denom2 = 1 + ones_vec_j @ (W2 @ state_array)

    # Use outer product to get (N, N) matrices
    term1 = (L1 / denom1) - (np.outer(L1a, onesW1) / (denom1**2))   # (N,N)
    term2 = (L2 / denom2) - (np.outer(L2a, onesW2) / (denom2**2))   # (N,N)

    J = (x_tot * term1) + (y_tot * term2)

    return J

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
    
def ALL_GUESSES(N, guesses):

    rng = np.random.default_rng()

    # 1. Very sharp corners: one component dominates (alpha << 1)
    for _ in range(N):
        alpha_sharp = np.full(N, 0.1)  # Small alpha pushes to corners
        guess = rng.dirichlet(alpha_sharp)
        guesses.append(guess)  # Remove last component for reduced system
    
    # 2. Edges: two components share mass (set others to very small alpha)
    edge_samples_per_pair = 2
    for i in range(N):
        for j in range(i+1, N):
            for _ in range(edge_samples_per_pair):
                alpha_edge = np.full(N, 0.05)  # Very small for components not on edge
                alpha_edge[i] = 1.0  # Moderate for edge components
                alpha_edge[j] = 1.0
                guess = rng.dirichlet(alpha_edge)
                guesses.append(guess)
    
    # 3. Faces: k components share mass equally
    for n_active in range(2, N+1):
        for _ in range(2):  # 2 samples per face
            alpha_face = np.full(N, 0.05)
            alpha_face[:n_active] = 1.0
            guess = rng.dirichlet(alpha_face)
            guesses.append(guess)
    
    # 4. Uniform-ish distribution (alpha = 1 is uniform)
    for _ in range(3):
        guess = rng.dirichlet(np.ones(N))
        guesses.append(guess)
    
    # 5. Slightly perturbed corners (exact corners)
    for i in range(N):
        corner = np.zeros(N)
        corner[i] = 1.0
        guesses.append(corner)

    return guesses

def duplicate(candidate, collection, euclidian_distance_tol):
    """Check if candidate is close to any vector in collection"""
    if len(collection) == 0:
        return False
    
    for v in collection:
        if (np.linalg.norm(np.array(v) - np.array(candidate)) < euclidian_distance_tol) and np.allclose(v, candidate):
            return True
    return False

def matrix_clip(matrix):
    rate_min, rate_max = 1e-1, 1e7
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
    gen_rates = all_parameter_generation(n, "distributive", "gamma", (0.123, 4.46e6), verbose = False)
    rate_min, rate_max = 1e-1, 1e7
    alpha_matrix = gen_rates.alpha_parameter_generation()
    beta_matrix = gen_rates.beta_parameter_generation()
    k_positive_rates = gen_rates.k_parameter_generation()[0]
    k_negative_rates = gen_rates.k_parameter_generation()[1]
    p_positive_rates = gen_rates.p_parameter_generation()[0]
    p_negative_rates = gen_rates.p_parameter_generation()[1]
    Kp, Pp, G, H, Q, M_mat, D, N_mat, L1, L2, W1, W2 = MATRIX_FINDER(n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates)

    rate_min, rate_max = 1e-1, 1e7

    k_positive_rates = k_positive_rates / np.mean(k_positive_rates)
    p_positive_rates = p_positive_rates / np.mean(p_positive_rates)
    k_positive_rates = np.clip(k_positive_rates, rate_min, rate_max)
    p_positive_rates = np.clip(p_positive_rates, rate_min, rate_max)
    k_negative_rates = k_positive_rates
    p_negative_rates = p_positive_rates

    # normalize
    alpha_matrix = alpha_matrix / np.mean(alpha_matrix)
    beta_matrix = beta_matrix / np.mean(beta_matrix)

    # clipping
    alpha_matrix = matrix_clip(alpha_matrix); beta_matrix = matrix_clip(beta_matrix)
    # print(alpha_matrix)
    # print(beta_matrix)
    # print(k_positive_rates)
    # print(k_negative_rates)
    # print(p_positive_rates)
    # print(p_negative_rates)

    test_array = np.array([0.2, 0.5, 0.3]); pertubation_parameter = 0.1
    print(f"initial state array: {test_array}")
    pertubation_array = pertubation_array_creation(test_array, pertubation_parameter)
    print(f"perturbed initial state: {test_array + pertubation_array}")

if __name__ == "__main__":
    main()