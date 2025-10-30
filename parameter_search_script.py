import numpy as np
import sympy as sp
from typing import List, Tuple, Dict, Any, Set
from scipy.stats import levy
import matplotlib.pyplot as plt
from scipy.optimize import root
from sympy import shape
from sympy import Trace
from joblib import Parallel, delayed
import pickle
from numpy import linalg as LA
import numpy as np
import sympy as sp
from scipy.optimize import root
from sympy import shape

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
        self.num_states = n + 1
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
        all_states = [self.padded_binary(i, self.n) for i in range(self.num_states)]

        valid_difference_vectors: Set[Tuple[int, ...]] = set()
        valid_X_reactions: List[List[Any]] = []
        valid_Y_reactions: List[List[Any]] = []

        for i in range(self.num_states):
            arr_i = self.binary_string_to_array(all_states[i])
            for j in range(self.num_states):
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

        # alpha_matrix = np.zeros((self.num_states, self.num_states))
        alpha_array = np.array([self.rng.gamma(shape, scale) for i in range(self.n)])
        alpha_matrix = np.diag(alpha_array, 1)
        
        # alpha_matrix = np.diag([self.rng.gamma(shape, scale)]*(min(self.num_states, self.num_states - 1)), 1)[:self.num_states, :self.num_states]
        # for _, _, i, j, _ in valid_X_reactions:

        #     alpha_matrix[i][j] = self.rng.gamma(shape, scale)

        return alpha_matrix


    def beta_parameter_generation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                            Dict[int, List[int]], Dict[int, List[int]],
                                            Dict[int, List[int]], Dict[int, List[int]]]:
        
        # valid_X_reactions, valid_Y_reactions = self.calculate_valid_transitions()

        shape, scale = self.params
        beta_array = np.array([self.rng.gamma(shape, scale) for i in range(self.n)])
        
        beta_matrix = np.diag(beta_array, -1)
        # beta_matrix = np.zeros((self.num_states, self.num_states))
        
        # for _, _, i, j, _ in valid_Y_reactions:

        #     beta_matrix[i][j] = self.rng.gamma(shape, scale)

        # beta_matrix = np.diag([self.rng.gamma(shape, scale)]*(min(self.num_states-1, self.num_states)), -1)[:self.num_states, :self.num_states]
        
        return beta_matrix
    
    def k_parameter_generation(self) -> Tuple[np.ndarray, np.ndarray]:
        # if self.distribution != "gamma":
        #     raise NotImplementedError("Only 'gamma' distribution implemented for a_parameter_generation")
        shape, scale = self.params
        if self.distribution == "gamma":
            k_positive_rates = self.rng.gamma(shape, scale, self.num_states - 1)
            k_negative_rates = self.rng.gamma(shape, scale, self.num_states - 1)
        if self.distribution == "levy":
            k_positive_rates = levy.rvs(loc=shape, scale=scale, size=self.num_states - 1, random_state=self.rng)
            k_negative_rates = levy.rvs(loc=shape, scale=scale, size=self.num_states - 1, random_state=self.rng)
        
        return k_positive_rates, k_negative_rates

    def p_parameter_generation(self) -> Tuple[np.ndarray, np.ndarray]:
        
        # if self.distribution != "gamma":
        #     raise NotImplementedError("Only 'gamma' distribution implemented for b_parameter_generation")
        shape, scale = self.params
        if self.distribution == "gamma":
            p_positive_rates = self.rng.gamma(shape, scale, self.num_states - 1)
            p_negative_rates = self.rng.gamma(shape, scale, self.num_states - 1)
        if self.distribution == "levy":
            p_positive_rates = levy.rvs(loc=shape, scale=scale, size=self.num_states - 1, random_state=self.rng)
            p_negative_rates = levy.rvs(loc=shape, scale=scale, size=self.num_states - 1, random_state=self.rng)

        return p_positive_rates, p_negative_rates
    
    def total_concentration_generation(self) -> Tuple[np.ndarray, np.ndarray]:
        
        # if self.distribution != "gamma":
        #     raise NotImplementedError("Only 'gamma' distribution implemented for b_parameter_generation")
        shape, scale = self.params
        if self.distribution == "gamma":
            x_tot_concentration = self.rng.gamma(shape, scale, 1)
            y_tot_concentration = self.rng.gamma(shape, scale, 1)

        return x_tot_concentration, y_tot_concentration

def polynomial_finder(n, alpha_matrix, beta_matrix, k_positive_rates, k_negative_rates, p_positive_rates, p_negative_rates):

    N = n + 1
    
    # x_tot = sp.Float(x_tot_value); y_tot = sp.Float(y_tot_value); a_tot = sp.Float(a_tot_value)

    ones_vec = np.ones(N - 1)
    # ones_vec = np.ones((N-1, 1))
    Kp = np.diag(np.append(k_positive_rates, 0))
    Km = np.append(np.diag(k_negative_rates), np.zeros((1, len(k_negative_rates))), axis=0)
    assert Kp.shape == (N, N), f"Kp shape must be ({N}, {N})"
    assert Km.shape == (N, N-1), f"Km shape must be ({N}, {N-1})"

    Pp = np.diag(np.insert(p_positive_rates, 0, 0))
    Pm = np.vstack([np.zeros((1, len(p_negative_rates))), np.diag(p_negative_rates)])
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

    return L1, L2, W1, W2

def stability_calculator(a_fixed_points, p, L1, L2, W1, W2):

    a_fixed_points = np.asarray(a_fixed_points, dtype=float).ravel()
    N = a_fixed_points.size

    ones_vec_j = np.ones((1, N-1), dtype = float) # shape (1, N-1)

    a_fixed_points = a_fixed_points.reshape((N, 1))  # shape (N, 1)
    L1 = np.array(L1, dtype=float)
    L2 = np.array(L2, dtype=float)
    W1 = np.array(W1, dtype=float)
    W2 = np.array(W2, dtype=float)

    # Compute denominators
    denom1 = float(1 + np.dot(ones_vec_j, (W1 @ a_fixed_points)).item())
    denom2 = float(1 + np.dot(ones_vec_j, (W2 @ a_fixed_points)).item())

    term1 = (L1 / denom1) - (((L1 @ a_fixed_points) @ (ones_vec_j @ W1)) / (denom1**2))
    term2 = (L2 / denom2) - (((L2 @ a_fixed_points) @ (ones_vec_j @ W2)) / (denom2**2))

    J = (p * term1) + term2
    return J

def fp_checker(sites_n, a_tot_value, x_tot_value, y_tot_value,
                      alpha_matrix, beta_matrix,
                      k_positive_rates, k_negative_rates,
                      p_positive_rates, p_negative_rates):
    
    N = sites_n + 1

    possible_steady_states = np.floor((sites_n + 2) / 2).astype(int)
    
    a_syms_full = sp.symbols([f"a{i}" for i in range(N)], real=True)
    a_syms_reduced = sp.symbols([f"a{i}" for i in range(N - 1)], real=True)
    a_tot_sym, x_tot_sym, y_tot_sym = sp.symbols("a_tot_sym x_tot_sym y_tot_sym", real=True)
    conservation_expression = 1 - sum(a_syms_reduced)
    a_vec_sym = sp.Matrix([sp.symbols([f"a{i}" for i in range(N - 1)], real=True)[i] if i < N-1 else conservation_expression for i in range(N)])

    L1, L2, W1, W2 = polynomial_finder(
        sites_n, alpha_matrix, beta_matrix,
        k_positive_rates, k_negative_rates,
        p_positive_rates, p_negative_rates
    )

    L1_sym = sp.Matrix(L1.tolist())
    L2_sym = sp.Matrix(L2.tolist())
    W1_sym = sp.Matrix(W1.tolist())
    W2_sym = sp.Matrix(W2.tolist())
    ones_vec_sym = sp.Matrix([[1] * W2_sym.rows])

    inner_W1 = (ones_vec_sym * W1_sym * a_vec_sym)[0, 0]
    inner_W2 = (ones_vec_sym * W2_sym * a_vec_sym)[0, 0]

    L1a = L1_sym * a_vec_sym
    L2a = L2_sym * a_vec_sym
    p = x_tot_sym / y_tot_sym
    poly_exprs = p * L1a * (1 + inner_W2) + L2a * (1 + inner_W1)   

    subs_numeric = {a_tot_sym: float(a_tot_value), x_tot_sym: float(x_tot_value), y_tot_sym: float(y_tot_value)}

    # plugging in numeric values
    polynomials_list_numeric = sp.Matrix([sp.N(p.subs(subs_numeric)) for p in poly_exprs]) # all 4 polynomials are included, will be evaluated at identified roots
    polynomials_list_reduced_numeric = [sp.simplify(polynomials_list_numeric[i, 0]) for i in range(N - 1)] # range(N - 1) means we only have 3 polynomials now

    jacobian = polynomials_list_numeric.jacobian(a_syms_full)

    divergence = jacobian.trace()

    divergence_subs_numeric = {a_syms_full[0]: float(0.1), a_syms_full[1]: float(0.1)}

    divergence_evaluated = divergence.evalf(subs=divergence_subs_numeric)
    print("divergence_evaluated:", divergence_evaluated)

    # jacobian_reduced = [sp.simplify(jacobian[i, 0]) for i in range(N - 1)]
    jacobian_reduced_lambdified = sp.lambdify(list(a_syms_reduced), jacobian, "numpy")
    ####
    # lambdifying functions
    polynomials_reduced_lambdified = sp.lambdify(list(a_syms_reduced), polynomials_list_reduced_numeric, "numpy")
    polynomials_lambdified = sp.lambdify(list(a_syms_full), polynomials_list_numeric, "numpy")
    
    # def jacobian_func(a_vec):
    #     a_vec = np.asarray(a_vec, dtype=float).ravel()             
    #     vals = jacobian_reduced_lambdified(*a_vec)                
    #     return np.asarray(vals, dtype=float).ravel()
    
    def residuals_vec_red(a_vec):
        a_vec = np.asarray(a_vec, dtype=float).ravel()             
        vals = polynomials_reduced_lambdified(*a_vec)                
        return np.asarray(vals, dtype=float).ravel()

    def residuals_vec(a_vec):
        a_vec = np.asarray(a_vec, dtype=float).ravel()              
        vals = polynomials_lambdified(*a_vec)            
        return np.asarray(vals, dtype=float).ravel()

    def duplicate(candidate, collection, norm_tol):
        for v in collection:
            norm_close = False
            euclidian_distance = np.linalg.norm(np.array(v) - np.array(candidate))
            if euclidian_distance < norm_tol:
                norm_close = True

            all_close = np.allclose(v, candidate)
            if (all_close and norm_close):
                True

            # if norm_close:
            #     return True
        return False

    final_sol_list_stable = []
    final_sol_list_unstable = []

    attempt_total_num = 10 # should be proportional to the number of sites/dimensions?
    guesses = []
    for u in range(0, attempt_total_num):
        rand_guess = np.random.rand(N-1)
        guesses.append(rand_guess / np.sum(rand_guess))

    for guess in guesses:
        print(guess)
        root_finder_tol = 1e-8
        try:
            sol = root(residuals_vec_red, guess, method = 'hybr', tol = root_finder_tol)
            if not sol.success:
                continue
            
        except Exception as e:
            continue
        sol = np.asarray(sol.x, dtype=float).ravel()

        full_sol = np.append(sol, 1-np.sum(sol))

        clipping_tolerance = 1e-8
        if np.any((full_sol < -clipping_tolerance) | (full_sol > 1+clipping_tolerance)):
            continue

        full_sol = np.clip(full_sol, 0-clipping_tolerance, 1+clipping_tolerance)

        if (np.any(residuals_vec(full_sol)) < 1e-6):
            print("Residual tolerance not satisfied")
            continue

        J = stability_calculator(full_sol, x_tot_value / y_tot_value, L1, L2, W1, W2)
        # if not np.all(np.isfinite(J)) or np.isnan(J).any():
        if not np.isfinite(J).all():
            continue  # skip this guess if J contains NaN or inf

        eigenvalues = LA.eigvals(J)
        eigenvalues_real = np.real(eigenvalues)  

        # norm_tol = root_finder_tol 
        norm_tol = 1e-4

        if np.any(eigenvalues_real > 0):
            if duplicate(full_sol, final_sol_list_unstable, norm_tol) == False:
                final_sol_list_unstable.append(full_sol)
        
        elif np.all(eigenvalues_real < 0):
            if duplicate(full_sol, final_sol_list_stable, norm_tol) == False:
                final_sol_list_stable.append(full_sol) 

        else:
            continue     

    print(f"# of stable points found is {len(final_sol_list_stable)}, and # of unsteady states found is {len(final_sol_list_unstable)}")

    if (len(final_sol_list_stable) == 0) and (len(final_sol_list_unstable) == 0):
        # print("Found no solutions.")
        return np.array([]), np.array([])  # failed
    return np.array(final_sol_list_stable), np.array(final_sol_list_unstable) # an array of arrays (a matrix)

def process_sample(i,
                   sites_n,
                   a_tot_value,
                   x_tot_value_parameter_array,
                   y_tot_value_parameter_array,
                   alpha_matrix_parameter_array,
                   beta_matrix_parameter_array,
                   k_positive_parameter_array,
                   k_negative_parameter_array,
                   p_positive_parameter_array,
                   p_negative_parameter_array):
    """
    Run the per-sample work that used to be inside the loop.
    Returns a tuple (mon_row_or_None, bist_row_or_None).
    """
    N = sites_n + 1

    k_positive_rates = k_positive_parameter_array[i]
    k_negative_rates = k_negative_parameter_array[i]
    p_positive_rates = p_positive_parameter_array[i]
    p_negative_rates = p_negative_parameter_array[i]
    x_tot_value = x_tot_value_parameter_array[i][0]
    y_tot_value = y_tot_value_parameter_array[i][0]
    alpha_matrix = alpha_matrix_parameter_array[i]
    beta_matrix = beta_matrix_parameter_array[i]
    stable_fp_array, unstable_fp_array = fp_checker(sites_n, a_tot_value, x_tot_value, y_tot_value,
                                                     alpha_matrix, beta_matrix, k_positive_rates,
                                                     k_negative_rates, p_positive_rates, p_negative_rates)
    print("alpha_matrix:")
    print(alpha_matrix)

    sim_num = np.array([i]).astype(int)

    monostable_results = None
    multistable_results = None
    
    possible_steady_states = np.floor((sites_n + 2) / 2).astype(int)
    possible_unsteady_states = possible_steady_states + 1

    if len(stable_fp_array) != 0 or len(unstable_fp_array) != 0:
        monostable_results = {
        "num_of_stable_states": np.array([len(stable_fp_array)]),
        "stable_states": stable_fp_array,
        "total_concentration_values": np.array([a_tot_value, x_tot_value, y_tot_value]),
        "alpha_matrix": alpha_matrix_parameter_array[i],
        "beta_matrix": beta_matrix_parameter_array[i],
        "k_positive": k_positive_parameter_array[i],
        "k_negative": k_negative_parameter_array[i],
        "p_positive": p_positive_parameter_array[i],
        "p_negative": p_negative_parameter_array[i]
        }

    # multistability condition
    # if len(stable_fp_array) == 2 and len(unstable_fp_array) == 2:
    if len(stable_fp_array) == 2 or len(unstable_fp_array) == 1:
        multistable_results = {
        "num_of_stable_states": len(stable_fp_array),
        "num_of_unstable_states": len(unstable_fp_array),
        "stable_states": np.array([stable_fp_array[k] for k in range(len(stable_fp_array))]),
        "unstable_states": np.array([unstable_fp_array[k] for k in range(len(unstable_fp_array))]),
        "total_concentration_values": np.array([a_tot_value, x_tot_value, y_tot_value]),
        "alpha_matrix": alpha_matrix_parameter_array[i],
        "beta_matrix": beta_matrix_parameter_array[i],
        "k_positive": k_positive_parameter_array[i],
        "k_negative": k_negative_parameter_array[i],
        "p_positive": p_positive_parameter_array[i],
        "p_negative": p_negative_parameter_array[i]
        }

    return monostable_results, multistable_results

    # return mon_row, multist_row

def simulation(sites_n, simulation_size):
    a_tot_value = 1
    N = sites_n + 1

    gen_rates = all_parameter_generation(sites_n, "distributive", "gamma", (0.123, 4.46e6), verbose = False)
    rate_min, rate_max = 1e-1, 1e7
    alpha_matrix_parameter_array = np.array([gen_rates.alpha_parameter_generation() for _ in range(simulation_size)]) # CANNOT CLIP
    beta_matrix_parameter_array = np.array([gen_rates.beta_parameter_generation() for _ in range(simulation_size)]) # CANNOT CLIP
    k_positive_parameter_array = np.array([np.clip(gen_rates.k_parameter_generation()[0], rate_min, rate_max) for _ in range(simulation_size)])
    k_negative_parameter_array = np.array([np.clip(gen_rates.k_parameter_generation()[1], rate_min, rate_max) for _ in range(simulation_size)])
    p_positive_parameter_array = np.array([np.clip(gen_rates.p_parameter_generation()[0], rate_min, rate_max) for _ in range(simulation_size)])
    p_negative_parameter_array = np.array([np.clip(gen_rates.p_parameter_generation()[1], rate_min, rate_max) for _ in range(simulation_size)])
    gen_concentrations = all_parameter_generation(sites_n, "distributive", "gamma", (0.40637, 0.035587), verbose = False)
    concentration_min, concentration_max = 1e-4, 1e-1
    x_tot_value_parameter_array = np.array([np.clip(gen_concentrations.total_concentration_generation()[0], concentration_min, concentration_max) for _ in range(simulation_size)])
    y_tot_value_parameter_array = np.array([np.clip(gen_concentrations.total_concentration_generation()[1], concentration_min, concentration_max) for _ in range(simulation_size)])

    results = Parallel(n_jobs=-2, backend="loky")(
        delayed(process_sample)(
            i,
            sites_n,
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
    
    monostable_results_list = []
    multistable_results_list = []
    for mon_res, multi_res in results:
        if mon_res is not None:
            monostable_results_list.append(mon_res)
        if multi_res is not None:
            multistable_results_list.append(multi_res)
    monofile = f"monostability_parameters_{simulation_size}_{sites_n}.pkl"
    multifile = f"multistability_parameters_{simulation_size}_{sites_n}.pkl"
    with open(monofile, "wb") as f:
        pickle.dump(monostable_results_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(multifile, "wb") as f:
        pickle.dump(multistable_results_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    return

def main():
    sites_n = 2
    simulation(sites_n, 10)

if __name__ == "__main__":
    main()
