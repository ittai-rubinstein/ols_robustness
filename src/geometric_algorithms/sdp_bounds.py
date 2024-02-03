import cvxpy as cp
import numpy as np
from typing import Tuple, Union


def optimize_matrix(outer_prod: np.ndarray, k: int, **kwargs) -> Tuple[np.ndarray, float]:
    """
    Optimize a symmetric matrix V subject to several constraints, with the objective
    of maximizing the dot product of V and a given matrix outer_prod.

    Parameters:
    - outer_prod: np.ndarray, the matrix with which the dot product is maximized.
    - k: int, a parameter that influences several constraints, including the sum of
         diagonal entries and the overall sum of elements in V.

    Returns:
    - Tuple[np.ndarray, float]: A tuple containing the optimized matrix V and the
      value of the objective function.
    """
    # Size of the matrix V is determined by outer_prod's dimensions
    n = outer_prod.shape[0]

    # Define the variable matrix V
    V = cp.Variable((n, n), symmetric=True)

    # Define the constraints
    constraints = [cp.diag(V) <= 1/k * cp.sum(V, axis=1),  # Diagonal dominance constraint
                   V >= 0,  # Non-negativity constraint
                   V <= 1,  # Upper bound constraint
                   cp.diag(V)[:, np.newaxis] >= V,  # Diagonal elements are the largest in their rows/columns
                   cp.sum(V) == k**2,  # Sum of all elements constraint
                   cp.sum(cp.diag(V)) == k]  # Sum of diagonal entries constraint

    # Define the objective function to maximize <V, outer_prod>
    objective = cp.Maximize(cp.trace(V @ outer_prod))

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(kwargs)

    # Return the optimized matrix V and the value of the objective function
    return V.value, problem.value

def optimize_matrix_sdp(outer_prod: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
    """
    Optimize a symmetric and PSD matrix V subject to several constraints, with the objective
    of maximizing the dot product of V and a given matrix outer_prod, using semidefinite programming.

    Parameters:
    - outer_prod: np.ndarray, the matrix with which the dot product is maximized.
    - k: int, a parameter that influences several constraints, including the sum of
         diagonal entries and the overall sum of elements in V.

    Returns:
    - Tuple[np.ndarray, float]: A tuple containing the optimized PSD matrix V and the
      value of the objective function.
    """
    # Size of the matrix V is determined by outer_prod's dimensions
    n = outer_prod.shape[0]

    # Define the variable matrix V with PSD constraint
    V = cp.Variable((n, n), PSD=True)

    # Define the constraints
    constraints = [cp.diag(V) <= 1/k * cp.sum(V, axis=1),  # Diagonal dominance constraint
                   V >= 0,  # Non-negativity constraint
                   V <= 1,  # Upper bound constraint
                   cp.diag(V)[:, np.newaxis] >= V,  # Diagonal elements are the largest in their rows/columns
                   cp.sum(V) == k**2,  # Sum of all elements constraint
                   cp.sum(cp.diag(V)) == k]  # Sum of diagonal entries constraint

    # Define the objective function to maximize <V, outer_prod>
    objective = cp.Maximize(cp.trace(V @ outer_prod))

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Return the optimized PSD matrix V and the value of the objective function
    return V.value, problem.value


import numpy as np
import scipy as sp


IDENTITY_COEFF_GAUSSIAN = np.sqrt(3 / 2)
ALPHA_GAUSSIAN = 2 / 3


def compute_i_phi_coefficients(alpha: float, d: Union[float, int]) -> Tuple[float, float]:
    """
    Given alpha and d, returns i and phi coefficients such that:
    (c_i + c_phi * phi)^2 * (alpha + d * (1 - alpha) * phi) = 1 modulu (phi^2 = phi)
    This polynomial equality is ensured if
    (c_i + c_phi * phi)^2 * (alpha + d * (1 - alpha) * phi) = 1
    for phi in {0, 1}
    This means that:
     1. substituting phi=1, we have: (c_i + c_phi) = sqrt(1 / (alpha + d*(1 - alpha)))
     2. substituting phi=0, we have: c_i = sqrt(1 / alpha)
    Therefore (c_i, c_phi) = sqrt(1 / alpha), sqrt(1 / (alpha + (d * (1 - alpha)))) - sqrt(1 / alpha)
    :param alpha: The ratio between the portion of Norm{v}^4 that we draw from Norm{V}^2 (vs tr(V)^2 / d)
    :param d: The dimension of v
    :return: (c_i, c_phi) used for computing inverse root of cost matrix.
    """
    return np.sqrt(1 / alpha), np.sqrt(1 / (alpha + (d * (1 - alpha)))) - np.sqrt(1 / alpha)


def bound_42_hypercontractivity_gm(
        X: np.ndarray,
        alpha: float = ALPHA_GAUSSIAN
) -> np.ndarray:
    """
    Given an n by d array X, bound the 4-2 hyper-contractivity of its rows when projected onto any unit vector v.
    In other words, find a real value C such that for all v:
    \sum_i <X_i, v>^4 <= C (\sum_i <X_i, v>^2)^2
    :param X:
    :return:
    """
    n, d = X.shape
    # Step 1: renormalize X to have covariance Identity / n.
    # With this normalization, for any vector v, sum_i <X_i, v>^2 = Norm_2{v}^2
    cov = X.T @ X
    X_normalized = (sp.linalg.sqrtm(np.linalg.inv(cov)) @ X.T).T

    # Step 2: Generate an n by d**2 array whose ith row is the outer product between the ith row of X and itself.
    # At this point, note that for any vector v, we can write the 4th moment of projecting X onto it as:
    # Norm_4{X@v}^4 = \sum_{i \in [n]} <X_i, v>^4 = Norm_2{x_tensor_x @ (tensor_product(v, v)}^2
    # Note that Norm_2{v tensor v} = Norm_2{v}^2, so the ratio between
    # Norm_2{x_tensor_x @ (tensor_product(v, v)}^2 and Norm_2{v tensor v}^2 is equal to the ratio between
    # Norm_4{X@v}^4 = sum_i <X_i, v>^4 and Norm_2{X@v}^4 = Norm_2{v}^4 = (sum_i <X_i, v>^2)^2.
    x_tensor_x = np.einsum('ij,ik->ijk', X_normalized, X_normalized).reshape(n, d * d)

    # phi_coefficient = (np.sqrt(3 / (2 + d)) - np.sqrt(3 / 2)) / d

    # Step 3: Introduce a better scoring function.
    # phi is a vector such that inner_product(phi, v tensor v) = \sum_i v_i v_i = Norm_2{v}^2
    # This is useful, because we can now also generate Norm_2{v}^4 through the quadratic form:
    # (v tensor v).T @ phi @ phi.T @ (v tensor v)
    # This is good, because if we are working with the relaxation of V ?= v tensor v,
    # the optimizer might want to put more weight on the diagonal of V than should come from a tensor product between a
    # vector and itself. By telling the optimizer that instead of the cost of a matrix V being Norm_2{V}^2
    # (which maps to Norm_2{v}^4, if V = v tensor v), we can use the improved score function
    # Score(V) = alpha * Norm_2{V}^2 + (1 - alpha) * (<V, phi>^2)
    # This score function is still a quadratic form, and incentivizes the optimizer to put a more reasonable weight on
    # the diagonal of V.
    phi = np.reshape(np.identity(d), (d * d,))

    # Step 4: Take the improved score function into account
    # Here we use some algebra magic taken from Freund and Hopkins' paper.
    # Let Phi = (phi phi.T)/d. Freund and Hopkins make the optimization that:
    # ((2/3)*I + d*(1/3)*Phi)^(-1/2) = (sqrt(3/(2+d)) - sqrt(3/2)) Phi + sqrt(3/2) I;
    # In cases where d**2 > n, computing the LHS directly might be less efficient than using a precomputed formula.
    # To prove this equality, we simply note that Phi^2 = Phi, so for it to hold as a polynomial equation mod this
    # equation, we need only prove it for Phi = 0 and for Phi = 1.
    # To allow experimentation with new alpha values, we generalize this fast inversion logic for any alpha\in(0, 1)
    identity_coefficient, phi_coefficient = compute_i_phi_coefficients(alpha, d)
    better_tensor = identity_coefficient * x_tensor_x + \
                    (phi_coefficient * (x_tensor_x @ phi)[:, np.newaxis] * phi[np.newaxis, :] / d)

    # Step 5: Reduce to problem 1.
    # We first note that for M = X tensor X = matrix whose ith row is a flattened X_i @ X_i.transpose()
    # For any vector v, taking V to be the flattened version of v tensor v, we have that:
    #       (M @ V)_i = <X_i, v>^2
    # Moreover, taking W = ((2/3) I + (1/3) phi@phi.T), we know that V.T @ W @ V = V @ V
    # (due to the structure that V is a tensor of a vector with itself).
    # Therefore, taking tilde{V} = W^{1/2} V, we have that Norm_2{tilde{V}} = Norm_2{V}, and that for
    #           M'=M @ W^{-1/2} (= better_tensor in the code),
    # we have that:
    #           M' @ tilde{V} = M @ W^{-1/2} @ W^{1/2} @ V = (<X_i, v>^2)_{i in [n]}
    # Therefore, from a direct application of the CS inequality, we have that for any set S subseteq [n]:
    #           Norm_2{1_S^T @ M'} @ Norm_2{tilde{V}} >= 1_S^T @ M' @ tilde{V} = (<X_i, v>^2)_{i in S}
    # Moreover, this is implicitly what happens when we generate the 4-2 hypercontractivity following the recipe
    # of Freund and Hopkins. This is because their bound relies on bounding Norm_2{M' @ tilde{V}} by
    # max{sigma(M')} * Norm_2{tilde{V}} = max(sigma) * Norm_2{V} = max(sigma) * Norm_2{v}^2
    # (where sigma denotes the singular values of M')
    # The CS inequality is then applied using with this bound along with the fact that Norm_2{1_S} <= sqrt(|S|).
    # Therefore, so long as our upper bound on Norm_2{1_S^T @ M'} is better than sqrt(|S|) * max(sigma), then we
    # should get tighter bounds on our target.

    return better_tensor @ better_tensor.transpose()

def bound_42_hypercontractivity(X: np.ndarray) -> float:
    """
    Given an n by d array X, bound the 4-2 hyper-contractivity of its rows when projected onto any unit vector v.
    In other words, find a real value C such that for all v:
    \sum_i <X_i, v>^4 <= C (\sum_i <X_i, v>^2)^2
    :param X:
    :return:
    """
    n, d = X.shape
    return n * max(np.linalg.eigvalsh(bound_42_hypercontractivity_gm(X)))
    #
    # # Step 1: renormalize X to have covariance Identity / n.
    # # With this normalization, for any vector v, sum_i <X_i, v>^2 = Norm_2{v}^2
    # cov = X.T @ X
    # X_normalized = (sp.linalg.sqrtm(np.linalg.inv(cov)) @ X.T).T
    #
    # # Step 2: Generate an n by d**2 array whose ith row is the outer product between the ith row of X and itself.
    # # At this point, note that for any vector v, we can write the 4th moment of projecting X onto it as:
    # # Norm_4{X@v}^4 = \sum_{i \in [n]} <X_i, v>^4 = Norm_2{x_tensor_x @ (tensor_product(v, v)}^2
    # # Note that Norm_2{v tensor v} = Norm_2{v}^2, so the ratio between
    # # Norm_2{x_tensor_x @ (tensor_product(v, v)}^2 and Norm_2{v tensor v}^2 is equal to the ratio between
    # # Norm_4{X@v}^4 = sum_i <X_i, v>^4 and Norm_2{X@v}^4 = Norm_2{v}^4 = (sum_i <X_i, v>^2)^2.
    # x_tensor_x = np.einsum('ij,ik->ijk', X_normalized, X_normalized).reshape(n, d * d)
    #
    # phi_coefficient = (np.sqrt(3 / (2 + d)) - np.sqrt(3 / 2)) / d
    # # Step 3: Introduce a better scoring function.
    # # phi is a vector such that inner_product(phi, v tensor v) = \sum_i v_i v_i = Norm_2{v}^2
    # # This is useful, because we can now also generate Norm_2{v}^4 through the quadratic form:
    # # (v tensor v).T @ phi @ phi.T @ (v tensor v)
    # # This is good, because if we are working with the relaxation of V ?= v tensor v,
    # # the optimizer might want to put more weight on the diagonal of V than should come from a tensor product between a
    # # vector and itself. By telling the optimizer that instead of the cost of a matrix V being Norm_2{V}^2
    # # (which maps to Norm_2{v}^4, if V = v tensor v), we can use the improved score function
    # # Score(V) = alpha * Norm_2{V}^2 + (1 - alpha) * (<V, phi>^2)
    # # This score function is still a quadratic form, and incentivizes the optimizer to put a more reasonable weight on
    # # the diagonal of V.
    # phi = np.reshape(np.identity(d), (d * d,))
    #
    # # Step 4: Take the improved score function into account
    # # Here we use some algebra magic taken from Freund and Hopkins' paper.
    # # Let Phi = (phi phi.T)/d. Freund and Hopkins make the optimization that:
    # # ((2/3)*I + d*(1/3)*Phi)^(-1/2) = (sqrt(3/(2+d)) - sqrt(3/2)) Phi + sqrt(3/2) I;
    # # In cases where d**2 > n, computing the LHS directly might be less efficient than using a precomputed formula.
    # # To prove this equality, we simply note that Phi^2 = Phi, so for it to hold as a polynomial equation mod this
    # # equation, we need only prove it for Phi = 0 and for Phi = 1.
    # better_tensor = IDENTITY_COEFF_GAUSSIAN * x_tensor_x + \
    #                 (phi_coefficient * (x_tensor_x @ phi)[:, np.newaxis] * phi[np.newaxis, :])
    #
    # sing_vals = np.linalg.svd(better_tensor, compute_uv=False)
    # # Step 5: Normalize output.
    # # The 4-2 hypercontractivity is given by (1/n sum_i <X_i, v>^4) / (1/n sum_i <X_i, v>^2)^2
    # # And we bounded (sum_i <X_i, v>^4) / (sum_i <X_i, v>^2)^2 = (1/n) H_{4,2}
    # return (np.max(sing_vals)**2) * n


# def get_C2(X):
#     '''
#     X = an n x d data matrix (numpy array)
#     beta = a length d vector of coefficients (numpy array)
#     '''
#     n = X.shape[0]
#     d = X.shape[1]
#     cov = np.dot(X.T, X) / n
#
#     Xpreconditioned = np.dot(sp.linalg.sqrtm(np.linalg.inv(cov)), X.T).T
#
#     WM2 = np.zeros((n, d * d))
#     for i in range(n):
#         phi_coeff = np.sqrt(3 / (2 + d)) - np.sqrt(3 / 2)
#         identity_coeff = np.sqrt(3 / 2)
#         phi = np.reshape(np.identity(d), d * d)
#         vec_square = np.kron(Xpreconditioned[i], Xpreconditioned[i])
#         WM2[i, :] = identity_coeff * vec_square + phi_coeff * np.dot(phi, vec_square) * phi / d
#
#     WM2 = 1 / np.sqrt(n) * WM2
#
#     sing_vals = np.linalg.svd(WM2, compute_uv=False)
#
#     return (sing_vals[0])
