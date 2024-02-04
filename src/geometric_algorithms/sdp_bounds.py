import cvxpy as cp
import numpy as np
from typing import Tuple


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


