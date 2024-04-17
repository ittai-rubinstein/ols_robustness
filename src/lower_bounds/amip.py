from typing import Tuple

import autograd.numpy as np  # Use autograd's numpy wrapper for automatic differentiation
from autograd import grad

def compute_beta_and_f(norm_X: np.ndarray, residuals: np.ndarray, w: np.ndarray, e: np.ndarray) -> Tuple[np.ndarray, float]:
    WX = norm_X * w[:, np.newaxis]  # Element-wise multiplication for weighted samples
    WR = residuals * w  # Weighted residuals

    # Linear regression to find beta
    beta = np.linalg.inv(WX.T @ WX) @ WX.T @ WR

    # Function f(w) = <beta, e>
    f_w = np.dot(beta, e)

    return beta, f_w


def f(w: np.ndarray, X: np.ndarray, residuals: np.ndarray, e: np.ndarray) -> float:
    _, f_w = compute_beta_and_f(X, residuals, w, e)
    return f_w


def approximate_most_influential_pertrubation(X: np.ndarray, residuals: np.ndarray, e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Initial weights (all ones)
    w_ones = np.ones(X.shape[0])

    # Compute gradient of f at w=ones
    grad_f = grad(f)(w_ones, X, residuals, e)

    # Verify f(w=ones) is approximately 0
    # if not np.isclose(f(w_ones, X, residuals, e), 0, atol=1e-3):
    #     raise ValueError(f"f(w=ones)={f(w_ones, X, residuals, e)} does not evaluate to 0.")

    # Sort indices based on gradient values in descending order
    sorted_indices = np.argsort(-grad_f)

    return sorted_indices, grad_f
