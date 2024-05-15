from typing import Tuple, List, Optional

import numpy as np
import tqdm
from sklearn.linear_model import LinearRegression


def diagonal_matrix_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Given an n by d matrix A and a d by n matrix B, returns the diagonal elements of A@B, in time d*n (instead of n*d^2).
    :param A: An n by d matrix
    :param B: A d by n matrix
    :return: The length n vector equal to (A@B).diag().
    """
    return (A*B.T).sum(axis=1)

def leave_one_out_regression_vectorized(X: np.ndarray, Y: np.ndarray, Sigma_inv_X_T: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Given a linear regression instance X@beta ~ Y, computes the effects of removing any single sample.
    :param X: an n by d matrix representing the feature vectors
    :param Y: a length n array representing the labels
    :param Sigma_inv_X_T:
    :return:
    """
    n, d = X.shape

    if Sigma_inv_X_T is None:
        Sigma = X.T @ X
        Sigma_inv = np.linalg.inv(Sigma)
        Sigma_inv_X_T = Sigma_inv @ X.T

    beta = Sigma_inv_X_T @ Y
    residuals = Y - X @ beta

    # Compute all XiT_Sigma_inv_Xi at once using einsum for efficiency
    # XiT_Sigma_inv_Xi = np.einsum('ij,jk,ki->i', X, np.linalg.inv(X.T @ X), X.T)
    XiT_Sigma_inv_Xi = diagonal_matrix_product(X, Sigma_inv_X_T)

    # Calculate alpha_i for all i
    alpha_i = 1 + XiT_Sigma_inv_Xi / (1 - XiT_Sigma_inv_Xi)

    # Calculate rho_i for all i using vectorized operations
    rho_i = Sigma_inv_X_T.T * residuals[:, np.newaxis]  # Adjust dimensions for broadcasting

    # Update B in a vectorized way
    B = beta[np.newaxis, :] - (alpha_i[:, np.newaxis] * rho_i)

    return B

# def compute_bounds(X: np.ndarray, residuals: np.ndarray, e: np.ndarray, samples_retained: List[int]) -> List[int]:
#     """
#     Computes upper bounds on the possible effect of removing any one of the remaining samples from our regression on the e axis of the fit.
#     :param X: Array of feature vectors.
#     :param residuals: Residuals of the regression.
#     :param e: Axis of interest.
#     :param samples_retained: The samples we have not removed yet.
#     :return:
#     """
#     # Number of total samples
#     n_samples = X.shape[0]
#
#     # Extract the retained samples from X and residuals
#     X_retained = X[samples_retained, :]
#     residuals_retained = residuals[samples_retained]
#
#     # Step 1: Compute Sigma and Sigma_inv
#     Sigma = X_retained.T @ X_retained
#     Sigma_inv = np.linalg.inv(Sigma)
#
#     # Step 2: Compute e_Sigma_inv
#     e_Sigma_inv = e.T @ Sigma_inv
#
#     # Step 3: Compute the regression coefficients for retained samples
#     beta_retained = Sigma_inv @ (X_retained.T @ residuals_retained)
#
#     # Compute updated residuals for retained samples
#     predicted_residuals = X_retained @ beta_retained
#     updated_residuals = residuals_retained - predicted_residuals
#
#     # Step 4: Compute the linear terms
#     linear_terms = e_Sigma_inv @ X_retained.T @ updated_residuals
#
#     # Step 5: Compute max svds for retained samples
#     X_retained_Sigma_inv = X_retained @ Sigma_inv
#     max_svds = np.linalg.norm(X_retained * X_retained_Sigma_inv, axis=1)
#
#     # Step 6: Compute inverse bounds
#     inverse_bounds = 1 / (1 - max_svds)
#
#     # Step 7: Compute Xe terms
#     norm_X_Sigma_inv = np.linalg.norm(X_retained_Sigma_inv, axis=1)
#     Xe_terms = norm_X_Sigma_inv * np.abs(X_retained @ e_Sigma_inv)
#
#     # Step 8: Compute XR terms
#     XR_terms = np.linalg.norm(X_retained * updated_residuals[:, np.newaxis], axis=1)
#
#     # Step 9: Compute upper bounds for retained samples
#     retained_upper_bounds = linear_terms + (inverse_bounds * Xe_terms * XR_terms)
#
#     # Initialize the full upper bounds array with -np.inf
#     upper_bounds = np.full(n_samples, -np.inf)
#     # Update only for retained indices
#     upper_bounds[samples_retained] = retained_upper_bounds
#
#     return upper_bounds
#
#
# def greedy_removals_efficient(X: np.ndarray, residuals: np.ndarray, e: np.ndarray):
#     n_samples = X.shape[0]
#     remaining_indices = list(range(n_samples))
#     removal_order = []
#     removal_effects = []
#
#
#
#     while remaining_indices:
#
#         # Placeholder for initializing bounds
#         upper_bounds = np.random.rand(n_samples) * 10  # Random upper bounds, replace with actual logic
#
#         # Select candidate indices where the upper bound is maximum
#         max_upper_bound = max(upper_bounds[remaining_indices])
#         candidates = [i for i in remaining_indices if upper_bounds[i] >= max_upper_bound]
#
#         max_effect = -np.inf
#         sample_to_remove = None
#
#         # Evaluate the actual effect of removing each candidate
#         for i in candidates:
#             # Temporarily remove sample i and fit regression
#             temp_indices = remaining_indices.copy()
#             temp_indices.remove(i)
#             X_temp = X[temp_indices]
#             y_temp = residuals[temp_indices]
#             model = LinearRegression().fit(X_temp, y_temp)
#             beta_S = model.coef_
#
#             # Compute actual effect (inner product with e)
#             actual_effect = np.dot(beta_S, e)
#
#             # Update bounds if needed (optional, based on your further optimizations)
#
#             # Determine if this sample has the maximum effect so far
#             if actual_effect > max_effect:
#                 max_effect = actual_effect
#                 sample_to_remove = i
#
#         # Update removal information
#         remaining_indices.remove(sample_to_remove)
#         removal_order.append(sample_to_remove)
#         removal_effects.append(max_effect)
#
#         # Update bounds for remaining samples
#         # (This would be based on additional logic you might add later)
#
#     return removal_order, removal_effects
#


def efficient_greedy_removals(
X: np.ndarray, residuals: np.ndarray, e: np.ndarray,progress_bar: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements a more efficient version of the greedy algorithm by KZC.
    The output should be exactly the same as running the original KZC but significantly faster.
    :param progress_bar: Should we display a progress bar. By default set to True.
    :param X: The features of the linear regression.
    :param residuals: The residuals of the linear regression.
    :param e: The axis of interest.
    :return: Removal order and removal effects
    """

    n, d = X.shape

    remaining_indices = list(range(n))
    samples_to_remove = []
    removal_effects = []
    X_retained = X.copy()
    R_retained = residuals.copy()
    # beta = np.linalg.inv(X.T @ X) @ X.T @ residuals
    for _ in tqdm.trange(n-d, desc="Efficient KZC", disable=not progress_bar):
        B = leave_one_out_regression_vectorized(X_retained, R_retained)
        scores = (e[np.newaxis, :] @ B.T).ravel()
        assert scores.shape == (len(remaining_indices),)
        best_score_idx = np.argmin(scores)
        absolute_idx = remaining_indices[best_score_idx]
        remaining_indices.pop(best_score_idx)
        X_retained = np.delete(X_retained, best_score_idx, axis=0)
        R_retained = np.delete(R_retained, best_score_idx)
        samples_to_remove.append(absolute_idx)
        removal_effects.append(scores[best_score_idx])

    return np.array(samples_to_remove), -np.array(removal_effects)


def greedy_removals(
X: np.ndarray, residuals: np.ndarray, e: np.ndarray,
efficient_kzc: bool = False, progress_bar: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements the greedy algorithm by KZC.
    :param progress_bar: Should we display a progress bar. By default set to True.
    :param X: The features of the linear regression.
    :param residuals: The residuals of the linear regression.
    :param e: The axis of interest.
    :param efficient_kzc: when set true, we will use a more efficient version of the kzc algorithm.
    :return: Removal order and removal effects
    """
    if efficient_kzc:
        return efficient_greedy_removals(X, residuals, e, progress_bar)

    n_samples, d = X.shape
    remaining_indices = list(range(n_samples))
    removal_order = []
    removal_effects = []

    for _ in tqdm.trange(0, n_samples-d, desc="Computing KZC Removals", disable=not progress_bar):
        min_score = np.inf
        sample_to_remove = None
        for i in remaining_indices:
            # Try removing sample i
            temp_indices = remaining_indices.copy()
            temp_indices.remove(i)

            # Fit regression without sample i
            X_temp = X[temp_indices]
            y_temp = residuals[temp_indices]
            model = LinearRegression(fit_intercept=False).fit(X_temp, y_temp)
            beta_S = model.coef_

            # Compute inner product of beta_S with e
            inner_product = np.dot(beta_S, e)

            # Check if this is the max inner product so far
            if inner_product < min_score:
                min_score = inner_product
                sample_to_remove = i

        # Remove the identified sample and update logs
        remaining_indices.remove(sample_to_remove)
        removal_order.append(sample_to_remove)
        removal_effects.append(min_score)

    return np.array(removal_order), -np.array(removal_effects)