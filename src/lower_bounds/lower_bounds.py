import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import scipy
import tqdm

from src.lower_bounds.amip import approximate_most_influential_pertrubation
from src.lower_bounds.one_hot_encodings import detect_one_hot_encodings, select_and_sort_quadrants, \
    compile_ordered_sample_indices, compute_quadrants_vectorized


# Flags and Enums
class BackupSortingLogic(Enum):
    XER = auto()
    AMIP = auto()
    NO_SORT = auto()

@dataclass
class Flags:
    USE_ONE_HOT: bool
    BACKUP_SORTING_LOGIC: BackupSortingLogic

# Data structures for one-hot encodings and quadrants
@dataclass
class OneHotEncoding:
    column_index: int
    sample_indices: List[int]
    column_name: Optional[str] = None

@dataclass
class QuadrantInfo:
    column_index: int
    quadrant: str
    sample_indices: List[int]
    weight: int
    value: float
    column_name: Optional[str] = None
    score: float = 0.0  # Initialized to 0.0, will be computed later


# Main function
def process_samples(X_orig: np.ndarray, norm_X: np.ndarray, residuals: np.ndarray, flags: Flags, axis_of_interest: np.ndarray) -> np.ndarray:
    ordered_indices = []

    if flags.USE_ONE_HOT:
        one_hot_encodings = detect_one_hot_encodings(X_orig)
        if len(one_hot_encodings) > 1:
            # Corrected call to compute_quadrants_vectorized
            quadrants = compute_quadrants_vectorized(norm_X, one_hot_encodings, axis_of_interest, residuals)
            # quadrants = [q for sublist in quadrants for q in sublist.sample_indices]  # Flatten the list
            sorted_quadrants = select_and_sort_quadrants(quadrants)
            ordered_indices = compile_ordered_sample_indices(sorted_quadrants)

    if not flags.USE_ONE_HOT or len(one_hot_encodings) <= 1 or len(ordered_indices) < len(norm_X):
        remaining_indices = [i for i in range(len(norm_X)) if i not in ordered_indices]

        if flags.BACKUP_SORTING_LOGIC == BackupSortingLogic.XER:
            products = norm_X @ axis_of_interest * residuals
            sorted_indices = np.argsort(-products)
            sorted_indices = [i for i in sorted_indices if i in remaining_indices]
            ordered_indices.extend(sorted_indices)
        elif flags.BACKUP_SORTING_LOGIC == BackupSortingLogic.AMIP:
            sorted_indices, _ = approximate_most_influential_pertrubation(norm_X[remaining_indices], residuals[remaining_indices], axis_of_interest)
            ordered_indices.extend([remaining_indices[i] for i in sorted_indices])
        elif flags.BACKUP_SORTING_LOGIC == BackupSortingLogic.NO_SORT:
            ordered_indices.extend(remaining_indices)

    # Remove duplicates and retain order
    final_indices = list(dict.fromkeys(ordered_indices))
    return np.array(final_indices)


def compute_removal_effects(
        X: np.ndarray, residuals: np.ndarray,
        axis_of_interest: np.ndarray,
        ordered_indices: List[int],
        k_vals: np.ndarray = None, verbose: bool = True, normalized: bool = True
) -> np.ndarray:
    """
    Computes the effect of removing the first k samples from the linear regression.
    <axis_of_interest, (I - sum over first k outer products)^(-1) @ (sum over first k rows of XR)>

    Parameters:
    - norm_X (ndarray): The normalized samples array (n x d).
    - residuals (ndarray): The residuals vector (n,).
    - axis_of_interest (ndarray): The axis of interest vector (d,).
    - ordered_indices (List[int]): The list of sample indices ordered by priority.
    - k_vals (ndarray, optional): An array of k values representing the budget for sample removals. Defaults to np.arange(1, num_samples // 10).

    Returns:
    - ndarray: An array of scores corresponding to each k in k_vals.
    """
    if normalized:
        norm_X = X
    else:
        Sigma = X.T @ X
        root_Sigma = scipy.linalg.sqrtm(Sigma)
        norm_X = X @ np.linalg.inv(root_Sigma)
        axis_of_interest = np.linalg.inv(root_Sigma) @ axis_of_interest
    num_samples = norm_X.shape[0]

    # Set default k_vals if not provided
    if k_vals is None:
        k_vals = np.arange(1, num_samples // 10)

    k_max = np.max(k_vals) + 1
    reordered_X = norm_X[ordered_indices[:k_max]]
    reordered_residuals = residuals[ordered_indices[:k_max]]

    # Compute outer products using np.einsum
    outer_products = np.einsum('bi,bj->bij', reordered_X, reordered_X)

    # Compute cumulative sum of outer products
    cumsum_outer_products = np.cumsum(outer_products, axis=0)

    # Compute XR
    XR = reordered_X * reordered_residuals[:, np.newaxis]

    # Compute cumulative sum of XR
    cumsum_xr = np.cumsum(XR, axis=0)

    scores = np.zeros(len(k_vals))
    identity_matrix = np.eye(norm_X.shape[1])

    for i, k_val in enumerate(tqdm.tqdm(k_vals, desc="Computing removal effects", disable=not verbose)):
        sum_outer_products = cumsum_outer_products[k_val - 1] if k_val > 0 else np.zeros_like(identity_matrix)
        sum_xr = cumsum_xr[k_val - 1] if k_val > 0 else np.zeros_like(XR[0])

        # Calculate the score
        matrix_inv = np.linalg.inv(identity_matrix - sum_outer_products)
        scores[i] = np.dot(axis_of_interest, matrix_inv @ sum_xr)

    return scores
