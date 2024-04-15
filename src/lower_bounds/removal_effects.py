from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import tqdm

from src.lower_bounds.amip import approximate_most_influential_pertrubation
from src.lower_bounds.lower_bounds import compute_removal_effects, compute_removal_effects_direct
from src.lower_bounds.problem_1_lower_bounds import Problem1LowerBounds
import src.lower_bounds.problem_1_lower_bounds as problem_1_lower_bounds


@dataclass
class LowerBoundConfig:
    run_triple_greedy = False
    run_amip: bool = True
    run_kzcs21: bool = False
    verbose: bool = True


import numpy as np


def greedy_optimization(gram_matrix: np.ndarray, initial_set: np.ndarray, k: int,
                        delta_scores: np.ndarray = None, time_out: Optional[int] = None) -> np.ndarray:
    """
    Execute a greedy optimization to maximize the score of set T of fixed size k.

    Parameters:
    - gram_matrix (np.ndarray): Gram matrix used to update delta_scores.
    - initial_set (np.ndarray): The initial set of indices as a numpy array.
    - k (int): Fixed size of the set T.
    - delta_scores (np.ndarray, optional): Initial delta scores; if None, will be computed.

    Returns:
    - np.ndarray: Optimized set T as a numpy array.
    """
    current_set = np.array(list(initial_set), dtype=int)
    n = gram_matrix.shape[0]

    if time_out is None:
        time_out = n

    if delta_scores is None:
        delta_scores = np.zeros(n)
        for idx in current_set:
            delta_scores += 2 * gram_matrix[idx]

    # Start the greedy optimization
    improved = True
    for i in range(time_out):
        improved = False

        # Calculate potential increases for all not in current_set
        outside_indices = np.setdiff1d(np.arange(n), current_set)
        addition_scores = delta_scores[outside_indices]
        removal_scores = delta_scores[current_set]

        # Best candidate to add
        if addition_scores.size > 0:
            max_increase_idx = np.argmax(addition_scores)
            max_increase = addition_scores[max_increase_idx]
            best_addition = outside_indices[max_increase_idx]

        # Best candidate to remove
        if removal_scores.size > 0:
            min_decrease_idx = np.argmin(removal_scores)
            min_decrease = removal_scores[min_decrease_idx]
            best_removal = current_set[min_decrease_idx]

        # Check if making these changes is beneficial
        if max_increase - min_decrease > 0:
            # Update the set
            current_set[min_decrease_idx] = best_addition
            # Update delta_scores
            delta_scores += 2 * (gram_matrix[best_addition] - gram_matrix[best_removal])
            delta_scores[best_addition] -= 2*gram_matrix[best_addition, best_addition]
            delta_scores[best_removal] += 2*gram_matrix[best_removal, best_removal]
            improved = True
        else:
            break

    return current_set


@dataclass
class LowerBoundResult:
    removal_effects: np.ndarray
    removal_sets: List[List[int]]


def triple_greedy(X: np.ndarray, R: np.ndarray, axis_of_interest: np.ndarray, print_tqdm:bool = True) -> LowerBoundResult:
    n, d = X.shape
    Z = X @ axis_of_interest
    XZ = X * Z[:, np.newaxis]
    XR = X * R[:, np.newaxis]
    asymmetric_matrix = XZ @ XR.T
    gram_matrix = (asymmetric_matrix + asymmetric_matrix.T) / 2
    XZR_bounds = Problem1LowerBounds(
        gram_matrix=gram_matrix, params=problem_1_lower_bounds.LowerBoundParams(
            run_greedy=True, run_very_greedy=False,
        ), print_tqdm=print_tqdm
    )
    k_vals = np.arange(1, 100)

    removal_sets = []
    for k in tqdm.tqdm(k_vals, desc="Running 2nd and 3rd greedy steps", disable=not print_tqdm):
        greedy1 = np.array(XZR_bounds.greedy_lower_bound.order_of_selection[:k])
        delta_scores = np.diagonal(gram_matrix).copy()
        greedy2 = greedy_optimization(gram_matrix, initial_set=greedy1, k=k, delta_scores=delta_scores)

        delta_scores += Z*R
        greedy3 = greedy_optimization(gram_matrix, initial_set=greedy2, k=k, delta_scores=delta_scores)
        removal_sets.append(greedy3.tolist())

    return LowerBoundResult(
        removal_sets=removal_sets,
        removal_effects=compute_removal_effects_direct(
            X, R, axis_of_interest, removal_sets, verbose=print_tqdm
        )
    )


class RemovalEffectsLowerBound:
    _config: LowerBoundConfig
    triple_greedy: Optional[LowerBoundResult] = None
    amip: Optional[LowerBoundResult] = None
    kzcs21: Optional[LowerBoundResult] = None

    def __init__(
            self, X: np.ndarray, R: np.ndarray,
            axis_of_interest: np.ndarray, config: LowerBoundConfig
    ):
        self._config = config
        if config.run_triple_greedy:
            if config.verbose:
                print("Computing the Triple Greedy Lower Bound...")
            self.triple_greedy = triple_greedy(X, R, axis_of_interest, config.verbose)
        if config.run_amip:
            if config.verbose:
                print("Computing the AMIP Lower Bound...")
            amip_order, _ = approximate_most_influential_pertrubation(
                X, R, axis_of_interest
            )
            self.amip = LowerBoundResult(
                removal_effects=compute_removal_effects(
                    X, R, axis_of_interest, amip_order, verbose=config.verbose, normalized=True
                ),
                removal_sets= [(amip_order[:k]).tolist() for k in range(1, len(amip_order) + 1)]
            )
        if config.run_kzcs21:
            raise NotImplementedError()