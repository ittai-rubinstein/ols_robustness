from typing import Tuple

import numpy as np
import tqdm
from sklearn.linear_model import LinearRegression


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
        raise NotImplementedError()

    n_samples = X.shape[0]
    remaining_indices = list(range(n_samples))
    removal_order = []
    removal_effects = []

    if efficient_kzc:
        raise NotImplementedError()

    for _ in tqdm.trange(0, n_samples-1, desc="Computing KZC Removals", disable=not progress_bar):
        min_score = np.inf
        sample_to_remove = None
        for i in remaining_indices:
            # Try removing sample i
            temp_indices = remaining_indices.copy()
            temp_indices.remove(i)

            # Fit regression without sample i
            X_temp = X[temp_indices]
            y_temp = residuals[temp_indices]
            model = LinearRegression().fit(X_temp, y_temp)
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