from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from tqdm import tqdm

def max_sum_ips(X: np.ndarray) -> np.ndarray:
    """
    Computes the maximum sum of inner products for each subset of vectors in X
    up to size k, for all k in the range 1 to n, using an efficient cumulative sum approach.

    Args:
        X (np.ndarray): An nxd array representing a set of n vectors in R^d.

    Returns:
        np.ndarray: A 1D array where the k-th element represents the maximum sum of
                    inner products for any subset of k vectors in X.

    Method:
        The function computes the cumulative sum of vectors in X and then takes the
        norm of each cumulative sum vector to find the maximum sum of inner products.
    """

    # Compute the cumulative sum of vectors in X
    cumsum_X = np.cumsum(X, axis=0)

    # Compute the norm of each cumulative sum vector
    max_sum_ips = np.linalg.norm(cumsum_X, axis=1)

    return max_sum_ips


import numpy as np

def max_sum_ips_squared(X: np.ndarray, k_values: np.ndarray) -> np.ndarray:
    """
    Computes the maximal sum of squared inner products for subsets of vectors in X,
    by using the largest eigenvalue of the matrix X[:k, :].T @ X[:k, :] for each k.

    Args:
        X (np.ndarray): An n x d array where each row represents a vector in R^d.
        k_values (np.ndarray): An array of k values for which to compute the maximal sum.

    Returns:
        np.ndarray: A 1D array where each element corresponds to the maximal sum of squared
                    inner products for subsets of size k, determined by the largest eigenvalue
                    of the corresponding matrix for each k in k_values.
    """

    # Initialize an array to store the maximal sum for each k
    max_sums = np.zeros(len(k_values))

    # Iterate over the array of k values
    for i, k in enumerate(k_values):
        # Construct the matrix for the top k vectors
        matrix_k = X[:k, :].T @ X[:k, :]

        # Compute the eigenvalues of the matrix, which is PSD
        eigenvalues = np.linalg.eigvalsh(matrix_k)

        # The maximal sum of squared inner products is given by the largest eigenvalue
        max_sums[i] = np.max(eigenvalues)

    return max_sums



@dataclass
class CoarseNetAlgorithm:
    vector_set: np.ndarray  # Descriptive name for the n by d array of vectors
    iterations: int  # Number of iterations of the coarse-net algorithm to run
    k_values_squared: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))  # Values of k for the sum of ips squared algorithm
    max_k_for_ips: int = 0  # Maximal k for the max sum inner products (not squared) part of the algorithm
    use_progress_bar: bool = True  # Whether to use tqdm progress bar in the algorithm

    _epsilon_tail_lengths: np.ndarray = field(init=False, default=None)  # Sum of lengths of the k longest rows as a function of k
    _max_sum_ips: np.ndarray = field(init=False, default=None)  # Maximal sum of inner products
    _max_sum_ips_squared: np.ndarray = field(init=False, default=None)  # Maximal sum of squared inner products

    def __post_init__(self):
        self.vector_set = np.copy(self.vector_set)  # Ensure the vector set is copied to avoid modifying the input array
        sorted_norms = np.sort(np.linalg.norm(self.vector_set, axis=1))[::-1]
        self.epsilon_tail_lengths = np.cumsum(sorted_norms)
        self.max_sum_ips = np.zeros(self.max_k_for_ips)
        self.max_sum_ips_squared = np.zeros(len(self.k_values_squared))
        self.run_coarse_net_algorithm()

    def run_coarse_net_algorithm(self):
        max_ips_result = np.zeros(self.max_k_for_ips)
        max_ips_squared_result = np.zeros(self.k_values_squared.shape)
        loop_range = tqdm(range(self.iterations)) if self.use_progress_bar else range(self.iterations)
        for _ in loop_range:
            # Select a random vector w from a normal distribution
            w = np.random.normal(0, 1 / np.sqrt(self.vector_set.shape[1]), self.vector_set.shape[1])

            # Sort the vector set by the absolute values of their inner products with w
            indices = np.argsort(-np.abs(self.vector_set @ w))
            sorted_vector_set = self.vector_set[indices]

            # Run max_sum_ips on the first k vectors
            max_ips_result = np.maximum(
                max_sum_ips(sorted_vector_set[:self.max_k_for_ips]),
                max_ips_result
            )
            # Run max_sum_ips_squared with the sorted rows and k_values_squared
            max_ips_squared_result = np.maximum(
                max_sum_ips_squared(sorted_vector_set, self.k_values_squared),
                max_ips_squared_result
            )
        self._max_sum_ips = max_ips_result
        self._max_sum_ips_squared = max_ips_squared_result