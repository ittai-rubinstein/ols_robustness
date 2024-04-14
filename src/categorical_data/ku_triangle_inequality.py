from dataclasses import dataclass
from typing import List, Union, Iterable, Optional, Tuple

import numpy as np
import tqdm

from src.categorical_data.dynamic_programming import dynamic_programming_2d


@dataclass
class KUMetadata:
    k_max: int
    u_max: int
    bucket_sizes: List[int]
    verbose: bool = True


@dataclass
class KUData:
    bucket_scores: List[np.ndarray]
    gram_matrix: np.ndarray

    def __add__(self, other):
        """
        Implement and addition operator with another KUData object. Addition is pointwise.
        :param other: A second KUData object.
        :return: Pointwise addition.
        """
        if not isinstance(other, KUData):
            raise ValueError("Can only add KUData instances.")

        # Ensure that the lengths of bucket_scores lists match
        if len(self.bucket_scores) != len(other.bucket_scores):
            raise ValueError("The lengths of bucket_scores lists must match.")

        # Pointwise addition of bucket_scores
        new_bucket_scores = [s + o for s, o in zip(self.bucket_scores, other.bucket_scores)]

        # Addition of gram matrices
        new_gram_matrix = self.gram_matrix + other.gram_matrix

        return KUData(bucket_scores=new_bucket_scores, gram_matrix=new_gram_matrix)

    def __mul__(self, factor):
        """
        Rescales the KU optimization problem.
        The individual bounds are scaled linearly, and the Gram matrix is scaled quadratically.
        :param factor:
        :return:
        """
        if not np.isscalar(factor):
            raise ValueError("Can only multiply by a scalar.")

        # Pointwise multiplication of bucket_scores by the factor
        new_bucket_scores = [s * factor for s in self.bucket_scores]

        # Multiplication of gram_matrix by factor squared
        new_gram_matrix = self.gram_matrix * (factor ** 2)

        return KUData(bucket_scores=new_bucket_scores, gram_matrix=new_gram_matrix)

    # Implementing the reverse multiplication (__rmul__) to handle scalar * KUData
    __rmul__ = __mul__

@dataclass
class KUBounds:
    averaging_effect_bounds: np.ndarray
    problem_1_bounds: np.ndarray

def compute_ku_bounds(problem: KUData, params: KUMetadata) -> KUBounds:
    """
    A wrapper for running our KU bounds logic.
    The idea is that we have a gram matrix of a set of samples divided into buckets, as well as some other effect
    due to the averaging within buckets. Averaging effects typically want to take a lot of samples from few buckets,
    while
    :param problem:
    :param params:
    :return:
    """
    return KUBounds(
        averaging_effect_bounds=dynamic_programming_2d(
            problem.bucket_scores, k_max=params.k_max, u_max=params.u_max, verbose=params.verbose
        ),
        problem_1_bounds=ku_triangle_inequality(
            problem.gram_matrix, bucket_sizes=params.bucket_sizes, k_max=params.k_max, u_max=params.u_max, verbose=params.verbose
        )
    )

def compute_linear_combination_ku_datas(ku_datas: List[KUData], coeffs: Union[List[float], Iterable[float], np.ndarray]) -> Tuple[KUData, int]:
    # Filter out zero coefficients and corresponding KUData objects
    filtered_ku_datas = [ku_data for ku_data, coeff in zip(ku_datas, coeffs) if coeff != 0]
    filtered_coeffs = [coeff for coeff in coeffs if coeff != 0]

    # Compute the number of non-zero coefficients
    num_non_zero_coeffs = len(filtered_coeffs)

    if num_non_zero_coeffs == 0:
        raise ValueError("At least one coefficient must be non-zero.")

    # Generate the linear combination
    linear_combination = filtered_coeffs[0] * filtered_ku_datas[0]
    for coeff, ku_data in zip(filtered_coeffs[1:], filtered_ku_datas[1:]):
        linear_combination = linear_combination + (coeff * ku_data)

    return linear_combination, num_non_zero_coeffs

def compute_linear_combination_ku_bounds(ku_datas: List[KUData], coeffs: Union[List[float], Iterable[float], np.ndarray], params: KUMetadata) -> KUBounds:
    linear_combination, num_non_zero_coeffs = compute_linear_combination_ku_datas(ku_datas, coeffs)

    # Compute KUBounds for the linear combination
    ku_bounds = compute_ku_bounds(linear_combination, params)

    # Rescale problem_1_bounds by sqrt(num_non_zero_coeffs)
    # This is because we computed problem 1 bounds = sqrt(sum(a_i^2)) geq (1/sqrt(n)) * sum(abs(a_i)) which is what we wanted.
    ku_bounds.problem_1_bounds *= np.sqrt(num_non_zero_coeffs)

    return ku_bounds

def extract_k_bounds(ku_bounds: KUBounds) -> np.ndarray:
    return np.max(
        ku_bounds.averaging_effect_bounds[1:, 1:-1] + ku_bounds.problem_1_bounds[:, 1:], axis=0
    )

def extract_u_of_k(ku_bounds: KUBounds) -> np.ndarray:
    return np.argmax(
        ku_bounds.averaging_effect_bounds[1:, 1:-1] + ku_bounds.problem_1_bounds[:, 1:], axis=0
    )




def ku_triangle_inequality(
        gram_matrix: np.ndarray, bucket_sizes: List[int], k_max: int, u_max: int,
        verbose: bool = True, return_indices_uk: Optional[np.ndarray] = None
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Goal: bound the norm of the sum of k vectors take from u unique buckets using triangle inequality logic.
    Method:
        For each vector (represented by a row in the Gram matrix), we divide its entries between:
        - a set of m=num_buckets entries representing the largest entry in each bucket.
        - a set of n-m entries representing entries that are *not* largest within their respective buckets.
        We can then use the fact that norm^2 = sum over k rows sum over k columns of the gram matrix
        and relax it to allow for any (potentially inconsistent) combination of rows and columns.

        We use the additional constraint that we must use u buckets is that we take u elements from the
        set of top elements of each bucket, and the other k-u from the 2nd or lower.
        Similarly, we only allow u of our choices of rows to be the best in their respective buckets.

    :param return_indices_uk: If this parameter is set, will return a list of the indices that go into the triangle inequality.
    :param gram_matrix:
    :param bucket_sizes:
    :param k_max:
    :param u_max:
    :param verbose:
    :return: triangle bounds [u x k]
    """
    gram_matrix = np.copy(gram_matrix)
    m = len(bucket_sizes)
    n = gram_matrix.shape[0]
    bucket_indices = np.concatenate((np.zeros(1, dtype=int), np.cumsum(bucket_sizes)))
    best_entries = np.zeros((n, m))
    # Each index added to T can only contribute its largest inner product with at most u of the buckets.
    # So we split the potential contributions of the elements of T into those constituting the largest element of their
    # respective buckets and those that do not.
    for j in tqdm.trange(m, desc="KU Triangle Inequality - splitting rows", disable=True):
        start_index, end_index = bucket_indices[j:j+2]
        gram_matrix[:, start_index:end_index] = -np.sort(-gram_matrix[:, start_index:end_index], axis=1)
        best_entries[:, j] = gram_matrix[:, start_index]
        gram_matrix[:, start_index] = -np.inf

    # The maximal contribution that a sample can have due to its $u$ highest inner products and its $k-u$ next best inner products.
    # We concat a 0 to each cumulative sum so that the 0 index of the cumulative sum represents a summation over 0 elements.
    best_u_contributions = np.concatenate((np.zeros((n, 1)), np.cumsum(np.sort(best_entries, axis=1)[:, ::-1], axis=1)), axis=1)
    best_kmu_contributions = np.concatenate((np.zeros((n, 1)), np.cumsum(np.sort(gram_matrix, axis=1)[:, ::-1], axis=1)), axis=1)

    # We now compute the total possible contribution of adding each possible sample to T, due to both its $u$ and
    # its $k-u$ inner products:
    sample_contributions = np.full((n, u_max, k_max), -np.inf)
    for u in tqdm.trange(
        min(u_max, k_max), desc="KU Triangle Inequality - combining contributions", disable=True
    ):
        sample_contributions[:, u, u:k_max] = best_u_contributions[:, u][:, np.newaxis] + best_kmu_contributions[:, :k_max - u]

    if return_indices_uk is not None:
        indices = []
        for k, u in enumerate(return_indices_uk):
            order = np.argsort(-sample_contributions[:, u, k+1])
            indices.append(order[:k+1])
        return indices


    # We now perform a similar trick to enforce the fact that T must utilize exactly $u$ separate buckets.
    # For each bucket, we take its highest contributing member (for each k,u) and remove it from the main list of possible
    # contributions to choose from. When optimizing T we can only take a summation of exactly u elements from the "best_contributions"
    # and k - u elements from the rest.
    best_contributions = np.zeros((m, u_max, k_max))
    for j in tqdm.trange(m, desc="KU Triangle Inequality - splitting columns", disable=True):
        start_index, end_index = bucket_indices[j:j + 2]
        # Use partition to move the largest elements of this bucket to the start:
        sample_contributions[start_index:end_index, :, :] = -np.partition(-sample_contributions[start_index:end_index, :, :], 0, axis=0)
        # Copy these elements to the side
        best_contributions[j, :, :] = sample_contributions[start_index, :, :]
        # Effectively removes these elements from further consideration in the k-u portion:
        sample_contributions[start_index, :, :] = -np.inf

    best_contributions = -np.sort(-best_contributions, axis=0)
    cumsum_best_contributions = np.cumsum(best_contributions, axis=0)
    # Concatenate 0 so that 0 index corresponds to taking a summation of only 0 elements
    cumsum_best_contributions = np.concatenate((np.zeros((1, u_max, k_max)), cumsum_best_contributions), axis=0)


    # We never look at the sample_contributions beyond the k_max largest at any index, so we may as well drop those indices
    # sample_contributions are still logically indexed as sample index "i" or "n" (now simply capped at k_max), u, k.
    # sample_contributions = -np.partition(-sample_contributions, k_max, axis=0)[:k_max, :, :]

    # Initialize norms_squared with -np.inf
    norms_squared = np.full((u_max, k_max), -np.inf)
    for u in tqdm.trange(u_max, desc="KU Triangle Inequality", disable=not verbose):
        for k in range(k_max):
            if k >= u:
                # Use partition to bring top k-u elements to the front
                sample_contributions[:, u, k] = -np.partition(-sample_contributions[:, u, k], k - u, axis=0)
                # Result of calculation: sum over k-u larges elements of sample_contributions and u largest elements of best_contributions
                norms_squared[u, k] = np.sum(sample_contributions[:k - u, u, k]) + cumsum_best_contributions[u, u, k]

    # Ignore warning about -np.infs in sqrt as these will be removed
    with np.errstate(invalid='ignore'):
        norms = np.sqrt(norms_squared)
        norms = np.nan_to_num(norms, nan=-np.inf)
    return norms
