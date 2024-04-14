from typing import List

import numpy as np
import tqdm

from src.categorical_data.dynamic_programming import dynamic_programming_1d


def categorical_triangle_inequality(split_X: List[np.ndarray], k: int, verbose: bool = True) -> List[np.ndarray]:
    """
    A version of the triangle inequality algorithm adapted to categorical datasets.

    We are given a list of arrays split_X as input.
    Each of these n_j by d sized arrays represents a set of n_j vectors in R^d.

    Our goal is to bound the L2 norm of the sum of any k vectors such that they are distributed between the buckets so
        that there are k_j samples taken from the jth bucket.
    This function should output a list of vectors bound_j such that
        max_T norm{sum_j sum_{i in T intersect B_j} X_i }^2 <= sum_j bound_j[k_j]

    Norm{sum_i X_i}^2 = sum_{i1, i2} <X_i1, X_i2> =
        sum_{j1, j2} sum_{i1 in T intersect B_j1, i2 in T intersect B_j2} <X_i1, X_i2> <=
        sum_{j} sum_{i1 in T intersect B_j} sum_{k_j, i2s in B_j - i1} <X_i1, X_i2> +
            sum_{k - k_j largest i2 not in B_j} <X_i1, X_i2>

    Total score = score1 + score2
    score1 <= sqrt{sum_j bound_j[k_j]}
    score2 <= sum_j contribution_j[k_j]

    """
    m = len(split_X)  # Number of buckets

    # Compute all possible inner products between buckets
    # First, concatenate all buckets along the first axis
    all_samples = np.vstack(split_X)

    # Compute the inner product of all combinations
    all_combinations_inner = np.dot(all_samples, all_samples.T)

    # Initialize a list to hold the matrices of inner products between different buckets
    cross_bucket_products = []

    # Indices to help extract submatrices for cross bucket inner products
    start_indices = np.cumsum([
        bucket_j.shape[0] for bucket_j in split_X
    ], dtype=int)
    start_indices = np.concatenate((np.zeros(1, dtype=int), start_indices))
    for j, bucket_j in enumerate(tqdm.tqdm(
            split_X, desc="Computing inner products", disable=not verbose
    )):
        start_idx = start_indices[j]
        end_idx = start_indices[j+1]

        # Initialize a list to hold inner product matrices for the jth bucket with all other buckets
        cross_products_j = []

        for j2 in range(m):
            if j2 == j:
                continue
            # Extract the submatrix of inner products between the jth and kth bucket
            start_idx_j2 = start_indices[j2]
            end_idx_j2 = start_indices[j2+1]
            # print(f"{start_idx=}, {end_idx=}, {start_idx_j2=}, {end_idx_j2=}")
            cross_product_jk = all_combinations_inner[start_idx:end_idx, start_idx_j2:end_idx_j2]
            cross_products_j.append(cross_product_jk)
        cross_products_j = np.concatenate(cross_products_j, axis=1)
        cross_bucket_products.append(cross_products_j)

    # Initialize a list to hold the bounds for each bucket
    bounds = []

    for j in tqdm.trange(
            m, desc="Computing triangle inequality entries", disable=not verbose
    ):
        # Sort the jth Gram matrix and take a cumulative sum along axis=1
        sorted_gram_j = np.sort(split_X[j] @ split_X[j].T, axis=1)[:, ::-1]
        cumsum_ordered_gram_matrix_j = np.cumsum(sorted_gram_j, axis=1)

        # Append a column of zeros at the start
        cumsum_ordered_gram_matrix_j = np.hstack(
            [np.zeros((cumsum_ordered_gram_matrix_j.shape[0], 1)), cumsum_ordered_gram_matrix_j])

        # Compute cumsum for the sorted cross product matrices and append a column of zeros at the start
        cumsum_ordered_cross_product_j = np.cumsum(np.sort(cross_bucket_products[j], axis=1)[:, ::-1], axis=1)
        cumsum_ordered_cross_product_j = np.hstack([
            np.zeros((cumsum_ordered_cross_product_j.shape[0], 1)), cumsum_ordered_cross_product_j
        ])

        # Initialize a numpy array to hold the bounds for the jth bucket
        bounds_j = np.zeros(cumsum_ordered_gram_matrix_j.shape[1])

        for k_j in range(1, min(cumsum_ordered_gram_matrix_j.shape[1], k + 1)):
            k_j_prime = k - k_j
            scores_j = (
                cumsum_ordered_gram_matrix_j[:, k_j] +
                cumsum_ordered_cross_product_j[:, k_j_prime]
            )

            # Find the sum over the k_j largest scores using np.partition
            partitioned_scores_j = np.partition(scores_j, -k_j)[-k_j:]
            bounds_j[k_j] = np.sum(partitioned_scores_j)

        bounds.append(bounds_j)

    return bounds


def sqrt_plus_linear_dynamic_programming(
        quadratic_scores: List[np.ndarray], linear_scores: List[np.ndarray],
        quadratic_score_total: np.ndarray, linear_score_total: np.ndarray,
        k: int, verbose: bool = False
) -> float:
    """
    Score function we want to bound is:
        max_{sum_j k_j = k} over (sqrt{sum_j Q_j [k_j]} + sum_j L_j[k_j])
    We expect second term to be much smaller if we can enforce both to use the same k_j.
    Strategy:
    Let $a$ be a scalar roughly equal to
        a = x / y = (max_{sum_j k_j = k} over (sum_j Q_j [k_j]) / (max_{sum_j k_j = k} over sqrt{sum_j Q_j [k_j]})
    We can bound b = max (a*x + y).
    Given a and b, maximizing the score sqrt{x} + y, we consider three cases:
        1. Near the point where x = 0 => we cannot have a local optimum there, since d score / dx -> infty as x -> 0^+
        2. Near the point y = 0, we have score = sqrt{x} = sqrt{b / a}
        3. Otherwise, Lagrange multipliers give us x = frac{1}{4a^2}, y = b - frac{1}{4a}.
        If y < 0, throw this case. Otherwise, result is sqrt{x} + y = b + frac{1}{4a}.
    :param quadratic_scores: a list of numpy arrays corresponding to the quadratic contributions to the rewards of assigning k_j removals to bucket j.
    :param linear_scores: a list of numpy arrays corresponding to the linear contributions to the rewards of assigning k_j removals to bucket j.
    :param quadratic_score_total: sqrt{Tau(k)} = best score obtained by optimizing only the quadratic score.
    :param linear_score_total: Alpha(k) = best scores obtained by optimizing only the linear score.
    :param k:
    :param verbose:
    :return:
    """
    a = linear_score_total[k - 1] / quadratic_score_total[k - 1]
    b = np.sqrt(
        dynamic_programming_1d(
            bounds_list=[
                (quadratic_score * a) +
                linear_score
                for quadratic_score, linear_score in
                zip(quadratic_scores, linear_scores)
            ], k_max=k + 1, verbose=verbose
        )[k]
    )
    print(f"{a=}")
    print(f"{b=}")
    if b >= (1 / (4 * a)):
        return b + (1 / (4 * a))
    else:
        return np.sqrt(b / a)


def sqrt_plus_linear_dynamic_programming_old(
        quadratic_scores: List[np.ndarray], linear_scores: List[np.ndarray],
        quadratic_score_total: np.ndarray, linear_score_total: np.ndarray,
        k: int, verbose: bool = False
) -> float:
    """
    Score function we want to bound is:
        max_{sum_j k_j = k} over (sqrt{sum_j Q_j [k_j]} + sum_j L_j[k_j])
    We expect second term to be much smaller if we can enforce both to use the same k_j.
    Strategy:
    We first bound each term independently to get a bound on Tau = sqrt{sum_j Q_j [k^1_j]} and Alpha = sum_j L_j[k^2_j])
    sqrt{Tau(k)} = quadratic_score_total, and Alpha(k) = linear_score_total are bounds
        that we can generate individually from the direct DP algorithm
    This gives us an initial bound of score <= bound_0 = sqrt{Tau} + Alpha
    We can now write:
    score^2 = sum_j (Q_j[k_j] + L_j[k_j] * (2 sqrt{sum_i Q_i[k_i]} + sum_i L_i[k_i])) <=
            <= sum_j (Q_j[k_j] + L_j[k_j] * (score + sqrt{Tau}))
    Therefore, we may define
    bound_{i+1} = sqrt{max_{k_1 + ... + k_m = k} sum_j T_j[k_j] + A_j[k_j] * (bound_i + sqrt{Tau})
    :param quadratic_scores: a list of numpy arrays corresponding to the quadratic contributions to the rewards of assigning k_j removals to bucket j.
    :param linear_scores: a list of numpy arrays corresponding to the linear contributions to the rewards of assigning k_j removals to bucket j.
    :param quadratic_score_total: sqrt{Tau(k)} = best score obtained by optimizing only the quadratic score.
    :param linear_score_total: Alpha(k) = best scores obtained by optimizing only the linear score.
    :param k:
    :param verbose:
    :return:
    """
    bound = quadratic_score_total[k - 1] + linear_score_total[k - 1]
    for i in range(2):
        bound = np.sqrt(
            dynamic_programming_1d(
                bounds_list=[
                    quadratic_score +
                    linear_score * (bound + quadratic_score_total[k - 1])
                    for quadratic_score, linear_score in
                    zip(quadratic_scores, linear_scores)
                ], k_max=k + 1, verbose=verbose
            )[k]
        )
    return bound
