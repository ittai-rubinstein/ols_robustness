from typing import List, Optional

import numpy as np
import tqdm


def dynamic_programming_1d(
bounds_list: List[np.ndarray], k_max: int, verbose: bool = True
) -> np.ndarray:
    """
    Dynamic programming algorithm for (easy instance of) integer knapsack.
    Assume that the input is a list of numpy arrays representing scores as a function of costs. In our case, the costs
        are samples removed from a bucket and the scores are the potential effect on the linear regression result.

    For a total budget of k (running from 0 to k_max using numpy vectorization), finds the optimal division of the budget
        between the various buckets, and outputs the highest total score possible as a function of k.

    Parameters:
    :param bounds_list: A list whose jth element is a vector whose kth element represents a bound on the effect of
        removing k samples from the jth bucket
    :param k_max: The maximal budget k to be considered.
    :param verbose: Whether to include a tqdm progress bar

    Returns:
    An array whose kth element is a bound on the total effect of k removals.
    """
    # Assuming bounds_dict is available and k_total is defined
    n = len(bounds_list)
    k_max += 1  # Adjusting for 0-based indexing in NumPy arrays

    # Initialize the DP table with -inf, and set the first column to 0
    dp = np.full((n + 1, k_max), -np.inf)

    # Fill the DP table
    for i in tqdm.trange(1, n + 1, disable = not verbose, desc="Dynamic programming"):
        bounds = bounds_list[i - 1]
        dp[i, :len(bounds)] = bounds[:k_max]
        dp[i, :] = np.maximum(dp[i, :], dp[i-1, :])

        delta_ks = range(1, min(k_max, len(bounds)))
        for delta_k in delta_ks:
            dp[i, delta_k:] = np.maximum(dp[i, delta_k:], dp[i-1, :-delta_k] + bounds[delta_k])
        # for k in range(1, k_max):
        #     # Vectorized update for dp[i, k]
        #     # Create a mask for valid updates to avoid out-of-bounds and unnecessary computations
        #     valid_updates = np.arange(max(0, k - len(bounds) + 1), k + 1, dtype=int)
        #     if valid_updates.size > 0:
        #         dp[i, k] = max(dp[i, k], np.max(dp[i - 1, valid_updates] + bounds[k - valid_updates]))

    # The total score bound is the final value in the DP table
    return dp[n, :k_max - 1]


def dynamic_programming_2d(
        bucket_scores: List[np.ndarray], k_max: int, u_max: Optional[int] = None, verbose: bool = True
) -> np.ndarray:
    """
    Given a list of bounds, solves the problem of maximizing
        score = sum_j bucket_scores[j][k_j]
    under the constraints that total budget = k = sum_j k_j
    and that the number of unique buckets we can use is u = sum_j 1_{k_j != 0}
    Outputs the maximal score for all values of k, u up to k_max, u_max.
    :param bucket_scores: The rewards of individual buckets
    :param k_max: The largest value of k to consider
    :param u_max: The largest value of u to consider (by default = k_max)
    :param verbose: Whether to print a progress bar (by default = True)
    :return: Maximal total score achievable as a function of total budget k and uniqueness budget u
    """
    if u_max is None:
        u_max = k_max

    cumsum_bucket_lengths = np.cumsum([len(bucket) for bucket in bucket_scores])
    dp_table = np.full((len(bucket_scores), u_max+1, k_max+1), -np.inf)
    dp_table[:, 0, 0] = 0
    for j in tqdm.trange(
            len(bucket_scores), desc="KU Dynamic Programming", disable=not verbose
    ):
        bucket = bucket_scores[j]
        u = min(u_max + 1, j + 1)
        k = min(k_max + 1, int(cumsum_bucket_lengths[j]))
        # Base case of the DP algorithm
        if j == 0:
            dp_table[0, 1, :k] = bucket[:k]
            continue

        # Deal with the case where we did not update our
        dp_table[j, 1:u, :k] = dp_table[j - 1, 1:u, :k]

        # Deal with the case where we added values from our new bucket
        delta_ks = np.arange(1, min(len(bucket), k))
        for delta_k in delta_ks:
            dp_table[j, 1:u, delta_k:k] = np.maximum(
                dp_table[j-1, :u-1, :k - delta_k] + bucket[delta_k],
                dp_table[j, 1:u, delta_k:k]
            )

    return dp_table[-1, :, :]
