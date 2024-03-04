from itertools import combinations
from time import time

import tqdm
import pandas as pd
import numpy as np
import scipy.linalg
from typing import List, Tuple, Dict, Optional
from sklearn.linear_model import LinearRegression

def split_and_normalize(
    df: pd.DataFrame, one_hot_columns: List[str], verbose: bool = True
) -> List[pd.DataFrame]:
    """
    Splits the input dataframe into multiple smaller dataframes based on the provided one-hot encoded columns.
    Each smaller dataframe corresponds to one of the categories represented by the one-hot encoding.
    The continuous features in these smaller dataframes are normalized by subtracting their respective means.

    Parameters:
    df (pd.DataFrame): The input dataframe containing both one-hot encoded columns and continuous features.
    one_hot_columns (List[str]): A list of column names that represent the one-hot encoding.

    Returns:
    List[pd.DataFrame]: A list of smaller dataframes, each normalized by subtracting the mean of continuous features.
    """

    smaller_dfs = []
    for col in tqdm.tqdm(one_hot_columns, desc="Separating by categorical columns", disable=not verbose):
        # Select rows where the one-hot encoded column is 1
        subset = df[df[col] == 1]

        # Drop the one-hot encoding columns for the current subset
        subset = subset.drop(columns=one_hot_columns)

        # Normalize the continuous features by subtracting the mean
        subset = subset - subset.mean()

        smaller_dfs.append(subset)

    return smaller_dfs


def perform_regression_and_append_residuals(
        split_dfs: List[pd.DataFrame],
        label: str,
        verbose: bool = True
) -> Dict[str, float]:
    """
    Perform linear regression on the combined dataframe, append residuals to each split dataframe in-place,
    and return a dictionary of coefficients for the features of interest.

    Parameters:
    split_dfs (List[pd.DataFrame]): A list of dataframes split by category (e.g., per country).
    label (str): The name of the label column for the regression.

    Returns:
    Dict[str, float]: A dictionary mapping feature names to their regression coefficients.
    """
    if verbose:
        print("Computing residuals...")
    combined_df = pd.concat(split_dfs)
    # Extract features
    features = [col for col in combined_df.columns if col != label]

    # Create a Linear Regression model
    model = LinearRegression()

    # Fit the model using the combined dataframe
    model.fit(combined_df[features], combined_df[label])

    # Make predictions on the combined dataframe
    predictions = model.predict(combined_df[features])

    # Calculate residuals
    combined_df['residuals'] = combined_df[label] - predictions
    Y = combined_df['residuals']
    X = combined_df[features]

    # Append residuals to each split dataframe in-place
    for df_subset in split_dfs:
        matching_indices = df_subset.index
        df_subset['residuals'] = combined_df.loc[matching_indices, 'residuals'].values
        df_subset.drop(columns=[label], inplace=True)

    assert np.allclose(Y, np.concatenate([df_subset['residuals'] for df_subset in split_dfs]))

    # Extract coefficients for features of interest
    coefficients = {feature: model.coef_[i] for i, feature in enumerate(features)}

    return coefficients

def normalize_and_split_dataframes(
    dataframes: List[pd.DataFrame], feature_of_interest: str, sign: float = -1
) -> Tuple[List[np.array], List[np.array], np.ndarray]:
    """
    Concatenates a list of dataframes, normalizes the features, splits them back into the original structure,
    and computes a normalized axis of interest.

    Parameters:
    dataframes (List[pd.DataFrame]): A list of dataframes to be concatenated and normalized.
    axis_of_interest (str): The name of the column that is of special interest, which will be normalized separately.

    Returns:
    Tuple[List[pd.DataFrame], List[pd.Series], np.ndarray]: A tuple containing the list of normalized feature dataframes (split_X),
     the list of residual series (split_R), and the normalized axis of interest.
    """
    # Concatenate all dataframes
    all_data = pd.concat(dataframes, ignore_index=True)

    # Separate features (X) and residuals (R)
    features = all_data.drop(columns=['residuals'])
    residuals = sign * all_data['residuals']

    # Compute Sigma and its square root
    feature_covariance = features.T @ features
    root_feature_covariance = scipy.linalg.sqrtm(feature_covariance)

    # Normalize X
    normalized_features = features @ np.linalg.inv(root_feature_covariance)

    # Split normalized X and Y back into the original dataframe structure
    split_X, split_R = [], []
    start_idx = 0
    for df in dataframes:
        end_idx = start_idx + len(df)
        split_X.append(normalized_features.iloc[start_idx:end_idx].to_numpy())
        split_R.append(residuals.iloc[start_idx:end_idx].to_numpy())
        start_idx = end_idx

    # Create an unnormalized axis of interest
    axis_of_interest = (features.columns == feature_of_interest).astype(int)

    # Normalize the axis of interest
    axis_of_interest_normalized = np.linalg.inv(root_feature_covariance) @ axis_of_interest

    return split_X, split_R, axis_of_interest_normalized


def full_enumeration_bucket(
    X: pd.DataFrame, R: pd.Series, axis_of_interest_normalized: np.ndarray, k: int, verbose: bool = True
) -> (float, list):
    """
    Calculate the highest score and the set achieving this score for a bucket of data,
    taking into account a normalized axis of interest.

    Parameters:
    X (pd.DataFrame): The feature dataframe for a specific bucket.
    R (pd.Series): The residuals series for the same bucket.
    axis_of_interest_normalized (np.ndarray): The normalized axis of interest.
    k (int): The parameter of the algorithm to determine the set size as n - k.

    Returns:
    (float, list): The highest score and the list of indices in the bucket that achieve this highest score.
    """
    n = len(X)

    # Convert the normalized axis of interest to a pandas Series for easy indexing
    Z_normalized = pd.Series(axis_of_interest_normalized.flatten(), index=X.columns)
    combs = list(combinations(range(n), n - k))
    scores = []
    highest_score = -np.inf
    best_set = None

    for i, subset_indices in enumerate(tqdm.tqdm(combs, disable=not verbose)):
        R_subset = R.iloc[list(subset_indices)]
        Z_subset = X.iloc[list(subset_indices)].dot(Z_normalized)

        A_S = (Z_subset * R_subset).sum()
        B_S = Z_subset.sum()
        C_S = R_subset.mean()

        score = A_S + B_S * C_S
        # scores.append(score)
        if score > highest_score:
            highest_score = score
            best_set = list(subset_indices)

    highest_score = max(scores)
    best_set = list(combs[np.argmax(scores)])
    return highest_score, best_set


def calculate_bounds_efficient(
    X: np.ndarray, R: np.ndarray, axis_of_interest_normalized: np.ndarray
) -> np.ndarray:
    """
    We define the score of a bucket / set S to be
        score(bucket B, set of retained samples S) = negative contribution to Delta =
        = -sum_{i in S intersect B} Z_i (R_i - E_{j in S intersect B} R_j) =
        = sum_{i in T intersect B} Z_i (R_i + frac{1}{abs{S intersect B}} sum_{j in T intersect B} R_j)

    It turns out that it is enough to give what seems like a fairly loose bound which can be computed very efficiently.
    For each of the following values:
        A = sum_{i in T intersect B} R_i Z_i
        B = sum_{i in T intersect B} Z_i
        C = sum_{i in T intersect B} R_i
    we bound it from above and below by taking the k largest / smallest values in our bucket.
    We bound B*C from above by max(B_min * C_min, B_max * C_max) and then output an upper bound on:
        A + frac{1}{abs{S intersect B}} (B*C)

    We do all of this numpy vectorized for all values of abs{T intersect B} = k = 0, ..., abs{B}.

    Parameters:
    X (pd.DataFrame): The feature dataframe for a specific bucket.
    R (pd.Series): The residuals series for the same bucket.
    axis_of_interest_normalized (np.ndarray): The normalized axis of interest.

    Returns:
    pd.Series: A series containing the bound for each k.
    """
    n, d = X.shape
    Z = X.dot(axis_of_interest_normalized)
    RZ = R * Z

    # Sort and cumulative sum
    sorted_RZ = np.sort(RZ)
    sorted_Z = np.sort(Z)
    sorted_R = np.sort(R)

    cumsum_RZ_rev = np.cumsum(sorted_RZ[::-1])
    cumsum_Z_rev = np.cumsum(sorted_Z[::-1])
    cumsum_R_rev = np.cumsum(sorted_R[::-1])
    cumsum_Z = np.cumsum(sorted_Z)
    cumsum_R = np.cumsum(sorted_R)

    ks = np.arange(1, n)
    A_max = cumsum_RZ_rev[:-1]
    B_max = cumsum_Z_rev[:-1]
    B_min = cumsum_Z[:-1]
    C_max = cumsum_R_rev[:-1]
    C_min = cumsum_R[:-1]
    bounds = A_max + (np.maximum(B_max * C_max, B_min * C_min) / (n - ks))
    bounds = np.concatenate((np.zeros(1), bounds, np.zeros(1)))

    return bounds


def compute_bounds_for_all(
    split_X: List[np.ndarray], split_R: List[np.ndarray],
    axis_of_interest_normalized: np.ndarray, verbose: bool = True
) -> List[np.ndarray]:
    """
    We define the score of a bucket / set S to be
        score(bucket B, set of retained samples S) = negative contribution to Delta =
         = -sum_{i in S intersect B} Z_i (R_i - E_{j in S intersect B} R_j) =
         = sum_{i in T intersect B} Z_i (R_i + frac{1}{abs{S intersect B}} sum_{j in T intersect B} R_j)
    For each bucket B_j, we bound the maximal score, conditioned on the size of k_j = abs{T intersect B}.

    Parameters:
    split_X (List[pd.DataFrame]): A list of split feature dataframes, each corresponding to a different bucket.
    split_R (List[pd.Series]): A list of split residuals series, each corresponding to the same bucket as in split_X.
    axis_of_interest_normalized (np.ndarray): The normalized axis of interest.

    Returns:
    Dict[int, pd.Series]: A dictionary where each key is the index of the split dataframe/bucket and the value is a pandas Series
                          containing the bound for each k for that dataframe/bucket.
    """
    bounds_list = []

    for i, (X, R) in enumerate(zip(
            tqdm.tqdm(split_X, desc="Bounding influence by category", disable=not verbose),
            split_R
    )):
        bounds = calculate_bounds_efficient(X, R, axis_of_interest_normalized)
        bounds_list.append(bounds)

    return bounds_list


def maximize_total_bound_with_score(
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


def compute_XZ_XR_categorical(
    split_X: List[np.ndarray], split_R: List[np.ndarray], axis_of_interest: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the XZ and XR arrays by scaling feature vectors in each bucket based on their inner product with
    the axis of interest and their residuals respectively.

    Parameters:
    - split_X: List of np.ndarray, where each array represents feature vectors (X_i) in a bucket.
    - split_R: List of np.ndarray, where each array represents residuals (R_i) in the same buckets as split_X.
    - axis_of_interest: np.ndarray, a d-dimensional vector representing a linear combination of features.

    Returns:
    - XZ: np.ndarray, 2n x d array representing the effect of rescaling samples by their value on the axis of interest.
    - XR: np.ndarray, 2n x d array representing the effect of rescaling samples by their residuals.
    """
    # Initialize empty lists to hold the scaled feature vectors
    XZ_list = []
    XR_list = []

    # Iterate over each bucket
    for X, R in zip(split_X, split_R):
        # Compute Z_i for each sample in the bucket
        Z = np.dot(X, axis_of_interest)
        R = np.array(R)

        # Find the smallest and largest values of Z and R in the bucket
        min_Z, max_Z = Z.min(), Z.max()
        min_R, max_R = R.min(), R.max()

        # Compute the scaled feature vectors and append to the lists
        XZ_list.append(X * (Z[:, np.newaxis] - max_Z))  # Rescale by (Z_i - max_Z)
        XZ_list.append(X * (Z[:, np.newaxis] - min_Z))  # Rescale by (Z_i - min_Z)

        XR_list.append(X * (R[:, np.newaxis] - max_R))  # Rescale by (R_i - max_R)
        XR_list.append(X * (R[:, np.newaxis] - min_R))  # Rescale by (R_i - min_R)

    # Concatenate all scaled feature vectors to form the final arrays
    XZ = np.vstack(XZ_list)
    XR = np.vstack(XR_list)

    return XZ, XR

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



def ku_triangle_inequality(
        gram_matrix: np.ndarray, bucket_sizes: List[int], k_max: int, u_max: int,
        verbose: bool = True
) -> np.ndarray:
    """
    Goal: bound the norm of the sum of k vectors take from u unique buckets using triangle inequality logic.
    Method:
        For each vector (represented by a row in the Gram matrix), we divide its entries between:
        - a set of m=num_buckets entries representing the largest entry in each bucket.
        - a set of n-m entries representing entries that are *not* largest within their respective buckets.
        We can then bound:
        Norm{sum_i X_i}^2 = sum_{i1, i2} <X_i1, X_i2> =
        = sum_{i1 in T intersect B_j_l_1, i2 in T intersect B_j_l_2}

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
    for j in tqdm.trange(m, desc="KU Triangle Inequality - splitting rows", disable=not verbose):
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
        min(u_max, k_max), desc="KU Triangle Inequality - combining contributions", disable=not verbose
    ):
        sample_contributions[:, u, u:k_max] = best_u_contributions[:, u][:, np.newaxis] + best_kmu_contributions[:, :k_max - u]

    # We now perform a similar trick to enforce the fact that T must utilize exactly $u$ separate buckets.
    # For each bucket, we take its highest contributing member (for each k,u) and remove it from the main list of possible
    # contributions to choose from. When optimizing T we can only take a summation of exactly u elements from the "best_contributions"
    # and k - u elements from the rest.
    best_contributions = np.zeros((m, u_max, k_max))
    for j in tqdm.trange(m, desc="KU Triangle Inequality - splitting columns", disable=not verbose):
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
    cumsum_best_contributions = np.concatenate((np.zeros((1, u_max, k_max)), cumsum_best_contributions))


    # We never look at the sample_contributions beyond the k_max largest at any index, so we may as well drop those indices
    # sample_contributions are still logically indexed as sample index "i" or "n" (now simply capped at k_max), u, k.
    # sample_contributions = -np.partition(-sample_contributions, k_max, axis=0)[:k_max, :, :]

    # Initialize norms_squared with -np.inf
    norms_squared = np.full((u_max, k_max), -np.inf)
    for u in tqdm.trange(u_max, desc="KU Triangle Inequality - combining results", disable=not verbose):
        for k in range(k_max):
            if k >= u:
                # Use partition to bring top k-u elements to the front
                sample_contributions[:, u, k] = -np.partition(-sample_contributions[:, u, k], k - u, axis=0)
                # Result of calculation: sum over k-u larges elements of sample_contributions and u largest elements of best_contributions
                # print(f"{u=}, {k=}")
                # print(f"{norms_squared[u, k]}")
                norms_squared[u, k] = np.sum(sample_contributions[:k - u, u, k]) + cumsum_best_contributions[u, u, k]

    # # Iterate over k since we need to partition and sum the top k-u elements
    # for k in tqdm.trange(k_max, desc="KU Triangle Inequality - combining results", disable=not verbose):
    #     # For each k, work on all u where u <= k (to satisfy k >= u)
    #     u_indices = np.arange(min(k + 1, u_max))
    #
    #     # Partition sample_contributions to bring the top k-u elements to the front
    #     partition_indices = np.argpartition(-sample_contributions[:, u_indices, k], k - u_indices, axis=0)[:k - u_indices]
    #
    #     # Sum over the top k-u elements for sample_contributions
    #     cumsum_samples = np.cumsum(sample_contributions[partition_indices, u_indices, k], axis=0)
    #     cumsum_samples = np.concatenate((np.zeros((1, u_max, k_max))))
    #
    #     # Add the sum of the u largest elements from cumsum_best_contributions, which is simply cumsum_best_contributions[u]
    #     norms_squared[u_indices, k] = cumsum_samples + cumsum_best_contributions[u_indices]


    # # Sort the sample_contributions so that we can easily access the sums over their largest members:
    #
    # # Step 1 - Transpose sample_contributions to bring k to the first axis
    # scores_transposed = np.transpose(sample_contributions, (2, 1, 0))  # New shape (k_max, u_max, n)
    #
    # # Step 2 - Partition on contiguous blocks of memory
    # for k in tqdm.trange(
    #     k_max, desc="KU Triangle Inequality Partition", disable=not verbose
    # ):
    #     scores_transposed[k, :, :] = -np.partition(-scores_transposed[k, :, :], k, axis=-1)
    #
    # # Step 3 - Transpose them back
    # sample_contributions = np.transpose(scores_transposed, (2, 1, 0))  # Back to (n, u_max, k_max)
    #
    # # We never touch the sample_contributions beyond the k_max entry.
    # sample_contributions = sample_contributions[:k_max+1, :, :]
    #
    # cumsum_scores = np.cumsum(sample_contributions, axis=0)
    # cumsum_scores = np.concatenate((np.zeros((1, u_max, k_max)), cumsum_scores), axis=0)
    # # Create an array of 'k' indices (for the first dimension of cumsum_scores)
    # k_indices = np.arange(k_max)
    # norms_squared = cumsum_scores[k_indices, np.arange(u_max)[:, None], k_indices]
    # Ignore warning about -np.infs in sqrt as these will be removed
    with np.errstate(invalid='ignore'):
        norms = np.sqrt(norms_squared)
        norms = np.nan_to_num(norms, nan=-np.inf)
    return norms


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
        maximize_total_bound_with_score(
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
            maximize_total_bound_with_score(
                bounds_list=[
                    quadratic_score +
                    linear_score * (bound + quadratic_score_total[k - 1])
                    for quadratic_score, linear_score in
                    zip(quadratic_scores, linear_scores)
                ], k_max=k + 1, verbose=verbose
            )[k]
        )
    return bound