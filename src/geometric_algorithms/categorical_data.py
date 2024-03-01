from itertools import combinations
import tqdm
import pandas as pd
import numpy as np
import scipy.linalg
from typing import List, Tuple, Dict
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
    bounds = np.concatenate(([0.], bounds, [0.]))

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
bounds_list: List[np.ndarray], k_max: int, use_tqdm: bool = True
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
    :param use_tqdm: Whether to include a tqdm progress bar

    Returns:
    An array whose kth element is a bound on the total effect of k removals.
    """
    # Assuming bounds_dict is available and k_total is defined
    n = len(bounds_list)
    k_max += 1  # Adjusting for 0-based indexing in NumPy arrays

    # Initialize the DP table with -inf, and set the first column to 0
    dp = np.full((n + 1, k_max), -np.inf)

    # Fill the DP table
    for i in tqdm.trange(1, n + 1, disable = not use_tqdm,
                         desc="Dynamic programming"):
        bounds = bounds_list[i - 1]
        dp[i, :len(bounds)] = bounds[:k_max]

        for k in range(1, k_max):
            # Vectorized update for dp[i, k]
            # Create a mask for valid updates to avoid out-of-bounds and unnecessary computations
            valid_updates = np.arange(max(0, k - len(bounds) + 1), k + 1, dtype=int)
            if valid_updates.size > 0:
                dp[i, k] = max(dp[i, k], np.max(dp[i - 1, valid_updates] + bounds[k - valid_updates]))

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
