import numpy as np


def spectral_decomposition_bound(
        outer_product: np.ndarray,
        d: int,
        k_vals: np.ndarray = None,
        eigvals: np.ndarray = None,
        eigvecs: np.ndarray = None
) -> np.ndarray:
    """
    Compute an upper bound on the L2 squared norm of the sum of any k rows of the input array
    using spectral decomposition.

    Args:
        outer_product (np.ndarray): The outer product matrix of an array of vectors.
        d (int): The number of dimensions to consider in the spectral decomposition.
        k_vals (np.ndarray): An array of integer indices k. Defaults to np.arange(1, n // 10).
        eigvals (np.ndarray): Precomputed eigenvalues of the outer product matrix. Optional.
        eigvecs (np.ndarray): Precomputed eigenvectors of the outer product matrix. Optional.

    Returns:
        np.ndarray: A vector where each entry corresponds to an upper bound on the L2 squared norm
                    for the corresponding value of k in k_vals.

    Methodology:
        1. Perform eigenvalue decomposition of the outer product matrix if not provided.
        2. Calculate bounds on the coefficients in the eigenbasis.
        3. Compute the upper bound on the L2 norm squared.
    """

    if k_vals is None:
        k_vals = np.arange(1, (outer_product.shape[0] // 10) + 1)

    if eigvals is None or eigvecs is None:
        # Perform eigenvalue decomposition if not provided
        eigenvalues, eigenvectors = np.linalg.eigh(outer_product)
    else:
        # Use provided eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigvals, eigvecs
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # sort the entries in each eigenvector:
    sorted_eigenvectors = np.sort((eigenvectors[:, :d]), axis=0)

    # The ith coefficient is a sum of k entries of its vector. Therefore, it is bounded from above and below by:
    min_sums = np.cumsum(sorted_eigenvectors, axis=0)
    max_sums = np.cumsum(sorted_eigenvectors[::-1], axis=0)

    # We want to bound abs(coeff)**2 <= max(abs(lower_bound)**2, abs(upper_bound)**2)
    coefficient_bounds = np.maximum(np.abs(min_sums) ** 2, np.abs(max_sums) ** 2)[k_vals, :]

    # Compute the cumulative sums over the coefficient bounds
    cumulative_sums = np.cumsum(np.pad(coefficient_bounds, ((0, 0), (1, 0)), mode='constant', constant_values=0),
                                axis=1)

    # Further limit the coefficient bounds to sum up to at most k:
    constrained_bounds = np.maximum(np.minimum(coefficient_bounds, k_vals[:, np.newaxis] - cumulative_sums[:, :-1]), 0)

    # Compute the output
    l2_norm_squared_bound = constrained_bounds @ eigenvalues[:d]

    return l2_norm_squared_bound