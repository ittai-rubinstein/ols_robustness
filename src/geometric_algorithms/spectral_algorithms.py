import numpy as np


def spectral_bound_sum_ips(
        gram_matrix: np.ndarray,
        d: int,
        k_vals: np.ndarray = None,
        eigvals: np.ndarray = None,
        eigvecs: np.ndarray = None
) -> np.ndarray:
    """
    Compute an upper bound on the L2 squared norm of the sum of any k rows of the input array
    using spectral decomposition.

    Args:
        gram_matrix (np.ndarray): The Gram matrix of an array of vectors.
        d (int): The number of dimensions to consider in the spectral decomposition.
        k_vals (np.ndarray): An array of integer indices k. Defaults to np.arange(1, n // 10).
        eigvals (np.ndarray): Precomputed eigenvalues of the outer product matrix. Optional.
        eigvecs (np.ndarray): Precomputed eigenvectors of the outer product matrix. Optional.

    Returns:
        np.ndarray: A vector where each entry corresponds to an upper bound on the L2 norm
                    for the corresponding value of k in k_vals.

    Methodology:
        1. Perform eigenvalue decomposition of the outer product matrix if not provided.
        2. Calculate bounds on the coefficients in the eigenbasis.
        3. Compute the upper bound on the L2 norm squared.
    """
    n = gram_matrix.shape[0]
    if k_vals is None:
        k_vals = np.arange(1, n)

    if eigvals is None or eigvecs is None:
        # Perform eigenvalue decomposition if not provided
        eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)
    else:
        # Use provided eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigvals, eigvecs
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

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

    return np.sqrt(l2_norm_squared_bound)


import numpy as np
import tqdm


def spectral_bound_sum_ips_improved(
        gram_matrix: np.ndarray,
        d: int,
        k_vals: np.ndarray = None,
        eigvals: np.ndarray = None,
        eigvecs: np.ndarray = None
) -> np.ndarray:
    n = gram_matrix.shape[0]
    d = min(d + 1, n)
    if k_vals is None:
        k_vals = np.arange(1, n)

    if eigvals is None or eigvecs is None:
        eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)
    else:
        eigenvalues, eigenvectors = eigvals, eigvecs
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    # print(f"{eigenvectors=}")

    sorted_eigenvectors = np.sort(eigenvectors[:, :d], axis=0)
    # print(f"{sorted_eigenvectors=}")
    min_sums = np.cumsum(sorted_eigenvectors[:, :], axis=0)
    max_sums = np.cumsum(sorted_eigenvectors[::-1, :], axis=0)

    # Compute max_abs_sums[i, k] = max_{S_i subset of [n] with k members} \abs{\sum_{j \in S} v_{i, j}}}
    max_abs_sums = np.maximum(max_sums, -min_sums).T
    # print(f"{max_abs_sums=}")

    # print(f'{max_abs_sums.shape=}')

    i_max = d
    j_max = n
    k_max = max(k_vals) + 1
    # print(f'{i_max=}, {j_max=}, {k_max=}')

    # Compute max_entry_influence[i, j, k] = abs{v_ij} * max_abs_sums[i, k]
    max_entry_influence = np.abs(eigenvectors.T[:d, :, None]) * max_abs_sums[:, None, :]
    # print(f'{max_entry_influence.shape=}')
    # print(f'{max_entry_influence=}')

    # Compute cumulative max influence = cumulative sum over eigenvector index i of upper bound on
    # contribution of j being in S to |<1_S, v_i>|^2
    cumulative_max_influences = np.cumsum(max_entry_influence, axis=0)
    # print(f'{cumulative_max_influences.shape=}')
    # print(f'{cumulative_max_influences=}')

    # Upper bound on max_{S subset of [n] of size k} sum_{i' <= i, j in S} of contribution of
    # j being in S to |<1_S, v_i>|^2 >= max_{S subset of [n] of size k} sum_{i' <= i} |<1_S, v_i>|^2
    max_total_influence = np.zeros((i_max, k_max))
    for k in tqdm.trange(k_max):
        partitioned_influences = -np.partition(-cumulative_max_influences[:, :, k], k, axis=1)
        max_total_influence[:, k] = np.sum(partitioned_influences[:, :k + 1], axis=1)
    # print(f'{max_total_influence.shape=}')
    # print(f'{max_total_influence=}')

    # Compute the cutoff index i_k = largest i for which we can use the bounds above to certify that
    # sum_{i' <= i} |<1_S, v_i>|^2 <= k
    # for any set of size |S| = k.
    cutoff_indices = np.array([
        np.max(np.arange(i_max)[max_total_influence[:, k] <= (k + 1)]) if np.any(
            max_total_influence[:, k] <= (k + 1)) else min(d, n - 1)
        for k in range(k_max)
    ])
    # print(f'{cutoff_indices=}')

    # Compute the eigenvalues at the cutoff. These are the smallest eigenvalues we cannot certify are not in play.
    cutoff_eigvals = eigenvalues[cutoff_indices]
    # print(f'{cutoff_eigvals=}')

    # Compute l_ik = lambda_i - lambda_(cutoff_k)
    shifted_eigenvalues = eigenvalues[:d, np.newaxis, np.newaxis] - cutoff_eigvals[np.newaxis, np.newaxis, :]
    # print(f'{shifted_eigenvalues=}')

    # We now want to select S of size k (where k is a parameter), which maximizes
    # sum_{i < cutoff_k, j in S} l_ik * (max_{S_i} | sum_{j' in S_i} v_{i,j'}|) * abs{v_{i, j}
    # max_entry_influence[i, j, k] = abs{v_ij} * max_abs_sums[i, k]
    lik_max_abs_sum_abs = shifted_eigenvalues * max_entry_influence
    # print(f'{max_entry_influence=}')
    # print(f'{lik_max_abs_sum_abs=}')
    cumulative_lik_max_abs_sum_abs = np.cumsum(lik_max_abs_sum_abs, axis=0)
    # Compute an array of sum_{i < cutoff_k} l_ik * (max_{S_i} | sum_{j' in S_i} v_{i,j'}|) * abs{v_{i, j}
    # as a function of j and k:
    sum_lik_max_abs_sum_abs = np.concatenate(
        [
            cumulative_lik_max_abs_sum_abs[cutoff, :, k][:, np.newaxis]
            for k, cutoff in enumerate(cutoff_indices)
        ], axis=1
    )
    sorted_sum_lik_max_abs_sum_abs = -np.sort(-sum_lik_max_abs_sum_abs, axis=0)
    max_sum_lik_max_abs_sum_abs = np.array(
        [np.sum(sorted_sum_lik_max_abs_sum_abs[:k + 1, k]) for k in range(k_max - 1)]
    )
    # Compute our spectral bound on the l2 squared norms:
    l2_norm_squared_bounds = (k_vals * cutoff_eigvals[:-1]) + max_sum_lik_max_abs_sum_abs
    # Take the square root
    return np.sqrt(l2_norm_squared_bounds)


def spectral_bound_sum_ips_squared(
        gram_matrix: np.ndarray,
        d: int,
        k_vals: np.ndarray = None
) -> np.ndarray:
    """
    An algorithm for bounding the maximum over unit vector v and subset T of size k out of n rows of X of:
            sum of ips squared (T, v) = sum_{i in S} <X_i, v>^2
    As with our analysis of triangle inequality for ips squared, let Sigma_T be defined as: sum_{i in T} X_i X_i ^T
    Therefore, sum of ips squared (T, v) = v^T Sigma_S v = sum_{i in T} v^T X_i X_i^T v.
    We can use a series of relaxations:
    v^T Sigma v <= (CS inequality, Norm{v} = 1) <= max{lambda(Sigma)} = (lambda(Sigma) = lambda(G)) = max{lambda(G)}
    where G_ij = <X_i, X_j> (for i, j in T) is the Gram matrix of X limited to T
    max{lambda(G)} <= (CS inequality) <= sqrt{sum_{i, j in T} G_{i,j}^2}

    Here we simply change the next relaxation to be that
    sqrt{sum_{i, j in T} G_{i,j}^2} = 1_T ^T H 1_T <= Spectral(H, |T|)
    where H_ij = <X_i, X_j>^2 for all i, j in [n].
    Args:
        gram_matrix (np.ndarray): The Gram matrix of an array of vectors.
        d (int): The number of dimensions to consider in the spectral decomposition.
        k_vals (np.ndarray): An array of integer indices k. Defaults to np.arange(1, n // 10).
    :return: Upper bound on max_{v, T} \sum_{i \in T} <X_i, v>^2 as a function of |T| = k, for k \in k_vals.
    """
    return spectral_bound_sum_ips(gram_matrix=np.abs(gram_matrix)**2, d=d, k_vals=k_vals)