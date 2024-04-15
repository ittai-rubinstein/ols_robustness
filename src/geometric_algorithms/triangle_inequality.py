from typing import List
from matplotlib import pyplot as plt
import numpy as np
from scipy.linalg import sqrtm
import tqdm

def refined_triangle_inequality_ips(gram_matrix: np.ndarray, verbose: bool = True) -> np.ndarray:
    """
    An algorithm for bounding the l2 norm of any subset of size k of out a set of vectors.
    The algorithm takes as input the Gram matrix, and utilizes the observation that for any set T of samples,
     the l2 norm squared of the sum of these samples is given by the following expression which can be relaxed as follows:
        l2 norm squared = max_T sum_{i, j in T} <X_i, X_j> <= (relaxation)
                        <= max_{T, S_i} sum_{i in T, j in S_i} <X_i, X_j>
    This relaxed version of the problem is then solved with a greedy algorithm.
    :param gram_matrix: A matrix whose i,j th entry is the inner product between X_i and X_j.
    :return: A vector of length n-1, whose k th entry is an upper bound on the l2 norm of the sum of any k
            vectors in X (where k goes from 1 to n).
    """
    n = gram_matrix.shape[0]
    gram_matrix = np.copy(gram_matrix)

    # Step 1: Copy and zero out the diagonal
    diag_elements = np.diag(gram_matrix).copy()
    np.fill_diagonal(gram_matrix, 0)

    # Step 2: Sort each row
    sorted_rows = -np.sort(-gram_matrix, axis=1)
    # Save memory
    del gram_matrix

    # Step 3: Prepend zeros and then compute cumulative sums over rows
    cumsum_rows = np.cumsum(np.hstack((np.zeros((n, 1)), sorted_rows)), axis=1)
    del sorted_rows

    # Step 4: Add diagonal elements to each row
    adjusted_rows = cumsum_rows + diag_elements[:, np.newaxis]
    del cumsum_rows

    # Step 5: Compute the vector of k largest elements sum for each column and take the square root
    result_vector = np.sqrt([np.sum(-np.partition(-adjusted_rows[:, k - 1], k)[:k])
                             for k in tqdm.trange(1, n, desc="Triangle Inequality", disable=not verbose)])

    return result_vector


def refined_triangle_inequality_ips_squared(gram_matrix: np.ndarray, verbose: bool = True) -> np.ndarray:
    """
    An algorithm for bounding the maximum over unit vector v and subset S of size k out of n rows of X of:
            sum of ips squared (S, v) = sum_{i in S} <X_i, v>^2
    Let Sigma_T be defined as: sum_{i in T} X_i X_i ^T
    Therefore, sum of ips squared (T, v) = v^T Sigma_S v = sum_{i in T} v^T X_i X_i^T v.
    We can use a series of relaxations:
    v^T Sigma v <= (CS inequality, Norm{v} = 1) <= max{lambda(Sigma)} = (lambda(Sigma) = lambda(G)) = max{lambda(G)}
    where G_ij = <X_i, X_j> (for i, j in T) is the Gram matrix of X limited to T
    max{lambda(G)} <= (CS inequality) <= sqrt{sum_{i, j in T} G_{i,j}^2}

    From here we can apply the relaxation:
    max_T{sqrt{sum_{i, j in T} G_{i,j}^2}} <= max_{T, T_1, ..., T_k} sqrt{sum_{i in T, j in T_i} G_{i, j}^2}

    :param gram_matrix: A matrix whose i,j th entry is the inner product between X_i and X_j.
    :return: A vector of length n-1, whose k th entry is an upper bound on the sum of inner products squared of any vector v
            with any k vectors in X (where k goes from 1 to n).
    """
    return refined_triangle_inequality_ips(np.abs(gram_matrix)**2, verbose=verbose)


def refined_triangle_inequality_indices(
gram_matrix: np.ndarray, verbose: bool = False
) -> List[np.ndarray]:
    """
    An algorithm for identifying the indices of a subset of vectors that contribute to the upper bound
    on the l2 norm of the sum of any subset of size k out of a set of vectors.

    :param gram_matrix: A matrix whose i, j-th entry is the inner product between X_i and X_j.
    :return: A list of arrays, where the k-th array contains indices of the vectors contributing to the
             upper bound on the l2 norm of the sum of any k vectors in X (where k goes from 1 to n).
    """
    n = gram_matrix.shape[0]
    gram_matrix = np.copy(gram_matrix)

    # Step 1: Copy and zero out the diagonal
    diag_elements = np.diag(gram_matrix).copy()
    np.fill_diagonal(gram_matrix, 0)

    # Step 2: Sort each row
    sorted_rows = -np.sort(-gram_matrix, axis=1)

    # Step 3: Prepend zeros and then compute cumulative sums over rows
    cumsum_rows = np.cumsum(np.hstack((np.zeros((n, 1)), sorted_rows)), axis=1)

    # Step 4: Add diagonal elements to each row
    adjusted_rows = cumsum_rows + diag_elements[:, np.newaxis]

    # Initialize a list to store the indices for each k
    indices_list = []

    # Step 5: Find the indices of the k largest elements sum for each column
    for k in tqdm.trange(1, n, desc="Compiling triangle bounds", disable=not verbose):
        indices = np.argpartition(-adjusted_rows[:, k - 1], k)[:k]
        indices = np.argsort(-adjusted_rows[:, k-1])[:k]
        # indices_sorted_by_value = indices[np.argsort(-adjusted_rows[indices, k - 1])]
        indices_list.append(indices)

    return indices_list



def test_refined_triangle_inequality_indices(n=1000, d=100, special_ratio=0.1, correlation_threshold=0.9):
    def generate_vectors():
        num_special = int(n * special_ratio)
        normal_vectors = np.random.normal(0, 1, (n - num_special, d))
        special_vectors = np.zeros((num_special, d))
        special_vectors[:, 0] = 1
        special_vectors[:,1:] = (0.1 / np.sqrt(d))*np.random.normal(0, 1, (num_special, d-1))
        X = np.vstack((special_vectors, normal_vectors))
        shuffle_indices = np.random.permutation(n)
        X = X[shuffle_indices, :]
        original_indices = np.arange(n)
        original_indices[shuffle_indices] = np.arange(n)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        covariance_matrix = X.T @ X
        X = X @ sqrtm(np.linalg.inv(covariance_matrix))
        special_indices = original_indices < num_special
        i = np.random.randint(n)
        assert i == original_indices[shuffle_indices[i]]
        return X, special_indices

    def check_correlation(indices_list, special_indices):
        correlations = []
        for k in range(1, int(0.1 * n) + 1):
            selected_indices = indices_list[k-1]
            fraction_special = np.mean(special_indices[selected_indices])
            correlations.append(fraction_special)
        return correlations
        # return np.mean(correlations) >= correlation_threshold

    # Generate the vectors and their labels
    X, special_indices = generate_vectors()

    # spec_inds = np.arange(X.shape[0])[special_indices]
    # plt.hist((X[spec_inds, :].T @ X[spec_inds, :]).ravel(), bins=100, alpha=0.7)
    # spec_inds = np.arange(X.shape[0])[np.logical_not(special_indices)]
    # plt.hist((X[spec_inds, :].T @ X[spec_inds, :]).ravel(), bins=100, alpha=0.7)


    # Compute the Gram matrix
    gram_matrix = X @ X.T

    # Run the modified function
    indices_list = refined_triangle_inequality_indices(gram_matrix)

    # Test for correlation
    correlations = check_correlation(indices_list, special_indices)
    # plt.plot(correlations)
    # print(f"Is the correlation above {correlation_threshold * 100}%? {'Yes' if is_correlated else 'No'}")