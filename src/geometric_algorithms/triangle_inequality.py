import numpy as np
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

    # Step 3: Prepend zeros and then compute cumulative sums over rows
    cumsum_rows = np.cumsum(np.hstack((np.zeros((n, 1)), sorted_rows)), axis=1)

    # Step 4: Add diagonal elements to each row
    adjusted_rows = cumsum_rows + diag_elements[:, np.newaxis]

    # Step 5: Compute the vector of k largest elements sum for each column and take the square root
    result_vector = np.sqrt([np.sum(-np.partition(-adjusted_rows[:, k-1], k)[:k])
                             for k in tqdm.trange(1, n, desc="Compiling triangle bounds", disable=not verbose)])

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