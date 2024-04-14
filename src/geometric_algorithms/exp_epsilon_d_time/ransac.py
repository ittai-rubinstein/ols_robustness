import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from tqdm import trange

from src.utils.binary_search import BinarySearchFunctor


def compute_smallest_eigenvalue(matrix: np.ndarray) -> float:
    """
    Computes the smallest eigenvalue of a positive semi-definite matrix.

    Parameters:
    matrix (np.ndarray): A positive semi-definite matrix.

    Returns:
    float: The smallest eigenvalue of the matrix.
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    return np.min(eigenvalues)


def compute_Sigma(X: np.ndarray, indices: np.ndarray) -> np.ndarray:
    # Selecting the subset of X
    X_prime = X[indices, :]
    # Extract the total number of samples (used for normalization) and the dimension of the samples based on X
    num_samples, dimension = X.shape
    # Extract the k parameter of the ransac alg from the number of indices.
    ransac_k = len(indices)

    # Computing the covariance matrix of X_prime
    # Sigma = np.cov(X_prime, rowvar=True)
    mu = np.average(X_prime, axis=0)
    X_prime = X_prime - np.reshape(mu, (1, dimension))
    # Correct the cov matrix by adding a small factor of I (to avoid numerical issues), and rescaling by the number of samples.
    Sigma = (X_prime.transpose() @ X_prime) * (num_samples / ransac_k) + (1e-5 * np.eye(X.shape[1]))
    return Sigma


@dataclass
class ScoringResult:
    e: np.ndarray # Our axis of interest (eg democracy)
    is_close: bool # Can we approximate the axis of interest with our samples (if not, then this means our error bar is infinite).
    e_Sigma_inv_e: float # Estimate of the error bar
    Sigma_inv_e: np.ndarray # Estimate of direction in which our fit can vary to cause errors
    indices: np.ndarray # The indices of the samples used in this
    smallest_eigenvalue: float # The smallest eigenvalue of Sigma
    norm_sigma_e: float # The norm of Sigma e.


def scoring_function(X: np.ndarray, Y: np.ndarray, indices: np.ndarray, e: np.ndarray = None) -> ScoringResult:
    """
    Calculates the scoring for detecting outliers in a linear regression model.

    Parameters:
    X (np.ndarray): A d x n array of feature vectors.
    Y (np.ndarray): An n-dimensional array of labels.
    indices (np.ndarray): Array of indices of samples to consider.
    e (np.ndarray): Axis of interest, defaulting to the first primary axis.

    Returns:
    tuple: A tuple containing a boolean indicating if e is close to Sigma @ f,
           the inner product of f and e, and the vector f.
    """
    if e is None:
        e = np.zeros(X.shape[1])
        e[0] = 1

    Sigma = compute_Sigma(X, indices)
    num_samples = len(Y)
    k = len(indices)

    f, _, _, _ = np.linalg.lstsq(Sigma, e, rcond=None)

    # Compute the smallest eigenvalue of Sigma
    min_eigenvalue = compute_smallest_eigenvalue(Sigma)

    # Outputs
    is_close = np.allclose(e, Sigma @ f, atol=1e-5)
    inner_product = np.dot(f, e)

    return ScoringResult(
        e=e,
        is_close=is_close,
        e_Sigma_inv_e=inner_product,
        Sigma_inv_e=f,
        indices=indices,
        smallest_eigenvalue=min_eigenvalue,  # Include the smallest eigenvalue in the result
        norm_sigma_e= np.linalg.norm((Sigma @ e))
    )

# A dataclass for storing the results of running the ransac algorithm
@dataclass
class RANSACOutput:
    ransac_results: List[ScoringResult] # The outputs of the runs of the RANSAC algorithm
    ransac_samples: int
    ransac_iterations: int
    total_samples: int
    dimension: int

    def __len__(self):
        return len(self.ransac_results)

    def __iter__(self):
        return iter(self.ransac_results)

def ransac_loop(
        X: np.ndarray, Y: np.ndarray, e: np.ndarray = None,
        ransac_samples: int = None, num_iterations: int = 100,
        use_tqdm: bool = False, simple: bool = True) -> RANSACOutput:
    """
    Performs a RANSAC loop to detect outliers in linear regression.

    Parameters:
    X (np.ndarray): A d x n array of feature vectors.
    Y (np.ndarray): An n-dimensional array of labels.
    e (np.ndarray): Axis of interest.
    ransac_samples (int): Number of samples for RANSAC, defaults to int(round(d*log(d))).
    num_iterations (int): Number of iterations of the RANSAC loop.
    use_tqdm (bool): Should we use TQDM to print a progress bar.
    simple (bool): Should we allow the RANSAC alg to choose the same sample more than once.

    Returns:
    List[Tuple[bool, float, np.ndarray, np.ndarray]]: Sorted list of RANSAC results.
    """
    n, d = X.shape
    if e is None:
        e = np.zeros(d)
        e[0] = 1
    if ransac_samples is None:
        ransac_samples = int(round(d * np.log(d)))

    results = []

    indices = range(num_iterations)
    if use_tqdm:
        indices = trange(num_iterations)

    for _ in indices:
        indices = np.random.choice(n, ransac_samples, replace=simple)
        results.append(scoring_function(X, Y, indices, e))
        # is_close, inner_product, f = scoring_function(X, Y, indices, e)
        # results.append((is_close, inner_product, f, indices))

    # Sorting the results
    results: List[ScoringResult]
    results.sort(key=lambda x: (x.is_close, -x.e_Sigma_inv_e))

    return RANSACOutput(
        ransac_results=results,
        ransac_samples=ransac_samples,
        ransac_iterations=num_iterations,
        total_samples=n,
        dimension=d
    )


def probability_disjoint_sets(n, k, epsilon):
    """
    Estimate the probability that two sets of size k and floor(epsilon * n)
    drawn randomly from n potential values are disjoint.

    Parameters:
    n (int): The total number of potential values.
    k (int): The size of the first set.
    epsilon (float): The fraction of n representing the size of the second set.

    Returns:
    float: The estimated probability of the two sets being disjoint.
    """
    k2 = int(np.floor(epsilon * n))
    prob = comb(n - k, k2) / comb(n, k2)
    return prob


import numpy as np
from scipy.special import comb
from typing import Union, List


# Updated ransac_probability_bound function
def ransac_probability_bound(N: int, k: int, w_min: float, bound: Union[float, np.ndarray],
                             epsilon: Union[float, np.ndarray], num_samples: int, simple: bool = True) -> np.ndarray:
    # Convert to numpy arrays
    bound = np.asarray(bound)
    epsilon = np.asarray(epsilon)

    if simple:
        # Simple RANSAC algorithm probability calculation
        probability_bounds = (1 - epsilon) ** k
    else:
        # Advanced RANSAC algorithm probability calculation
        # Using probability_disjoint_sets function (defined separately)
        probability_bounds = probability_disjoint_sets(num_samples, k, epsilon)

    alpha_values = (w_min / bound) * (1 - epsilon)
    unscaled_bounds = np.clip(np.nan_to_num(1 - probability_bounds * (1 - (1 / alpha_values)), nan=1.0), 0, 1)
    scaled_bounds = unscaled_bounds ** N

    return scaled_bounds


def process_ransac_results(ransac_results: RANSACOutput, bound: Union[float, np.ndarray],
                           epsilon: Union[float, np.ndarray], simple: bool = True) -> np.ndarray:
    '''

    :param ransac_results:
    :param bound: The bound we want to claim on max(lambda(inv(Sigma))
    :param epsilon: Fraction of samples removed
    :param simple:
    :return:
    '''
    # Extract the result with the smallest smallest_eigenvalue
    smallest_result = min(ransac_results, key=lambda result: result.smallest_eigenvalue)

    # Define w_min, k, and N
    w_min = smallest_result.smallest_eigenvalue
    k = len(smallest_result.indices)
    N = len(ransac_results)

    # Compute and return the probability bounds
    return ransac_probability_bound(N, k, w_min, bound, epsilon, num_samples=ransac_results.total_samples, simple=simple)


# Define the probability_disjoint_sets function
def probability_disjoint_sets(n, k, epsilon):
    k2 = np.array(np.round(epsilon * n), dtype=int)
    prob = comb(n - k, k2) / comb(n, k2)
    return prob



# Example usage of process_ransac_results
# ransac_results, bound, epsilon should be defined
# simple = True for simple RANSAC, False for advanced RANSAC
# results = process_ransac_results(ransac_results, bound, epsilon, simple=True)
@dataclass
class EpsilonBoundFunctor(BinarySearchFunctor):
    target_probability: Union[float, np.ndarray]
    simple: bool
    ransac_results: RANSACOutput
    epsilon_values: np.ndarray

    def __call__(self, bounds: np.ndarray) -> np.ndarray:
        computed_probabilities = process_ransac_results(
            ransac_results=self.ransac_results,
            bound=bounds,  # Note: Adjust this if necessary
            epsilon=self.epsilon_values,  # Note: Adjust this if necessary
            simple=self.simple
        )
        return computed_probabilities >= self.target_probability

    def get_initial_bounds(self, size: Optional[int] = None) -> (np.ndarray, np.ndarray):
        if size is None:
            size = len(self.epsilon_values)
        low = np.zeros(size)
        high = np.ones(size)  # Adjust based on expected range
        return low, high



# # Usage
# functor = EpsilonBoundFunctor(target_probability, simple_value, ransac_results)
# result = parallel_binary_search(functor, size=len(target_probability))
