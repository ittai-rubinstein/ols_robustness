from typing import Tuple

import numpy as np
from numpy.linalg import lstsq

def compute_ols_and_error(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the best fit beta for the linear regression problem X@beta ≈ Y
    and the standard error of each individual coefficient.
    Parameters:
    X (np.ndarray): A 2D numpy array of predictors.
    Y (np.ndarray): A 1D numpy array of responses.

    Returns:
    tuple: A tuple containing the best fit beta coefficients and the standard error of each individual coefficient
    """
    # Solving for beta using least squares
    beta, residuals, _, _ = lstsq(X, Y, rcond=None)

    # Estimating the variance of the error terms (assuming homoscedasticity)
    # It's the sum of squared residuals divided by the degrees of freedom (n - p)
    n, p = X.shape
    variance = residuals / (n - p)

    # The standard error of beta coefficients is the square root of the diagonal elements
    # of the covariance matrix of beta
    cov_beta = variance * np.linalg.inv(X.T @ X).diagonal()
    std_error_beta = np.sqrt(cov_beta)

    return beta, std_error_beta


def eigenvectors_below_threshold(matrix: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    """
    Finds eigenvectors of a PSD matrix corresponding to eigenvalues below a given threshold.

    Parameters:
    matrix (np.ndarray): The input PSD matrix.
    threshold (float): The threshold for eigenvalues.

    Returns:
    np.ndarray: An array of eigenvectors corresponding to eigenvalues below the threshold.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    return eigenvectors[:, eigenvalues < threshold].transpose()
