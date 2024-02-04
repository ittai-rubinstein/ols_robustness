from typing import Union, Tuple

import numpy as np
import scipy as sp

IDENTITY_COEFF_GAUSSIAN = np.sqrt(3 / 2)
ALPHA_GAUSSIAN = 2 / 3


def compute_i_phi_coefficients(alpha: float, d: Union[float, int]) -> Tuple[float, float]:
    """
    Given alpha and d, returns i and phi coefficients such that:
    (c_i + c_phi * phi)^2 * (alpha + d * (1 - alpha) * phi) = 1 modulu (phi^2 = phi)
    This polynomial equality is ensured if
    (c_i + c_phi * phi)^2 * (alpha + d * (1 - alpha) * phi) = 1
    for phi in {0, 1}
    This means that:
     1. substituting phi=1, we have: (c_i + c_phi) = sqrt(1 / (alpha + d*(1 - alpha)))
     2. substituting phi=0, we have: c_i = sqrt(1 / alpha)
    Therefore (c_i, c_phi) = sqrt(1 / alpha), sqrt(1 / (alpha + (d * (1 - alpha)))) - sqrt(1 / alpha)
    :param alpha: The ratio between the portion of Norm{v}^4 that we draw from Norm{V}^2 (vs tr(V)^2 / d)
    :param d: The dimension of v
    :return: (c_i, c_phi) used for computing inverse root of cost matrix.
    """
    return np.sqrt(1 / alpha), np.sqrt(1 / (alpha + (d * (1 - alpha)))) - np.sqrt(1 / alpha)


def bound_42_hypercontractivity_gm(
        X: np.ndarray,
        alpha: float = ALPHA_GAUSSIAN
) -> np.ndarray:
    """
    Given an n by d array X, bound the 4-2 hyper-contractivity of its rows when projected onto any unit vector v.
    In other words, find a real value C such that for all v:
    \sum_i <X_i, v>^4 <= C (\sum_i <X_i, v>^2)^2
    :param X:
    :return:
    """
    n, d = X.shape
    # Step 1: renormalize X to have covariance Identity / n.
    # With this normalization, for any vector v, sum_i <X_i, v>^2 = Norm_2{v}^2
    cov = X.T @ X
    X_normalized = (sp.linalg.sqrtm(np.linalg.inv(cov)) @ X.T).T

    # Step 2: Generate an n by d**2 array whose ith row is the outer product between the ith row of X and itself.
    # At this point, note that for any vector v, we can write the 4th moment of projecting X onto it as:
    # Norm_4{X@v}^4 = \sum_{i \in [n]} <X_i, v>^4 = Norm_2{x_tensor_x @ (tensor_product(v, v)}^2
    # Note that Norm_2{v tensor v} = Norm_2{v}^2, so the ratio between
    # Norm_2{x_tensor_x @ (tensor_product(v, v)}^2 and Norm_2{v tensor v}^2 is equal to the ratio between
    # Norm_4{X@v}^4 = sum_i <X_i, v>^4 and Norm_2{X@v}^4 = Norm_2{v}^4 = (sum_i <X_i, v>^2)^2.
    x_tensor_x = np.einsum('ij,ik->ijk', X_normalized, X_normalized).reshape(n, d * d)

    # phi_coefficient = (np.sqrt(3 / (2 + d)) - np.sqrt(3 / 2)) / d

    # Step 3: Introduce a better scoring function.
    # phi is a vector such that inner_product(phi, v tensor v) = \sum_i v_i v_i = Norm_2{v}^2
    # This is useful, because we can now also generate Norm_2{v}^4 through the quadratic form:
    # (v tensor v).T @ phi @ phi.T @ (v tensor v)
    # This is good, because if we are working with the relaxation of V ?= v tensor v,
    # the optimizer might want to put more weight on the diagonal of V than should come from a tensor product between a
    # vector and itself. By telling the optimizer that instead of the cost of a matrix V being Norm_2{V}^2
    # (which maps to Norm_2{v}^4, if V = v tensor v), we can use the improved score function
    # Score(V) = alpha * Norm_2{V}^2 + (1 - alpha) * (<V, phi>^2)
    # This score function is still a quadratic form, and incentivizes the optimizer to put a more reasonable weight on
    # the diagonal of V.
    phi = np.reshape(np.identity(d), (d * d,))

    # Step 4: Take the improved score function into account
    # Here we use some algebra magic taken from Freund and Hopkins' paper.
    # Let Phi = (phi phi.T)/d. Freund and Hopkins make the optimization that:
    # ((2/3)*I + d*(1/3)*Phi)^(-1/2) = (sqrt(3/(2+d)) - sqrt(3/2)) Phi + sqrt(3/2) I;
    # In cases where d**2 > n, computing the LHS directly might be less efficient than using a precomputed formula.
    # To prove this equality, we simply note that Phi^2 = Phi, so for it to hold as a polynomial equation mod this
    # equation, we need only prove it for Phi = 0 and for Phi = 1.
    # To allow experimentation with new alpha values, we generalize this fast inversion logic for any alpha\in(0, 1)
    identity_coefficient, phi_coefficient = compute_i_phi_coefficients(alpha, d)
    better_tensor = identity_coefficient * x_tensor_x + \
                    (phi_coefficient * (x_tensor_x @ phi)[:, np.newaxis] * phi[np.newaxis, :] / d)

    # Step 5: Reduce to problem 1.
    # We first note that for M = X tensor X = matrix whose ith row is a flattened X_i @ X_i.transpose()
    # For any vector v, taking V to be the flattened version of v tensor v, we have that:
    #       (M @ V)_i = <X_i, v>^2
    # Moreover, taking W = ((2/3) I + (1/3) phi@phi.T), we know that V.T @ W @ V = V @ V
    # (due to the structure that V is a tensor of a vector with itself).
    # Therefore, taking tilde{V} = W^{1/2} V, we have that Norm_2{tilde{V}} = Norm_2{V}, and that for
    #           M'=M @ W^{-1/2} (= better_tensor in the code),
    # we have that:
    #           M' @ tilde{V} = M @ W^{-1/2} @ W^{1/2} @ V = (<X_i, v>^2)_{i in [n]}
    # Therefore, from a direct application of the CS inequality, we have that for any set S subseteq [n]:
    #           Norm_2{1_S^T @ M'} @ Norm_2{tilde{V}} >= 1_S^T @ M' @ tilde{V} = (<X_i, v>^2)_{i in S}
    # Moreover, this is implicitly what happens when we generate the 4-2 hypercontractivity following the recipe
    # of Freund and Hopkins. This is because their bound relies on bounding Norm_2{M' @ tilde{V}} by
    # max{sigma(M')} * Norm_2{tilde{V}} = max(sigma) * Norm_2{V} = max(sigma) * Norm_2{v}^2
    # (where sigma denotes the singular values of M')
    # The CS inequality is then applied using with this bound along with the fact that Norm_2{1_S} <= sqrt(|S|).
    # Therefore, so long as our upper bound on Norm_2{1_S^T @ M'} is better than sqrt(|S|) * max(sigma), then we
    # should get tighter bounds on our target.

    return better_tensor @ better_tensor.transpose()


def bound_42_hypercontractivity(X: np.ndarray) -> float:
    """
    Given an n by d array X, bound the 4-2 hyper-contractivity of its rows when projected onto any unit vector v.
    In other words, find a real value C such that for all v:
    \sum_i <X_i, v>^4 <= C (\sum_i <X_i, v>^2)^2
    :param X:
    :return:
    """
    n, d = X.shape
    return n * max(np.linalg.eigvalsh(bound_42_hypercontractivity_gm(X)))
