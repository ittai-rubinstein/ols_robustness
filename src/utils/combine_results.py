import itertools
from typing import List

import numpy as np
import tqdm

def score_function_combined_CS_RA(a, b, c):
    """
    Compute the score function score(a, b, c) = a + 0.5*b / (1 - sqrt(c))
    with the condition that the score is infinity if c >= 1 and -infinity if c < 0.

    This function represents the strength of the effect of the removal of a set of samples
    from our linear regression. Here
        a = sum_{i in T} R_i <X_i, e>
        b = ||sum_(i in T) R_i X_i||^2 + ||sum_(i in T) <X_i, e> X_i||^2 >= 2 ||sum_(i in T) R_i X_i|| * ||sum_(i in T) <X_i, e> X_i||
        c = (sum_(i in T) <X_i, v>^2)^2 <= ||sum_(i in T) X_i tensor X_i||^2

    Our bound on the total influence of these removals is:
        result = bound on: a + 0.5 * b / (1 - sqrt(c))

    On the other hand, we can bound each of a,b,c as well as any linear combination of them.
    We define the Gram matrices of each component:
        G_a = diag(sqrt(R_i <X_i, e>)); note that some R_i <X_i, e> are bound to be negative, in which case, it may be best to set this diagonal to be imaginary even if the resulting matrix is no longer PSD.
        G_b = Gram Matrix(R_i X_i) + Gram Matrix(<X_i, e_i> X_i)
        G_c = Gram Matrix(X_i tensor X_i); In some cases it is better to use a slightly different representation - see hypercontractivity.
    We can then bound any linear combination of a, b, c by applying our problem-1 algorithms for
        the respective linear combination of its gram matrices.

    Our approach to bounding score function will be to enumerate over n_a potential values of a and n_b values of b.
    For each choice of a,b we compute the largest value c for which none of our conditions are violated (by enumeration over linear constraints).
    We then use the monotonicity of our score and compute it on a' = a + delta_a, b' = b + delta_b, c'=c to bound its values on the whole area of [a, a+delta_a] times [b, b+delta_b] times {c <= bound(a, b)}.

    This component of the algorithm runs in time n_a times n_b times N_{linear constraints} times n, and it is easy to show that its output is an upper bound that is within a factor of
        (1 + 3(1/n_a + 1/n_b)) of being tight.

    :param a: Numpy array of 'a' values.
    :param b: Numpy array of 'b' values.
    :param c: Numpy array of 'c' values.
    :return: Numpy array of computed scores.
    """
    # Calculate the square root components
    non_negative_c_indices = (c >= 0)
    sqrt_c = np.zeros_like(c)
    sqrt_c[non_negative_c_indices] = np.sqrt(c[non_negative_c_indices])

    # Initialize the score array with the first component of the score function
    scores = np.copy(a)

    # Calculate the second component where c < 1
    no_divide_by_zero = c < 1
    second_component = np.zeros_like(b)  # Initialize with zeros
    # Update only where c < 1 to avoid division by zero or negative values
    second_component[no_divide_by_zero] = 0.5 * b[no_divide_by_zero] / (1 - sqrt_c[no_divide_by_zero])

    # Add the second component to the scores
    scores += second_component

    # Set score to infinity where c >= 1 and -infinity when c < 0
    scores[np.logical_not(no_divide_by_zero)] = np.inf
    scores[np.logical_not(non_negative_c_indices)] = -np.inf

    return scores

from dataclasses import dataclass


@dataclass
class LinearConstraint:
    l: np.ndarray  # Coefficients as a row vector [l_a, l_b, l_c]
    r: np.ndarray  # Right-hand side values as a column vector for multiple problem instances

    def __post_init__(self):
        self.l = np.atleast_2d(self.l)
        self.r = np.atleast_2d(self.r).T

    def check(self, variables):
        a, b, c = variables
        lhs = self.l @ np.vstack([a, b, c])
        return np.all(lhs <= self.r, axis=0)


def solve_for_missing_variable(constraint: LinearConstraint, given_values: dict, max_values: dict) -> np.ndarray:
    """
    Solve for the missing variable given two of a, b, c, a linear constraint, and their max values.

    :param constraint: LinearConstraint object containing the linear constraint.
    :param given_values: Dictionary with two out of three variables a, b, c provided as numpy arrays.
    :param max_values: Dictionary containing the max values for a, b, c as a_max, b_max, c_max.

    :return: Numpy array containing the values of the missing variable.
    """
    # Extract coefficients and right-hand side from the constraint
    l_a, l_b, l_c = constraint.l.flatten()
    r = constraint.r.flatten()

    # Determine the missing variable and its max value
    variables = {'a': None, 'b': None, 'c': None}
    variables.update(given_values)
    missing_var = next(key for key, value in variables.items() if value is None)
    max_value = max_values[f'{missing_var}_max']

    # Calculate the value of the missing variable
    if missing_var == 'a' and l_a != 0:
        calculated_value = (r - l_b * variables['b'] - l_c * variables['c']) / l_a
    elif missing_var == 'b' and l_b != 0:
        calculated_value = (r - l_a * variables['a'] - l_c * variables['c']) / l_b
    elif missing_var == 'c' and l_c != 0:
        calculated_value = (r - l_a * variables['a'] - l_b * variables['b']) / l_c
    else:
        # If the coefficient of the missing variable is 0, return its max value
        return np.full_like(r, max_value)

    # Ensure the calculated value does not exceed its max value
    return np.minimum(calculated_value, max_value)

@dataclass
class CombinedInfluenceBound:
    a_max: np.ndarray
    a_min: np.ndarray
    b_max: np.ndarray
    c_max: np.ndarray
    constraints: List[LinearConstraint]

    def solve(self, n_a=1000, n_b=1000, use_tqdm=True):
        delta_a = (self.a_max - self.a_min) / n_a
        delta_b = self.b_max / n_b

        best_score = -np.inf * np.ones_like(self.a_max)  # Initialize the best score

        iterator = itertools.product(range(n_a), range(n_b))
        if use_tqdm:
            iterator = tqdm.tqdm(list(iterator))  # Wrap iterator with tqdm for progress bar

        for i_a, i_b in iterator:
            a_val = self.a_min + (i_a * delta_a)
            b_val = i_b * delta_b

            # Initialize an array to hold 'c' values for all constraints
            c_vals = np.repeat(self.c_max[np.newaxis, :], len(self.constraints), axis=0)

            # Iterate over constraints to solve for 'c'
            for i_constraint, constraint in enumerate(self.constraints):
                c_val = solve_for_missing_variable(constraint, {'a': np.array([a_val]), 'b': np.array([b_val])},
                                                   {'a_max': self.a_max, 'b_max': self.b_max, 'c_max': self.c_max})
                c_val = np.minimum(c_val, self.c_max)  # Cap 'c' value by c_max
                c_vals[i_constraint, :] = c_val

            # Take the minimum 'c' value obtained from all constraints
            min_c_val = np.min(c_vals, axis=0)

            # Compute score with a' = a + delta_a and b' = b + delta_b
            score = score_function_combined_CS_RA(a_val, b_val, min_c_val)

            # Update best score
            best_score = np.maximum(best_score, score)

        return best_score

