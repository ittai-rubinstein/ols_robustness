import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import tqdm


@dataclass
class Problem1LowerBound:
    algorithm: str
    lower_bounds: List[float]
    order_of_selection: List[int]


@dataclass
class LowerBoundParams:
    run_greedy: bool = True
    run_very_greedy: bool = False

class Problem1LowerBounds:
    greedy_lower_bound: Optional[Problem1LowerBound] = None
    very_greedy_lower_bound: Optional[Problem1LowerBound] = None

    def __init__(self, gram_matrix: np.ndarray, params: LowerBoundParams, print_tqdm: bool = True):
        if params.run_greedy:
            self.greedy_lower_bound = greedy_lower_bound(gram_matrix, print_tqdm)
        if params.run_very_greedy:
            self.very_greedy_lower_bound = very_greedy_lower_bound(gram_matrix, print_tqdm)


def very_greedy_lower_bound(gram_matrix:np.ndarray, print_tqdm: bool = True) -> Problem1LowerBound:
    n = gram_matrix.shape[0]
    indices = np.argsort(-np.diag(gram_matrix))
    scores = []
    order_of_selection = indices.tolist()

    for k in tqdm.trange(1, n + 1, desc="Very Greedy Lower Bound", disable=not print_tqdm):
        T = indices[:k]
        T_indicator = np.zeros(n)
        T_indicator[T] = 1
        score = np.sqrt(T_indicator @ gram_matrix @ T_indicator)
        scores.append(score)

    return Problem1LowerBound(lower_bounds=scores, order_of_selection=order_of_selection, algorithm="Very Greedy")

def greedy_lower_bound(gram_matrix: np.ndarray, print_tqdm: bool = True) -> Problem1LowerBound:
    n = gram_matrix.shape[0]
    T = []
    scores = [0]
    delta_scores = np.copy(np.diagonal(gram_matrix))

    for _ in tqdm.trange(1, n, desc="Greedy Lower Bound", disable=not print_tqdm):
        best_index = np.argmax(delta_scores)

        T.append(best_index)
        scores.append(scores[-1] + delta_scores[best_index])
        delta_scores += 2 * gram_matrix[best_index, :]
        delta_scores[best_index] = - np.inf

    k = int(np.round(np.sqrt(n)))
    T_ind = np.zeros(n)
    T_ind[T[:k]] = 1
    score = T_ind @ gram_matrix @ T_ind
    # print(f"{score}?={scores[k]}")
    assert np.isclose(score, scores[k])

    return Problem1LowerBound(lower_bounds=np.sqrt(scores[1:]), order_of_selection=T, algorithm="Greedy")
