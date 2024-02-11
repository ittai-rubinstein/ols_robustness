from dataclasses import dataclass, field
from typing import List, Optional, NamedTuple
import numpy as np

@dataclass
class OneHotEncoding:
    column_index: int
    sample_indices: List[int]
    column_name: Optional[str] = None

def detect_one_hot_encodings(data: np.ndarray, column_names: Optional[List[str]] = None) -> List[OneHotEncoding]:
    """
    Detects columns in the given array that represent one-hot encodings and returns
    a list of OneHotEncoding dataclasses with details about these columns.

    Parameters:
    - data: np.ndarray. The input array of samples.
    - column_names: Optional[List[str]]. A list of column names corresponding to the array's columns.

    Returns:
    - List[OneHotEncoding]: A list of OneHotEncoding objects, each representing a one-hot encoded column.
    """
    results = []
    n_samples, n_features = data.shape

    for col_index in range(n_features):
        column = data[:, col_index]
        unique_values = np.unique(column)

        if set(unique_values).issubset({0, 1}):
            one_hot_indices = np.where(column == 1)[0]
            column_name = column_names[col_index] if column_names else None
            results.append(OneHotEncoding(col_index, list(one_hot_indices), column_name))

    return results


@dataclass
class QuadrantInfo:
    column_index: int
    quadrant: str
    sample_indices: List[int]
    weight: int
    value: float
    column_name: Optional[str] = None
    score: float = field(init=False)  # Score will be computed after instantiation

    def __post_init__(self):
        self.score = self.value / self.weight if self.weight > 0 else float('-inf')



def compute_quadrants_vectorized(samples: np.ndarray, one_hot_encodings: List[OneHotEncoding], e: np.ndarray, residuals: np.ndarray) -> List[QuadrantInfo]:
    """
    Computes the potential sets of sample indices of interest, classified into quadrants based on the signs
    of <X_i, e> and R_i, for each one-hot encoding, using partial NumPy vectorization for efficiency.

    Parameters are the same as in the non-vectorized version.
    """
    results = []

    # Compute dot products for all samples at once
    dot_products = samples @ e  # Assuming e is a column vector; if not, use np.dot(samples, e) instead

    for encoding in one_hot_encodings:
        col_index, one_hot_samples, col_name = encoding.column_index, encoding.sample_indices, encoding.column_name

        # Filter samples for the current one-hot encoding
        filtered_dots = dot_products[one_hot_samples]
        filtered_residuals = residuals[one_hot_samples]

        # Conditions for quadrants
        plus_plus_condition = (filtered_dots > 0) & (filtered_residuals > 0)
        minus_minus_condition = (filtered_dots < 0) & (filtered_residuals < 0)

        # Extract indices for each quadrant
        plus_plus_indices = np.array(one_hot_samples)[plus_plus_condition]
        minus_minus_indices = np.array(one_hot_samples)[minus_minus_condition]

        # Compute attributes for each quadrant
        for quadrant, indices in [('+ +', plus_plus_indices), ('- -', minus_minus_indices)]:
            if indices.size > 0:
                quadrant_dots = filtered_dots[plus_plus_condition if quadrant == '+ +' else minus_minus_condition]
                quadrant_residuals = filtered_residuals[plus_plus_condition if quadrant == '+ +' else minus_minus_condition]

                weight = indices.size
                CS = len(one_hot_samples) / (len(one_hot_samples) - weight)

                # Compute the value using vectorized operations
                value = (quadrant_dots * quadrant_residuals).sum() + \
                        quadrant_dots.sum() * quadrant_residuals.sum() * CS

                results.append(QuadrantInfo(col_index, quadrant, indices.tolist(), weight, value, col_name))

    return results


def select_and_sort_quadrants(quadrants: List[QuadrantInfo]) -> List[QuadrantInfo]:
    highest_score_quadrants = {}

    # Select the highest score quadrant for each one-hot encoding
    for quadrant in quadrants:
        if quadrant.column_index not in highest_score_quadrants or \
           quadrant.score > highest_score_quadrants[quadrant.column_index].score:
            highest_score_quadrants[quadrant.column_index] = quadrant

    # Sort the selected quadrants by score in descending order
    sorted_quadrants = sorted(highest_score_quadrants.values(), key=lambda q: q.score, reverse=True)

    return sorted_quadrants


def compile_ordered_sample_indices(sorted_quadrants: List[QuadrantInfo]) -> List[int]:
    ordered_indices = []
    seen_indices = set()

    for quadrant in sorted_quadrants:
        for index in quadrant.sample_indices:
            if index not in seen_indices:
                ordered_indices.append(index)
                seen_indices.add(index)

    return ordered_indices
