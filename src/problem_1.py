from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from src.geometric_algorithms.spectral_algorithms import spectral_bound_sum_ips
from src.geometric_algorithms.triangle_inequality import refined_triangle_inequality_ips
from src.lower_bounds.problem_1_lower_bounds import Problem1LowerBounds, LowerBoundParams


@dataclass
class Problem1Params:
    lower_bound_params: LowerBoundParams = field(default_factory=LowerBoundParams)

    print_tqdm: bool = True
    use_triangle_inequality: bool = True
    use_spectral: bool = False

    # We may store the gram matrix even after we are done using it. If this is set to True, we will discard it to save RAM.
    save_memory: bool = True



@dataclass
class Problem1UpperBounds:
    combined_bound: np.ndarray  # This is not optional: ensure at least one method runs
    triangle_upper_bound: Optional[np.ndarray] = None
    spectral_upper_bound: Optional[np.ndarray] = None

class Problem1:
    _gram_matrix: Optional[np.ndarray] = None
    _params: Problem1Params
    upper_bounds: Problem1UpperBounds
    lower_bounds: Problem1LowerBounds

    def __init__(self, gram_matrix: np.ndarray, params: Problem1Params):
        """
        Wrapper to run all relevant problem 1 algorithms.

        In a problem 1, we are given a Gram matrix G, and asked to compute upper (and sometimes also lower) bounds on:
         max_{T subseteq [n] of size k} sqrt{1_T^T @ G @ 1_T}

        This class runs the requested subset of upper and lower bound algorithms (as specified by the params) input.
        """
        self._params = params
        self._gram_matrix = gram_matrix
        self.compute_bounds()
        if self._params.save_memory:
            self._gram_matrix = None

    def compute_bounds(self):
        self.compute_lower_bounds()
        self.compute_upper_bounds()

    def compute_lower_bounds(self):
        self.lower_bounds = Problem1LowerBounds(self._gram_matrix, self._params.lower_bound_params, self._params.print_tqdm)

    def compute_upper_bounds(self):
        self.upper_bounds = Problem1UpperBounds(
            combined_bound=np.full(self._gram_matrix.shape[0]-1, np.inf)
        )
        if self._params.use_triangle_inequality:
            self.upper_bounds.triangle_upper_bound = refined_triangle_inequality_ips(self._gram_matrix, self._params.print_tqdm)
            self.upper_bounds.combined_bound = np.minimum(
                self.upper_bounds.triangle_upper_bound, self.upper_bounds.combined_bound
            )
        if self._params.use_spectral:
            self.upper_bounds.spectral_upper_bound = spectral_bound_sum_ips(
                gram_matrix=self._gram_matrix,
                d=self._gram_matrix.shape[0]
            )
            self.upper_bounds.combined_bound = np.minimum(
                self.upper_bounds.spectral_upper_bound, self.upper_bounds.combined_bound
            )

    def plot_bounds(self) -> Axes:
        fig, ax = plt.subplots()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle='-', linewidth='0.5', color='grey')

        # Plot upper bounds if they exist
        k_vals = np.arange(1, len(self.upper_bounds.combined_bound)//10 + 1)
        if self.upper_bounds.triangle_upper_bound is not None:
            ax.plot(k_vals, self.upper_bounds.triangle_upper_bound[:len(k_vals)], linestyle='--', color='blue',
                    label='Triangle Upper Bound')
        if self.upper_bounds.spectral_upper_bound is not None:
            ax.plot(k_vals, self.upper_bounds.spectral_upper_bound[:len(k_vals)], linestyle='-.', color='blue',
                    label='Spectral Upper Bound')
        ax.plot(k_vals, self.upper_bounds.combined_bound[:len(k_vals)], linestyle='-', color='blue',
                label='Combined Upper Bound')

        # Plot lower bounds if they exist
        if self.lower_bounds.greedy_lower_bound is not None:
            ax.plot(k_vals[:len(self.lower_bounds.greedy_lower_bound.lower_bounds)],
                    self.lower_bounds.greedy_lower_bound.lower_bounds[:len(k_vals)], linestyle='--', color='red',
                    label='Greedy Lower Bound')
        if self.lower_bounds.very_greedy_lower_bound is not None:
            ax.plot(k_vals[:len(self.lower_bounds.very_greedy_lower_bound.lower_bounds)],
                    self.lower_bounds.very_greedy_lower_bound.lower_bounds[:len(k_vals)], linestyle='-.', color='red',
                    label='Very Greedy Lower Bound')

        ax.set_xlabel('Number of Samples Removed (k)')
        ax.set_ylabel('Problem 1 Estimates')
        ax.legend()

        fig.tight_layout
        plt.close(fig)

        return ax