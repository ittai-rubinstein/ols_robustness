import os
from pathlib import Path
from typing import Optional, List

import numpy as np
import scipy.linalg
from dataclasses import dataclass, field

from matplotlib import pyplot as plt

from src.categorical_data.categorical_data import compute_bounds_for_all
from src.categorical_data.dynamic_programming import dynamic_programming_1d
from src.categorical_data.ku_triangle_inequality import KUMetadata, KUData, compute_ku_bounds, extract_k_bounds
from src.geometric_algorithms.hypercontractivity_reduction import bound_42_hypercontractivity_gm
from src.geometric_algorithms.spectral_algorithms import spectral_bound_sum_ips
from src.geometric_algorithms.triangle_inequality import refined_triangle_inequality_ips
from src.problem_1 import Problem1, Problem1Params
from src.lower_bounds.removal_effects import LowerBoundConfig, RemovalEffectsLowerBound
from src.utils.linear_regression import LinearRegression
from src.utils.safe_min import safe_min
import time
from memory_profiler import memory_usage

SWITCH_TO_FLOAT32 = False
# @dataclass
# class Problem1Params:
#     use_triangle_inequality: bool = True
#     use_spectral: bool = False



@dataclass
class AuditorConfig:
    output_dir: Path
    reaverage: bool = True
    problem_1_params: Problem1Params = field(default_factory=Problem1Params)
    verbose: int = 2
    lower_bounds_params: LowerBoundConfig = field(default_factory=LowerBoundConfig)
    compute_freund_hopkins: bool = False

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


@dataclass
class ProfileResult:
    execution_time: float
    peak_memory_usage: float


@dataclass
class ParsedLinearRegression:
    R: np.ndarray
    Z: np.ndarray
    X: np.ndarray
    Sigma: np.ndarray
    root_Sigma: np.ndarray
    axis_of_interest: np.ndarray
    beta_e: float
    beta_e_sign: float
    delta_beta_e: float
    num_samples: int
    dimension: tuple
    XR: np.ndarray
    XZ: np.ndarray
    linear_effects: np.ndarray


def compute_ku_data(gram_matrix: np.ndarray, split_factors: List[np.ndarray], category_norm_bounds):
    highest_factor_averages = [
        np.concatenate((np.zeros(1), np.maximum(
            np.abs(np.cumsum(np.sort(factor_bucket))), np.abs(np.cumsum(np.sort(factor_bucket)[::-1]))
        ) / np.arange(1, len(factor_bucket) + 1))) for factor_bucket in split_factors
    ]
    averaging_effect_bounds = [
        hfa[::-1] * cnb for hfa, cnb in zip(highest_factor_averages, category_norm_bounds)
    ]
    return KUData(gram_matrix=gram_matrix, bucket_scores=averaging_effect_bounds)

@dataclass
class CategoricalUpperBounds:
    cs_bound: np.array
    xe_bound: np.array
    xr_bound: np.array
    direct_effect_bounds: np.ndarray

class RobustnessAuditor:
    linear_regression: LinearRegression
    config: AuditorConfig
    parsed_data: ParsedLinearRegression

    covariance_shift: Problem1
    XR_bounds: Problem1
    XZ_bounds: Problem1
    XZR_bounds: Problem1
    linear_effect_bounds: Problem1
    k_singular: int
    upper_bound: np.ndarray
    linear_effect: np.ndarray
    removal_effect_lower_bounds: RemovalEffectsLowerBound
    freund_and_hopkins_upper_bound: Optional[np.ndarray] = None
    categorical_upper_bounds: Optional[CategoricalUpperBounds] = None
    profile_results: Optional[ProfileResult] = None

    def __init__(self, linear_regression: LinearRegression, config: AuditorConfig):
        self.linear_regression = linear_regression
        self.config = config
        os.makedirs(self.config.output_dir, exist_ok=True)
        self._parse_regression()

    def _parse_regression(self):
        # Extract matrices from linear regression instance
        X = np.array(self.linear_regression.regression_arrays.X)
        R = np.array(self.linear_regression.regression_arrays.R)
        num_samples, dimension = X.shape

        # Access the fit coefficient and its standard error for the column of interest
        beta_e = self.linear_regression.model.params[self.linear_regression.column_of_interest]
        delta_beta_e = self.linear_regression.model.bse[self.linear_regression.column_of_interest]

        # If the sign of our fit was negative to begin with, then we want to bound removals that increase the fit value.
        beta_e_sign = np.sign(beta_e)
        if beta_e < 0:
            beta_e = -beta_e
            R = -R

        # # Reaverage X if the config flag is set
        # if self.config.reaverage:
        #     X = X - X.mean(axis=0)

        # Compute the covariance matrix Sigma and its square root
        Sigma = X.T @ X
        root_Sigma = scipy.linalg.sqrtm(Sigma)

        # Normalize X and adjust the axis of interest
        X_normalized = X @ np.linalg.inv(root_Sigma)
        axis_of_interest_normalized = np.array([1] + ([0]*(dimension-1))) @ np.linalg.inv(root_Sigma)
        Z = X_normalized @ axis_of_interest_normalized

        # Compute additional members
        XR = R[:, np.newaxis] * X_normalized
        XZ = Z[:, np.newaxis] * X_normalized
        linear_effects = XR @ axis_of_interest_normalized


        # Store all parsed and computed data in the dataclass
        self.parsed_data = ParsedLinearRegression(
            R=R, Z=Z, X=X_normalized, Sigma=Sigma, root_Sigma=root_Sigma,
            axis_of_interest=axis_of_interest_normalized, beta_e=beta_e, beta_e_sign=beta_e_sign, delta_beta_e=delta_beta_e,
            num_samples=num_samples, dimension=(num_samples, dimension),
            XR=XR, XZ=XZ, linear_effects=linear_effects
        )

    def compute_covariance_shift(self):
        """
        Wrapper to compute bounds on the covariance shift portion of the effect of removal of some samples.
        The goal of this computation is to bound the largest eigenvalue of Sigma_T where
            Sigma_T = sum_{i in T} X_i X_i^T
        is the contribution of the samples in T to the covariance matrix Sigma, and T is maximized over any set of size
         at most k (for k going from 1 to n).
        :return:
        """
        self.log("Computing bounds on the covariance shift...")
        X = self.parsed_data.X
        if SWITCH_TO_FLOAT32:
            X = X.astype(np.float32)
        self.covariance_shift = Problem1(
            gram_matrix= (X @ X.T)**2, params=self.config.problem_1_params
        )
        if self.config.verbose >= 2:
            self.log("Plotting results...")
            ax = self.covariance_shift.plot_bounds()
            ax.set_title("Bounds on CS")
            ax.figure.tight_layout()
            ax.figure.savefig(self.config.output_dir / "covariance_shift.png")

    def compute_XR_bounds(self):
        self.log("Computing problem 1 bound on XR...")
        XR = self.parsed_data.XR
        if SWITCH_TO_FLOAT32:
            XR = XR.astype(np.float32)
        self.XR_bounds = Problem1(
            gram_matrix=XR @ XR.T, params=self.config.problem_1_params
        )
        if self.config.verbose >= 2:
            self.log("Plotting results...")
            ax = self.XR_bounds.plot_bounds()
            ax.set_title(r"Bounds on $\Vert XR \Vert$")
            ax.figure.tight_layout()
            ax.figure.savefig(self.config.output_dir / "xr_bounds.png")

    def compute_XZ_bounds(self):
        self.log("Computing problem 1 bound on XZ...")
        XZ = self.parsed_data.XZ
        if SWITCH_TO_FLOAT32:
            XZ = XZ.astype(np.float32)
        self.XZ_bounds = Problem1(
            gram_matrix=XZ @ XZ.T, params=self.config.problem_1_params
        )
        if self.config.verbose >= 2:
            self.log("Plotting results...")
            ax = self.XZ_bounds.plot_bounds()
            ax.set_title(r"Bounds on $\Vert XZ \Vert$")
            ax.figure.tight_layout()
            ax.figure.savefig(self.config.output_dir / "xz_bounds.png")

    def compute_XZR_bounds(self):
        self.log("Computing problem 1 bound on <e Sigma_T, sum_i X_i R_i>...")
        XZ = self.parsed_data.XZ
        XR = self.parsed_data.XR
        if SWITCH_TO_FLOAT32:
            XZ = XZ.astype(np.float32)
            XR = XR.astype(np.float32)
        asymmetric_matrix = XZ @ XR.T
        gram_matrix = (asymmetric_matrix + asymmetric_matrix.T)/ 2
        self.XZR_bounds = Problem1(
            gram_matrix=gram_matrix, params=self.config.problem_1_params
        )
        if self.config.verbose >= 2:
            self.log("Plotting results...")
            ax = self.XZR_bounds.plot_bounds()
            ax.set_title(r"Bounds on $<e^T \Sigma_T, \sum_{i\in T} X_i R_i>$")
            ax.figure.tight_layout()
            ax.figure.savefig(self.config.output_dir / "xzr_bounds.png")

    def compute_linear_effect_bounds(self):
        self.log("Computing problem 1 bound on the linear effect...")
        self.linear_effect_bounds = Problem1(
            gram_matrix=np.diag(self.parsed_data.linear_effects), params=Problem1Params(use_triangle_inequality=True, use_spectral=False)
        )
        if self.config.verbose >= 2:
            self.log("Plotting results...")
            ax = self.linear_effect_bounds.plot_bounds()
            ax.set_title(r"Bounds on the Direct Effects")
            ax.figure.tight_layout()
            ax.figure.savefig(self.config.output_dir / "direct_effects.png")

    def _compute_k_singular(self):
        """
        Computes the smallest k for which the overall bound on the covariance shift
        is greater than or equal to 1.

        :return: The smallest index k (1-based index) or 0 if no such k exists.
        """

        # Find the first index where the bound is >= 1
        k_indices = np.where(self.upper_bound < 0)[0]

        if k_indices.size > 0:
            self.k_singular =  k_indices[0] + 1  # Return 1-based index
        else:
            self.k_singular = self.parsed_data.num_samples

    def log(self, msg: str):
        if self.config.verbose:
            print(msg)

    def compute_lower_bounds(self):
        self.removal_effect_lower_bounds = RemovalEffectsLowerBound(
            X = self.parsed_data.X,
            R = self.parsed_data.R,
            axis_of_interest = self.parsed_data.axis_of_interest,
            config=self.config.lower_bounds_params
        )

    def compute_all_bounds(self, categorical_aware: bool = False):
        def wrapper():
            self._compute_all_bounds(categorical_aware)

        start_time = time.time()
        mem_usage, _ = memory_usage((wrapper,), max_usage=True, retval=True)
        end_time = time.time()

        execution_time = end_time - start_time
        peak_memory_usage = mem_usage
        self.profile_results = ProfileResult(execution_time, peak_memory_usage)

    def _compute_all_bounds(self, categorical_aware: bool = False):

        self.removal_effect_lower_bounds = RemovalEffectsLowerBound(
            X=self.parsed_data.X,
            R=self.parsed_data.R,
            axis_of_interest=self.parsed_data.axis_of_interest,
            config=self.config.lower_bounds_params
        )

        if categorical_aware:
            return self.compute_all_bounds_categorical()

        self.compute_covariance_shift()
        self.compute_XR_bounds()
        self.compute_XZ_bounds()
        self.compute_XZR_bounds()
        self.compute_linear_effect_bounds()

        self.linear_effect = self.linear_effect_bounds.upper_bounds.combined_bound ** 2
        cs_term = self.covariance_shift.upper_bounds.combined_bound
        xz_term = self.XZ_bounds.upper_bounds.combined_bound
        xr_term = self.XR_bounds.upper_bounds.combined_bound
        xzr_term = self.XZR_bounds.upper_bounds.combined_bound
        self.upper_bound = self.linear_effect + (xzr_term ** 2) + ((cs_term * xr_term * xz_term) / (1 - cs_term))
        self._compute_k_singular()




    def compute_all_bounds_categorical(self):
        assert self.linear_regression.categorical_aware is not None

        # Extract information about the linear regression from the relevant field of self:
        split_X = self.linear_regression.categorical_aware.split_X
        split_R = self.linear_regression.categorical_aware.split_R
        split_weights = self.linear_regression.categorical_aware.split_weights
        X = np.vstack(split_X)
        residuals = np.concatenate(split_R)
        num_samples, dimension = X.shape
        axis_of_interest_normalized = self.linear_regression.categorical_aware.axis_of_interest_normalized

        # Set the high-level parameters for the ku bounds that we will use later on.
        ku_params = KUMetadata(k_max=int(num_samples * self.config.problem_1_params.k_max_factor) + 1, u_max=len(split_X) + 1,
                               bucket_sizes=[len(sr) for sr in split_R])

        # Step 1: Compute bounds on the direct effects of removals for the categorical data:

        # Compute a mapping from bucket index to upper bounds on its direct influences as a function of the number of samples removed from it:
        bounds_list = compute_bounds_for_all(
            split_X, split_R, axis_of_interest_normalized, split_weights
        )
        # Use a dynamic programming algorithm to solve the integer knapsack to maximize the total direct influences

        total_score_bounds = dynamic_programming_1d(bounds_list, int(num_samples * self.config.problem_1_params.k_max_factor) + 1)[1:]

        # Step 2: Bound the categorical aware covariance shift phenomena

        # These are bounds on the norm of the sum of any $k$ members of any bucket:
        category_norm_bounds = [
            refined_triangle_inequality_ips(np.diag(np.sqrt(bws)) @ bucket @ bucket.T @ np.diag(np.sqrt(bws)), verbose=False)
            for bucket, bws in zip(split_X, split_weights)
        ]
        # In sum ips functions, we do not include the options for 0 or n removals
        category_norm_bounds = [
            np.concatenate([[0], category_norm_bound, [0]]) for category_norm_bound in category_norm_bounds
        ]
        # This probably won't change too much, but we can use the fact that the sum over all samples in a bucket is always 0, so max norm sum over k samples == max norm sum over n-k samples.
        category_norm_bounds = [
            np.minimum(category_norm_bound, category_norm_bound[::-1]) for category_norm_bound in category_norm_bounds
        ]
        averaging_effect_CS_bounds = [
            (category_norm_bound[:-1] ** 2) / (len(category_norm_bound) - 1 - np.arange(len(category_norm_bound) - 1))
            for category_norm_bound in category_norm_bounds
        ]
        averaging_effect_CS_bounds = [
            np.concatenate((averaging_effect_CS_bound, np.zeros(1))) for averaging_effect_CS_bound in
            averaging_effect_CS_bounds
        ]

        # Compute bounds on the covariance shift and problem 1s as with any dataset.
        gram_matrix_CS = X @ X.transpose()
        ku_cs_data = KUData(gram_matrix=gram_matrix_CS ** 2, bucket_scores=averaging_effect_CS_bounds)
        ku_cs_bounds = compute_ku_bounds(ku_cs_data, ku_params)
        del ku_cs_data, gram_matrix_CS
        covariance_shift_bound = extract_k_bounds(ku_cs_bounds)


        # Step 3: Compute categorical problem 1 bounds on XR and XZ

        # Generate a version of the Xe and XR arrays tailored to the categorical dataset
        XR = X * residuals[:, np.newaxis]
        gram_matrix_XR = XR @ XR.transpose()

        ku_xr_data = compute_ku_data(gram_matrix_XR, [sr * np.sqrt(sw) for sr, sw in zip(split_R, split_weights)], category_norm_bounds)
        ku_xr_bounds = compute_ku_bounds(ku_xr_data, ku_params)
        del ku_xr_data, gram_matrix_XR

        XZ = X * (X @ axis_of_interest_normalized)[:, np.newaxis]
        split_Z = [bucket @ axis_of_interest_normalized for bucket in split_X]
        gram_matrix_XZ = XZ @ XZ.transpose()

        ku_xe_data = compute_ku_data(gram_matrix_XZ, [sz * np.sqrt(sw) for sz, sw in zip(split_Z, split_weights)], category_norm_bounds)
        ku_xe_bounds = compute_ku_bounds(ku_xe_data, ku_params)
        del ku_xe_data, gram_matrix_XZ

        xr_bound = extract_k_bounds(ku_xr_bounds)
        xe_bound = extract_k_bounds(ku_xe_bounds)

        self.categorical_upper_bounds = CategoricalUpperBounds(
            direct_effect_bounds=total_score_bounds,
            cs_bound=covariance_shift_bound,
            xe_bound=xe_bound,
            xr_bound=xr_bound
        )
        self.upper_bound = total_score_bounds + (xe_bound * xr_bound / (1 - covariance_shift_bound))

    def summary(self):
        self.upper_bound[self.upper_bound < 0] = np.inf
        result = {
            "fit_value": self.parsed_data.beta_e * self.parsed_data.beta_e_sign,
            "fit_sign": self.parsed_data.beta_e_sign,
            "error_bar": self.parsed_data.delta_beta_e,
            "dimension": self.parsed_data.dimension,
            "num_samples": self.parsed_data.num_samples
        }
        if self.linear_regression.special_categorical:
            result['singularity'] = min(map(len, self.linear_regression.categorical_aware.split_R))
        indices = np.arange(1, self.parsed_data.num_samples)
        result["Lower Bound"] = safe_min(indices[:len(self.upper_bound)][self.upper_bound > self.parsed_data.beta_e])
        if self.removal_effect_lower_bounds.amip:
            amip = self.removal_effect_lower_bounds.amip.removal_effects
            result["AMIP"] = safe_min(indices[:len(amip)][amip > self.parsed_data.beta_e])

        if self.removal_effect_lower_bounds.single_greedy:
            single_greedy = self.removal_effect_lower_bounds.single_greedy.removal_effects
            result["Greedy"] = safe_min(indices[:len(single_greedy)][single_greedy > self.parsed_data.beta_e])

        if self.removal_effect_lower_bounds.triple_greedy:
            triple_greedy = self.removal_effect_lower_bounds.single_greedy.removal_effects
            result["Triple Greedy"] = safe_min(indices[:len(triple_greedy)][ triple_greedy > self.parsed_data.beta_e])


        if self.removal_effect_lower_bounds.kzc21:
            kzc21 = self.removal_effect_lower_bounds.kzc21.removal_effects
            result["KZC21"] = safe_min(indices[:len(kzc21)][kzc21 > self.parsed_data.beta_e])

        if self.profile_results:
            result["Runtime"] = self.profile_results.execution_time
            result["Memory"] = self.profile_results.peak_memory_usage

        return result

    def plot_removal_effects(self):
        self._compute_k_singular()
        k_vals = np.arange(1, len(self.upper_bound) + 1)
        self.upper_bound[self.upper_bound < 0] = np.inf

        fig, ax = plt.subplots()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle='--', linewidth='0.75', color='grey')
        ax.grid(True, which="major", linestyle='-', linewidth='1', color='grey')

        # Plotting bounds
        ax.plot(k_vals[:self.k_singular+1], self.upper_bound[:self.k_singular+1],
                'b-', label='Upper Bound', linewidth=3)
        if self.freund_and_hopkins_upper_bound is not None:
            ax.plot(k_vals, self.freund_and_hopkins_upper_bound,
                    'g-', label='Freund & Hopkins', linewidth=3)

        # Plotting lower bounds
        if self.removal_effect_lower_bounds.amip:
            ax.plot(k_vals[:len(self.removal_effect_lower_bounds.amip.removal_effects)],
                    self.removal_effect_lower_bounds.amip.removal_effects, 'r--', label='AMIP Lower Bound', linewidth=3)
        if self.removal_effect_lower_bounds.kzc21:
            ax.plot(k_vals[:len(self.removal_effect_lower_bounds.kzc21.removal_effects)],
                    self.removal_effect_lower_bounds.kzc21.removal_effects, 'r-.', label='KZC21 Lower Bound', linewidth=3)
        if self.removal_effect_lower_bounds.triple_greedy:
            ax.plot(
                k_vals[:len(self.removal_effect_lower_bounds.triple_greedy.removal_effects)],
                self.removal_effect_lower_bounds.triple_greedy.removal_effects, 'r-', label='Triple Greedy Lower Bound',
                linewidth=3
            )
        if self.removal_effect_lower_bounds.single_greedy:
            ax.plot(k_vals[:len(self.removal_effect_lower_bounds.single_greedy.removal_effects)],
                    self.removal_effect_lower_bounds.single_greedy.removal_effects, 'r:', label=r'Greedy $\left \langle \Sigma_T e, \sum_{i \in T} X_i R_i \right \rangle$ Lower Bound', linewidth=3)

        ax.axhline(y=self.parsed_data.beta_e, color='black', label=r'$<\beta, e>$', linewidth=3)

        # Calculate the desired upper y-limit
        upper_y_limit = self.parsed_data.beta_e + 10 * self.parsed_data.delta_beta_e
        ylim = ax.get_ylim()
        ax.set_ylim(bottom=None, top=min(ylim[1], upper_y_limit))

        # Legend, labels, and layout
        ax.legend()
        ax.set_xlabel('Number of Samples Removed (k)')
        ax.set_ylabel('Removal Effect Estimates')
        ax.figure.tight_layout()
        # Save the figure
        output_file = Path(self.config.output_dir) / "removal_effects_plot.png"
        fig.savefig(output_file)

        # Return the axes for further manipulation if necessary
        return ax


    def compute_freund_hopkins(self):
        X = self.parsed_data.X
        ks = np.arange(1, self.parsed_data.num_samples)
        hyper_gram_matrix = bound_42_hypercontractivity_gm(X)
        eigvals_hyper, eigvecs_hyper = np.linalg.eigh(hyper_gram_matrix)
        # Replicate the Freund and Hopkins analysis as a baseline:
        XR = self.parsed_data.XR
        C1 = np.max(np.linalg.eigvalsh(XR @ XR.T))
        C2 = np.max(eigvals_hyper)
        self.freund_and_hopkins_upper_bound = np.sqrt(C1 * ks) / (1 - np.sqrt(C2 * ks))
        self.freund_and_hopkins_upper_bound[C2 * ks >= 1] = np.inf