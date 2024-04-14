from pathlib import Path

import numpy as np
import scipy.linalg
from dataclasses import dataclass, field

from src.problem_1 import Problem1, Problem1Params
from src.lower_bounds.removal_effects import LowerBoundConfig, RemovalEffectsLowerBound
from src.utils.linear_regression import LinearRegression


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



@dataclass
class ParsedLinearRegression:
    R: np.ndarray
    Z: np.ndarray
    X: np.ndarray
    Sigma: np.ndarray
    root_Sigma: np.ndarray
    axis_of_interest: np.ndarray
    beta_e: float
    delta_beta_e: float
    num_samples: int
    dimension: tuple
    XR: np.ndarray
    XZ: np.ndarray
    linear_effects: np.ndarray


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

    def __init__(self, linear_regression: LinearRegression, config: AuditorConfig):
        self.linear_regression = linear_regression
        self.config = config
        self._parse_regression()

    def _parse_regression(self):
        # Extract matrices from linear regression instance
        X = self.linear_regression.regression_arrays.X
        R = self.linear_regression.regression_arrays.R
        num_samples, dimension = X.shape

        # Reaverage X if the config flag is set
        if self.config.reaverage:
            X = X - X.mean(axis=0)

        # Compute the covariance matrix Sigma and its square root
        Sigma = X.T @ X
        root_Sigma = scipy.linalg.sqrtm(Sigma)

        # Normalize X and adjust the axis of interest
        X_normalized = X @ np.linalg.inv(root_Sigma)
        axis_of_interest_normalized = np.array([1] + ([0]*(dimension-1))) @ np.linalg.inv(root_Sigma)
        Z = X_normalized @ axis_of_interest_normalized

        # Compute additional members
        XR = R[:, np.newaxis] * X
        XZ = Z[:, np.newaxis] * X
        linear_effects = XR @ axis_of_interest_normalized

        # Access the fit coefficient and its standard error for the column of interest
        beta_e = self.linear_regression.model.params[self.linear_regression.column_of_interest]
        delta_beta_e = self.linear_regression.model.bse[self.linear_regression.column_of_interest]

        # Store all parsed and computed data in the dataclass
        self.parsed_data = ParsedLinearRegression(
            R=R, Z=Z, X=X_normalized, Sigma=Sigma, root_Sigma=root_Sigma,
            axis_of_interest=axis_of_interest_normalized, beta_e=beta_e, delta_beta_e=delta_beta_e,
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
        self.covariance_shift = Problem1(
            gram_matrix= (X @ X.T)**2, params=self.config.problem_1_params
        )
        self._compute_k_singular()
        if self.config.verbose >= 2:
            self.log("Plotting results...")
            ax = self.covariance_shift.plot_bounds()
            ax.set_title("Bounds on CS")
            ax.figure.savefig(self.config.output_dir / "covariance_shift.png")

    def compute_XR_bounds(self):
        self.log("Computing problem 1 bound on XR...")
        XR = self.parsed_data.XR
        self.XR_bounds = Problem1(
            gram_matrix=XR @ XR.T, params=self.config.problem_1_params
        )
        if self.config.verbose >= 2:
            self.log("Plotting results...")
            ax = self.XR_bounds.plot_bounds()
            ax.set_title(r"Bounds on $\Vert XR \Vert$")
            ax.figure.savefig(self.config.output_dir / "xr_bounds.png")

    def compute_XZ_bounds(self):
        self.log("Computing problem 1 bound on XZ...")
        XZ = self.parsed_data.XZ
        self.XZ_bounds = Problem1(
            gram_matrix=XZ @ XZ.T, params=self.config.problem_1_params
        )
        if self.config.verbose >= 2:
            self.log("Plotting results...")
            ax = self.XZ_bounds.plot_bounds()
            ax.set_title(r"Bounds on $\Vert XZ \Vert$")
            ax.figure.savefig(self.config.output_dir / "xz_bounds.png")

    def compute_XZR_bounds(self):
        self.log("Computing problem 1 bound on <e Sigma_T, sum_i X_i R_i>...")
        XZ, XR = self.parsed_data.XZ, self.parsed_data.XR
        asymmetric_matrix = XZ @ XR.T
        gram_matrix = (asymmetric_matrix + asymmetric_matrix.T)/ 2
        self.XZR_bounds = Problem1(
            gram_matrix=gram_matrix, params=self.config.problem_1_params
        )
        if self.config.verbose >= 2:
            self.log("Plotting results...")
            ax = self.XZR_bounds.plot_bounds()
            ax.set_title(r"Bounds on $<e^T \Sigma_T, \sum_{i\in T} X_i R_i>$")
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
            ax.figure.savefig(self.config.output_dir / "direct_effects.png")

    def _compute_k_singular(self):
        """
        Computes the smallest k for which the overall bound on the covariance shift
        is greater than or equal to 1.

        :return: The smallest index k (1-based index) or 0 if no such k exists.
        """
        # Get the combined upper bounds from the problem instance
        combined_bounds = self.covariance_shift.upper_bounds.combined_bound

        # Find the first index where the bound is >= 1
        k_indices = np.where(combined_bounds >= 1)[0]

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

    def compute_all_bounds(self):
        self.compute_covariance_shift()
        self.compute_XR_bounds()
        self.compute_XZ_bounds()
        self.compute_XZR_bounds()
        self.compute_linear_effect_bounds()

        self.linear_effect = self.linear_effect_bounds.upper_bounds.combined_bound ** 2
        cs_term = self.covariance_shift.upper_bounds.combined_bound
        xz_term = self.XZ_bounds.upper_bounds.combined_bound
        xr_term =  self.XR_bounds.upper_bounds.combined_bound
        xzr_term = self.XZR_bounds.upper_bounds.combined_bound
        self.upper_bound = self.linear_effect + xzr_term + ((cs_term * xr_term * xz_term) / (1 - cs_term))
