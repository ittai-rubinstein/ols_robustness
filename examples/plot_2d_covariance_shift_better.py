import numpy as np
from typing import Tuple, Optional

# Generate the dataset
np.random.seed(0)
n_inliers = 1000
n_outliers = 10
# Set gamma and c
gamma = np.array([-1])
c = 1

DPI = 300
FIGSIZE = (6, 5)

def make_non_degenerate(X: np.ndarray, Y: np.ndarray, c: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perturbs X and Y slightly to make them non-degenerate.

    Parameters:
    X (np.ndarray): Feature matrix of shape (n, d-1).
    Y (np.ndarray): Response vector of shape (n,).
    c (float): Perturbation magnitude.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Perturbed feature matrix and response vector.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    rank = np.sum(s > 1e-10)
    if rank < X.shape[1]:
        perturbation_X = np.random.randn(*X.shape)
        perturbation_Y = np.random.randn(*Y.shape)
        perturbation_X = (c / 2) * perturbation_X / np.linalg.norm(perturbation_X)
        perturbation_Y = (c / 2) * perturbation_Y / np.linalg.norm(perturbation_Y)
        return X + perturbation_X, Y + perturbation_Y
    return X, Y


def construct_modified_dataset(X: np.ndarray, Y: np.ndarray, gamma: np.ndarray, c: float) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Constructs modified datasets X' and Y' such that the ordinary least squares solution
    on a subset S is given by gamma, while ensuring X' and Y' are close to X and Y respectively.

    Parameters:
    X (np.ndarray): Original feature matrix of shape (n, d).
    Y (np.ndarray): Original response vector of shape (n,).
    gamma (np.ndarray): Target regression coefficients of shape (d-1,).
    c (float): Perturbation magnitude.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Modified feature matrix X' and response vector Y'.
    """
    n, d = X.shape
    # Identify the set S where the d-th feature is 0
    S = np.where(X[:, d - 1] == 0)[0]
    X_S = X[S, :-1]
    Y_S = Y[S]

    # Ensure X_S and Y_S are non-degenerate
    X_S_prime, Y_S_prime = make_non_degenerate(X_S, Y_S, c)

    # Calculate the residual vector R
    R = Y_S_prime - X_S_prime @ gamma

    # Set the values of X_prime_{S, d}
    X_prime_S_d = (c / (2 * np.linalg.norm(R))) * R

    # Construct the modified X_prime and Y_prime
    X_prime = X.copy()
    Y_prime = Y.copy()
    X_prime[S, d - 1] = X_prime_S_d
    Y_prime[S] = Y_S_prime
    # Y_prime[S] = X_prime[S, :-1] @ np.append(gamma, -2 * np.linalg.norm(R) / c)

    return X_prime, Y_prime



import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt



# Inliers
X1_inliers = np.random.normal(0, 1, n_inliers)
X2_inliers = np.zeros(n_inliers)
Y_inliers = X1_inliers + np.random.normal(0, 1, n_inliers)

# Outliers
X1_outliers = 0.1 * np.random.normal(0, 1, n_outliers)
X2_outliers = np.ones(n_outliers)
Y_outliers = X1_outliers + np.random.normal(0, 1, n_outliers)

# Combine inliers and outliers
X1 = np.concatenate([X1_inliers, X1_outliers])
X2 = np.concatenate([X2_inliers, X2_outliers])
Y = np.concatenate([Y_inliers, Y_outliers])

# Store in a DataFrame
df_original = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})



# Prepare the data for the function
X = df_original[['X1', 'X2']].values
Y = df_original['Y'].values

# Compute the perturbed dataset
X_prime, Y_prime = construct_modified_dataset(X, Y, gamma, c)

# Store the perturbed dataset in a DataFrame
df_perturbed = pd.DataFrame(X_prime, columns=['X1', 'X2'])
df_perturbed['Y'] = Y_prime

# Identify the inliers in the perturbed dataset (where X2 is 0)
df_perturbed_inliers = df_perturbed[df_original['X2'] == 0]

# Perform intercept-free regression on both dataframes
reg_original = LinearRegression(fit_intercept=False).fit(df_original[['X1', 'X2']], df_original['Y'])
reg_perturbed = LinearRegression(fit_intercept=False).fit(df_perturbed[['X1', 'X2']], df_perturbed['Y'])
reg_perturbed_inliers = LinearRegression(fit_intercept=False).fit(df_perturbed_inliers[['X1', 'X2']], df_perturbed_inliers['Y'])

# Print the results
print("Original dataset regression coefficients:")
print(reg_original.coef_)

print("\nPerturbed dataset regression coefficients:")
print(reg_perturbed.coef_)

print("\nPerturbed dataset regression coefficients (without outliers):")
print(reg_perturbed_inliers.coef_)

# Perform intercept-free regression on the inliers of the original dataset
df_original_inliers = df_original[df_original['X2'] == 0]
reg_original_inliers = LinearRegression(fit_intercept=False).fit(df_original_inliers[['X1', 'X2']], df_original_inliers['Y'])

# Perform intercept-free regression on the inliers of the perturbed dataset
reg_perturbed_inliers = LinearRegression(fit_intercept=False).fit(df_perturbed_inliers[['X1', 'X2']], df_perturbed_inliers['Y'])

# Plotting the perturbed dataset
plt.figure(figsize=FIGSIZE, dpi=DPI)
plt.scatter(X_prime[:, 0], X_prime[:, 1], color='blue')
plt.xlabel('$X_1$', fontsize=20)
plt.ylabel('$X_2$', fontsize=20)
plt.grid(True)

# Helper function to format coefficient
def format_coefficient(coef):
    exponent = int(np.floor(np.log10(abs(coef))))
    significand = coef / 10**exponent
    return f"{significand:.1f}e{exponent}"

# Add regression formulas for the perturbed dataset
formula_perturbed = f"Fit: $Y = {reg_perturbed.coef_[0]:.1f} X_1 + {reg_perturbed.coef_[1]:.1f} X_2$"
formula_perturbed_inliers = f"Inliers fit: $Y = {reg_perturbed_inliers.coef_[0]:.1f} X_1 + {format_coefficient(reg_perturbed_inliers.coef_[1])} \\times X_2$"
plt.text(0.5, 0.8, formula_perturbed, fontsize=14, horizontalalignment='center', verticalalignment='center',
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=1'))
plt.text(0.5, 0.6, formula_perturbed_inliers, fontsize=14, horizontalalignment='center', verticalalignment='center',
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=1'))

plt.title("Brittle Regression", fontsize=28)

ylim = plt.ylim()
plt.tight_layout()
plt.savefig('./results/figures/perturbed_dataset.png')


# Plotting the original dataset
plt.figure(figsize=FIGSIZE, dpi=DPI)
plt.scatter(X1, X2, color='blue')
plt.xlabel('$X_1$', fontsize=20)
plt.ylabel('$X_2$', fontsize=20)
plt.grid(True)

# Add regression formulas for the original dataset
formula_original = f"Fit: $Y = {reg_original.coef_[0]:.1f} X_1 - {-reg_original.coef_[1]:.1f} X_2$"
formula_original_inliers = f"Inliers fit: $Y = {reg_original_inliers.coef_[0]:.1f} X_1 + {reg_original_inliers.coef_[1]:.1f} X_2$"
plt.text(0.5, 0.8, formula_original, fontsize=14, horizontalalignment='center', verticalalignment='center',
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=1'))
plt.text(0.5, 0.6, formula_original_inliers, fontsize=14, horizontalalignment='center', verticalalignment='center',
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=1'))

plt.title("Robust Regression", fontsize=28)
plt.ylim(ylim)
plt.tight_layout()
plt.savefig('./results/figures/original_dataset.png')

# assert False

from src.robustness_auditor import AuditorConfig, RobustnessAuditor
from src.utils.linear_regression import LinearRegression as CustomLinearRegression


# Assuming the previously defined functions and data generation are available

# Define the analysis function
def run_analysis(df, formula, column_of_interest, output_dir, categorical_column: Optional[str] = None):
    """
    Run robustness analysis on a given DataFrame and regression formula.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    formula (str): Formula for the regression.
    column_of_interest (str): Column of interest for the regression.
    output_dir (str): Output directory for the analysis results.

    Returns:
    tuple: Summary of the analysis and the KZC21 removal sets.
    """
    config = AuditorConfig(output_dir=output_dir, reaverage=False, verbose=0)
    config.problem_1_params.k_max_factor = 0.99
    ra = RobustnessAuditor(
        CustomLinearRegression(data=df, formula=formula, column_of_interest=column_of_interest, special_categorical=categorical_column),
        config=config
    )

    ra.compute_all_bounds(categorical_aware=categorical_column is not None)
    summary = ra.summary()
    kzc = ra.removal_effect_lower_bounds.kzc21
    if np.isfinite(summary["KZC21"]):
        removal_sets = kzc.removal_sets[summary["KZC21"]]
    else:
        removal_sets = []

    print(summary)
    print(ra.linear_regression.model.summary())
    print(removal_sets)

    return summary, removal_sets


# Prepare the data for analysis
df_original = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})
df_perturbed = pd.DataFrame(X_prime, columns=['X1', 'X2'])
df_perturbed['Y'] = Y_prime
df_original_inliers = df_original[df_original['X2'] == 0]
df_perturbed_inliers = df_perturbed[df_original['X2'] == 0]

# Set output directory
CS_OUTPUT_DIR = './results/synthetic/2d_covariance_shift/'

# Run analysis on the original dataset
print("Original dataset analysis:")
summary_original, removal_sets_original = run_analysis(
    df_original, "Y ~ X1 + C(X2)", "X1", CS_OUTPUT_DIR, categorical_column="X2"
)
assert False
# Run analysis on the perturbed dataset
print("\nPerturbed dataset analysis:")
summary_perturbed, removal_sets_perturbed = run_analysis(df_perturbed, "Y ~ X1 + X2 - 1", "X1", CS_OUTPUT_DIR)

# Run analysis on the inliers of the original dataset
print("\nOriginal inliers dataset analysis:")
summary_original_inliers, removal_sets_original_inliers = run_analysis(df_original_inliers, "Y ~ X1 + X2 - 1", "X1",
                                                                       CS_OUTPUT_DIR)

# Run analysis on the inliers of the perturbed dataset
print("\nPerturbed inliers dataset analysis:")
summary_perturbed_inliers, removal_sets_perturbed_inliers = run_analysis(df_perturbed_inliers, "Y ~ X1 + X2 - 1", "X1",
                                                                         CS_OUTPUT_DIR)

