from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

from src.robustness_auditor import AuditorConfig, RobustnessAuditor

# Updated parameters
num_inliers = 1000
num_outliers = 10
epsilon = num_outliers / num_inliers
delta = 0.01

# Generate inliers
X2_inliers = delta * np.random.normal(0, 1, num_inliers)
X1_inliers = (X2_inliers / delta) + 0.1 * np.random.normal(0, 1, num_inliers)
Y_inliers = 2 * (X2_inliers / delta) - X1_inliers + (0.0 * np.random.normal(0, 1, num_inliers)
)
# Generate outliers with a scatter effect
X2_outliers = np.ones(num_outliers)
X1_outliers = 0.1 * np.random.normal(0, 1, num_outliers)
Y_outliers = np.zeros(num_outliers)

# Combine the datasets
X1 = np.concatenate([X1_inliers, X1_outliers])
X2 = np.concatenate([X2_inliers, X2_outliers])
Y = np.concatenate([Y_inliers, Y_outliers])
X = np.vstack((X1, X2)).T

# Fit linear regression on the combined dataset
model_combined = LinearRegression(fit_intercept=True)
model_combined.fit(X, Y)
Y_pred_combined = model_combined.coef_[0] * X1 + model_combined.coef_[1] * X2

# Fit linear regression on the inliers only
model_inliers = LinearRegression(fit_intercept=True)
model_inliers.fit(X[:num_inliers, :], Y[:num_inliers])
Y_pred_inliers = model_inliers.coef_[0] * X1_inliers + model_inliers.coef_[1] * X2_inliers

# Plotting the graph
plt.figure(figsize=(12, 8), dpi=400)
plt.scatter(X1_inliers, X2_inliers, color='blue', label='Inliers')
plt.scatter(X1_outliers, X2_outliers, color='red', label='Outliers', marker='x')
plt.xlabel('$X_1$', fontsize='xx-large')
plt.ylabel('$X_2$', fontsize='xx-large')
# plt.title('Updated Scatter plot of $X_1$ vs $X_2$ with $\epsilon=0.1$')
plt.legend(fontsize="x-large")
plt.grid(True)

formula_all = f"Combined fit: $Y = {model_combined.coef_[0]:.1f} X_1 + {model_combined.coef_[1]:.1f} X_2 + {model_combined.intercept_:.1f}$"
formula_in = f"Inlier fit: $Y = {model_inliers.coef_[0]:.1f} X_1 + {model_inliers.coef_[1]:.1f} X_2$ + {model_inliers.intercept_:.1f}"
plt.text(0.5, 0.65, formula_all + "\n" + "\n" + formula_in, fontsize='xx-large', horizontalalignment='center', verticalalignment='center',
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=1'))
plt.tight_layout()
# plt.show()
plt.savefig('./results/2d_cs.png')

# Format the results
# print(f"Combined fit: Y = {model_combined.coef_[0]:.3f} X_1 + {model_combined.coef_[1]:.3f} X_2")
# print(f"Inlier fit: Y = {model_inliers.coef_[0]:.3f} X_1 + {model_inliers.coef_[1]:.3f} X_2")
print(formula_all)
print(formula_in)

# Create a DataFrame from X and Y
df = pd.DataFrame(X, columns=['X1', 'X2'])
df['Y'] = Y

print(df.head())  # Display the first few rows of the DataFrame to verify

from src.utils.linear_regression import LinearRegression as MyLinearRegression

CURRENT_DIR = Path(__file__).resolve().parent
output_dir = CURRENT_DIR / "results" / "synthetic" / "2d_covariance_shift"
regression = MyLinearRegression(data=df, formula="Y~X1 + X2")
config = AuditorConfig(output_dir=output_dir)
ra = RobustnessAuditor(regression, config)

ra.compute_all_bounds(categorical_aware=False)
ra.plot_removal_effects()
print(ra.summary())

# print(ra.covariance_shift.upper_bounds.combined_bound[:10])
# print(ra.XR_bounds.upper_bounds.combined_bound[:10])
# print(ra.XZ_bounds.upper_bounds.combined_bound[:10])
# print(ra.upper_bound[:10])
#
# X = ra.parsed_data.X
# R = ra.parsed_data.R
# XR = ra.parsed_data.XR
# XZ = ra.parsed_data.XZ
# print(X.T @ X)
# print(X[:5, :])
# print(X[-5:, :])
# print(R[:5])
# print(R[-5:])
# print(np.linalg.norm(np.sum(XR[-10:, :], axis=0)))
# print(np.linalg.norm(np.sum(XZ[-10:, :], axis=0)))
# print(X[:-10, :].T @ X[:-10, :])
