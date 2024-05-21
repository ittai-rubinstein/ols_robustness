from pathlib import Path
from matplotlib import  pyplot as plt
import numpy as np
import pandas as pd

from src.problem_1 import Problem1Params
from src.robustness_auditor import RobustnessAuditor, AuditorConfig
from src.utils.linear_regression import LinearRegression

# Parameters
n = 4000
d = 50

np.random.seed(1)
# Generate features
X = np.random.normal(size=(n, d)) / np.sqrt(n)

# Generate labels with no correlation to features
Y = np.random.normal(size=n) / np.sqrt(n)

# Create a DataFrame for the dataset
feature_columns = [f'X{i+1}' for i in range(d)]
df = pd.DataFrame(X, columns=feature_columns)
df['Y'] = Y

# Generate regression formula
regression_formula = 'Y ~ ' + ' + '.join(feature_columns)

regression = LinearRegression(data=df, formula=regression_formula)

CURRENT_DIR = Path(__file__).resolve().parent
base_dir = CURRENT_DIR / "results" / "synthetic" / "normally_distributed_acre"
ra = RobustnessAuditor(
    regression,
    AuditorConfig(
        output_dir=base_dir,
        verbose=0,
        problem_1_params=Problem1Params(use_spectral=False)
    )
)

ra.compute_all_bounds()

ra_spectral = RobustnessAuditor(
    regression,
    AuditorConfig(
        output_dir=base_dir,
        verbose=0,
        problem_1_params=Problem1Params(use_spectral=True)
    )
)

ra_spectral.compute_all_bounds()
ra.compute_freund_hopkins()

print(ra.summary())
print(ra_spectral.summary())

two_sigma = 2 * ra.parsed_data.delta_beta_e
two_sigma: float
ACRE = ra.upper_bound
ACRE_spectral = ra_spectral.upper_bound
ACRE: np.ndarray
KZC21 = ra.removal_effect_lower_bounds.kzc21.removal_effects
KZC21: np.ndarray
AMIP = ra.removal_effect_lower_bounds.amip.removal_effects
AMIP: np.ndarray
Freund_Hopkins = ra.freund_and_hopkins_upper_bound
Freund_Hopkins: np.ndarray
# k = np.arange(1, ra.parsed_data.num_samples // 10)
k = np.arange(1, 150)


# Determine k_2_sigma: the first value of k for which ACRE >= two_sigma
k_2_sigma = k[np.where(ACRE >= two_sigma)[0][0]]

# Create the plot
plt.figure(figsize=(12, 10), dpi=500)
LINEWIDTH = 5
# Plotting ACRE in red
plt.plot(k[:len(ACRE)], ACRE[:len(k)], color='red', linewidth=LINEWIDTH, label='ACRE')

# Plotting Freund_Hopkins in green
plt.plot(k[:len(Freund_Hopkins)], Freund_Hopkins[:len(k)], color='green', linewidth=LINEWIDTH, label='Freund & Hopkins')

# Plotting KZC21 in a muted blue shade
plt.plot(k[:len(KZC21)], KZC21[:len(k)], color='dodgerblue', linestyle='--', linewidth=LINEWIDTH, label='KZC21')

# Plotting AMIP in a different shade of blue with a different line style
plt.plot(k[:len(AMIP)], AMIP[:len(k)], color='deepskyblue', linestyle=':', linewidth=LINEWIDTH, label='AMIP')

# Adding the horizontal line for two_sigma
plt.axhline(y=two_sigma, color='black', linestyle='-', linewidth=LINEWIDTH, label='$2\sigma$')

# Adding the vertical line at k_2_sigma
plt.axvline(x=k_2_sigma, color='black', linestyle='--', linewidth=LINEWIDTH, label='$k_{2\sigma}$')

# Setting log-log scale
plt.xscale('log')
plt.yscale('log')

# Adding labels and title
plt.xlabel('$k$', fontsize=28)
plt.ylabel('$\Delta_k(e)$', fontsize=28)
plt.title('ACRE on Synthetic Data', fontsize=32)

# Adding grid
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

# Setting legend
plt.legend(fontsize=24)

# Setting tick parameters for better readability
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=12)

plt.tight_layout()
# Show plot
plt.savefig(CURRENT_DIR / "results" / "figures" / "synthetic_acre.png")
plt.show()



# Create the plot
plt.figure(figsize=(12, 10), dpi=500)
LINEWIDTH = 5
# Plotting ACRE in red
plt.plot(k[:len(ACRE)], ACRE[:len(k)], '--', color='red', linewidth=LINEWIDTH, label=r'$\text{ACRE}_\text{RTI}$')
plt.plot(k[:len(ACRE_spectral)], ACRE_spectral[:len(k)], '-', color='red', linewidth=LINEWIDTH, label=r'$\text{ACRE}_\text{spectral}$')

# Plotting Freund_Hopkins in green
plt.plot(k[:len(Freund_Hopkins)], Freund_Hopkins[:len(k)], color='green', linewidth=LINEWIDTH, label='Freund & Hopkins')

# Plotting KZC21 in a muted blue shade
plt.plot(k[:len(KZC21)], KZC21[:len(k)], color='dodgerblue', linestyle='--', linewidth=LINEWIDTH, label='KZC21')

# Plotting AMIP in a different shade of blue with a different line style
plt.plot(k[:len(AMIP)], AMIP[:len(k)], color='deepskyblue', linestyle=':', linewidth=LINEWIDTH, label='AMIP')

# Adding the horizontal line for two_sigma
plt.axhline(y=two_sigma, color='black', linestyle='-', linewidth=LINEWIDTH, label='$2\sigma$')

# Adding the vertical line at k_2_sigma
plt.axvline(x=k_2_sigma, color='black', linestyle='--', linewidth=LINEWIDTH, label='$k_{2\sigma}$')

# Setting log-log scale
plt.xscale('log')
plt.yscale('log')

# Adding labels and title
plt.xlabel('$k$', fontsize=28)
plt.ylabel('$\Delta_k(e)$', fontsize=28)
plt.title('ACRE on Synthetic Data', fontsize=32)

# Adding grid
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

# Setting legend
plt.legend(fontsize=24)

# Setting tick parameters for better readability
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=12)

plt.tight_layout()
# Show plot
plt.savefig(CURRENT_DIR / "results" / "figures" / "synthetic_acre_w_spectral.png")
plt.show()