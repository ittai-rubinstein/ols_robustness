import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Load the Martinez data
current_file_path = Path(__file__).parent

# Define the paths to the CSV files relative to the current file path
cash_transfers_path = current_file_path / 'results/cash_transfers/results.csv'
martinez_path = current_file_path / 'results/martinez/results.csv'
eubank_path = current_file_path / 'results/eubank/results.csv'
ohie_path = current_file_path / 'results/ohie_categorical/iv/robustness_bounds.csv'


COLUMNS_TO_KEEP = ['AMIP', 'Lower Bound', 'Paper', 'singularity', 'dimension']


martinez_df = pd.read_csv(martinez_path).drop(columns=["Unnamed: 0"])

# Extract relevant details
amip = martinez_df['AMIP'].iloc[0]  # Assuming AMIP value is constant across rows
categorical_aware_bound = martinez_df.loc[martinez_df['categorical_aware'] == True, 'Lower Bound'].min()
categorical_unaware_bound = martinez_df.loc[martinez_df['categorical_aware'] == False, 'Lower Bound'].min()

martinez_summary = pd.DataFrame({
    'AMIP': [amip],
    'Categorical Aware Lower Bound': [categorical_aware_bound],
    'Categorical Unaware Lower Bound': [categorical_unaware_bound],
    'dimension': [martinez_df.loc[martinez_df['categorical_aware'] == True, 'dimension'].min()],
    'singularity': [martinez_df.loc[martinez_df['categorical_aware'] == True, 'singularity'].min()]
})

eubank_df = pd.read_csv(eubank_path).drop(columns=["Unnamed: 0"])

# Extract relevant details
amip = eubank_df['AMIP'].iloc[0]  # Assuming AMIP value is constant across rows
categorical_aware_bound = eubank_df.loc[eubank_df['categorical_aware'] == True, 'Lower Bound'].min()
categorical_unaware_bound = eubank_df.loc[eubank_df['categorical_aware'] == False, 'Lower Bound'].min()

eubank_summary = pd.DataFrame({
    'AMIP': [amip],
    'Categorical Aware Lower Bound': [categorical_aware_bound],
    'Categorical Unaware Lower Bound': [categorical_unaware_bound],
    'dimension': [eubank_df.loc[eubank_df['categorical_aware'] == True, 'dimension'].min()],
    'singularity': [eubank_df.loc[eubank_df['categorical_aware'] == True, 'singularity'].min()]
})


ohie_df = pd.read_csv(ohie_path).drop(columns=["Unnamed: 0"])

# Process experiments
experiments = ohie_df['experiment'].str.replace('_end', '').str.replace('_out', '').unique()
ohie_summary = []

for exp in experiments:
    exp_data = ohie_df[ohie_df['experiment'].str.contains(exp)]
    min_amip = exp_data['AMIP'].min()
    min_lower_bound = exp_data['Lower Bound'].min()
    dimension = exp_data['dimension'].min()
    if "singularity" in exp_data.columns:
        singularity = exp_data['singularity'].iloc[0]
    else:
        singularity = np.inf
    #
    ohie_summary.append({'Experiment': exp, 'Min AMIP': min_amip, 'Min Lower Bound': min_lower_bound,
                         "dimension":dimension, "singularity":singularity})

ohie_summary_df = pd.DataFrame(ohie_summary)


# Load the Cash Transfers data
cash_transfers_df = pd.read_csv(cash_transfers_path).drop(columns=["Unnamed: 0"])

# Unravel experiment names
def parse_experiment(name):
    parts = name.split('_')
    return {'Time Point': int(parts[1]), 'Treatment': parts[3], 'Log Hectares': parts[-1] == 'True'}

parsed_experiments = cash_transfers_df['experiment'].apply(parse_experiment).apply(pd.Series)
cash_transfers_df = pd.concat([cash_transfers_df, parsed_experiments], axis=1)

# Sort the dataframe by Treatment and Time Point
cash_transfers_df.sort_values(by=['Treatment', 'Time Point'], ascending=[True, True], inplace=True)
print(cash_transfers_df.head())



import matplotlib.pyplot as plt

# Martinez - already summarized
martinez_df = martinez_summary
martinez_df['Paper'] = 'Nightlights'

# Eubank - already summarized
eubank_df = eubank_summary
eubank_df['Paper'] = 'Incarceration'

# Cash Transfers - sorting by treatment and time
cash_transfers_df['Paper'] = 'Cash Transfer'

# OHIE - sorting by a predefined order
order = [
    "health_genflip_bin_12m",
    "health_notpoor_12m",
    "health_chgflip_bin_12m",
    "notbaddays_tot_12m",
    "notbaddays_phys_12m",
    "notbaddays_ment_12m",
    "nodep_screen_12m"
]
ohie_summary_df['Experiment'] = pd.Categorical(ohie_summary_df['Experiment'], categories=order, ordered=True)
ohie_summary_df = ohie_summary_df.sort_values('Experiment')
ohie_summary_df['Paper'] = 'OHIE'
# Martinez Data Adjustment
martinez_df.rename(columns={'Categorical Aware Lower Bound': 'Lower Bound'}, inplace=True)
martinez_df['AMIP'] = martinez_df['AMIP']  # Ensure there is an 'AMIP' column if missing
martinez_df = martinez_df[COLUMNS_TO_KEEP]

eubank_df.rename(columns={'Categorical Aware Lower Bound': 'Lower Bound'}, inplace=True)
eubank_df = eubank_df[COLUMNS_TO_KEEP]

# OHIE Data Adjustment
ohie_summary_df.rename(columns={'Min AMIP': 'AMIP', 'Min Lower Bound': 'Lower Bound'}, inplace=True)
ohie_summary_df = ohie_summary_df[COLUMNS_TO_KEEP]

# Filtering Cash Transfers data for plotting
plot_cash_transfers_df = cash_transfers_df[cash_transfers_df['Log Hectares'] == True]

# Concatenate all dataframes for plotting
all_data = pd.concat([martinez_df, plot_cash_transfers_df, ohie_summary_df, eubank_df])



# Plotting code as previously described, using the 'all_data' DataFrame
fig, ax = plt.subplots(figsize=(12, 8), dpi=500)
COLUMNS_TO_PLOT = ["AMIP", "Lower Bound"]
colors = ['red', 'blue', 'green']
papers = all_data['Paper'].unique()
x_pos = 0
width = 0.35
group_width = width * len(COLUMNS_TO_PLOT) + 0.1
group_gap = 1.5
positions = []
labels = []

for paper in papers:
    paper_data = all_data[all_data['Paper'] == paper]
    start_pos = x_pos
    for i in range(len(paper_data)):
        ax.bar(x_pos, paper_data.iloc[i][COLUMNS_TO_PLOT[0]], width, color=colors[0])
        ax.bar(x_pos + width, paper_data.iloc[i][COLUMNS_TO_PLOT[1]], width, color=colors[1])
        # n, d = map(int, paper_data.iloc[i]['dimension'].strip('()').split(', '))
        # ax.bar(x_pos + 2*width, np.ceil(50 / np.log2(n)), width, color=colors[2])
        x_pos += group_width
    end_pos = x_pos - group_width + width
    positions.append((start_pos + end_pos) / 2)
    labels.append(paper)
    x_pos += group_gap

ax.set_ylabel(r'$k_{\text{sign}}$', fontsize=20)
ax.set_title('Removals for Sign Change', fontsize=28)
ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.set_yscale('log')
ax.legend(['AMIP Upper Bound', 'Lower Bound'], loc='upper right', fontsize=18, shadow=True, fancybox=True, framealpha=1, edgecolor='black')
# plt.show()
plt.grid(which='major', linestyle='-', axis='y')
plt.grid(which='minor', linestyle='--', axis='y')
plt.tight_layout()
plt.ylim(1, plt.ylim()[1])
plt.savefig(current_file_path / "results" / "figures" / "bar_plot.png")