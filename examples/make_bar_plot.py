import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# Load the Martinez data
current_file_path = Path(__file__).parent

# Define the paths to the CSV files relative to the current file path
cash_transfers_path = current_file_path / 'results/cash_transfers/results.csv'
martinez_path = current_file_path / 'results/martinez/results.csv'
eubank_path = current_file_path / 'results/eubank/results.csv'
ohie_path = current_file_path / 'results/ohie_categorical/iv/robustness_bounds.csv'

COLUMNS_FOR_TABLE = ["Runtime", "Memory"]  # Update this list as needed
COLUMNS_TO_KEEP = ['AMIP', 'Lower Bound', 'Paper', 'singularity', 'dimension', "KZC21", "Regression"] + COLUMNS_FOR_TABLE


def process_results(file_path):
    # Read the CSV file and drop the unnecessary column
    df = pd.read_csv(file_path).drop(columns=["Unnamed: 0"])

    # Print column names and descriptive statistics
    print(df.columns)
    print(df.describe())

    # Extract relevant details
    amip = df['AMIP'].iloc[0]  # Assuming AMIP value is constant across rows
    categorical_aware_bound = df.loc[df['categorical_aware'] == True, 'Lower Bound'].min()
    categorical_unaware_bound = df.loc[df['categorical_aware'] == False, 'Lower Bound'].min()
    dimension = df.loc[df['categorical_aware'] == True, 'dimension'].min()
    singularity = df.loc[df['categorical_aware'] == True, 'singularity'].min()
    kzc21 = df['KZC21'].min()
    runtime = df.loc[df['categorical_aware'] == True, 'Runtime'].min()
    memory = df.loc[df['categorical_aware'] == True, 'Memory'].min()

    summary = pd.DataFrame({
        'AMIP': [amip],
        'Categorical Aware Lower Bound': [categorical_aware_bound],
        'Categorical Unaware Lower Bound': [categorical_unaware_bound],
        'dimension': [dimension],
        'singularity': [singularity],
        'KZC21': [kzc21],
        'Runtime': [runtime],
        'Memory': [memory],
        'Regression': ""
    })

    return summary


# Process the Martinez results
martinez_summary = process_results(martinez_path)

# Process the Eubank results
eubank_summary = process_results(eubank_path)


def process_ohie_experiment(df, exp, exp_to_name):
    exp_data = df[df['experiment'].str.contains(exp)]
    min_amip = exp_data['AMIP'].min()
    min_lower_bound = exp_data['Lower Bound'].min()
    dimension = exp_data['dimension'].min()
    singularity = exp_data['singularity'].iloc[0] if "singularity" in exp_data.columns else np.inf
    kzc21 = exp_data["KZC21"][exp_data["KZC21"] != np.inf].min().astype(int)
    runtime = exp_data['Runtime'].sum()
    memory = exp_data['Memory'].max()

    return {
        'Experiment': exp,
        'AMIP': min_amip,
        'Lower Bound': min_lower_bound,
        'dimension': dimension,
        'singularity': singularity,
        'KZC21': kzc21,
        'Runtime': runtime,
        'Memory': memory,
        'Regression': exp_to_name.get(exp, "")
    }

def process_ohie_results(file_path, exp_to_name, order):
    df = pd.read_csv(file_path).drop(columns=["Unnamed: 0"])
    experiments = df['experiment'].str.replace('_end', '').str.replace('_out', '').unique()

    summary = [process_ohie_experiment(df, exp, exp_to_name) for exp in experiments]
    summary_df = pd.DataFrame(summary)

    summary_df['Experiment'] = pd.Categorical(summary_df['Experiment'], categories=order, ordered=True)
    summary_df = summary_df.sort_values('Experiment')
    summary_df['Paper'] = 'OHIE'
    return summary_df

# Define the mapping and order
OHIE_EXP_TO_NAME = {
    "health_genflip_bin_12m": "Health genflip",
    "health_notpoor_12m": "Health notpoor",
    "health_chgflip_bin_12m": "Health change flip",
    "notbaddays_tot_12m": "Not bad days total",
    "notbaddays_phys_12m": "Not bad days physical",
    "notbaddays_ment_12m": "Not bad days mental",
    "nodep_screen_12m": "Nodep Screen"
}

order = [
    "health_genflip_bin_12m",
    "health_notpoor_12m",
    "health_chgflip_bin_12m",
    "notbaddays_tot_12m",
    "notbaddays_phys_12m",
    "notbaddays_ment_12m",
    "nodep_screen_12m"
]

# Process the OHIE results
ohie_summary_df = process_ohie_results(ohie_path, OHIE_EXP_TO_NAME, order)
print(ohie_summary_df)




# Load the Cash Transfers data
cash_transfers_df = pd.read_csv(cash_transfers_path).drop(columns=["Unnamed: 0"])

# Unravel experiment names
def parse_experiment(name):
    parts = name.split('_')
    return {'Time Point': int(parts[1]), 'Treatment': parts[3], 'Log Hectares': parts[-1] == 'True', "Regression":f"({parts[3]}, {(int(parts[1]))})"}

parsed_experiments = cash_transfers_df['experiment'].apply(parse_experiment).apply(pd.Series)
cash_transfers_df = pd.concat([cash_transfers_df, parsed_experiments], axis=1)

# Sort the dataframe by Treatment and Time Point
cash_transfers_df.sort_values(by=['Treatment', 'Time Point'], ascending=[True, True], inplace=True)
# Cash Transfers - sorting by treatment and time
cash_transfers_df['Paper'] = 'Cash\nTransfer'


# Martinez - already summarized
martinez_df = martinez_summary
martinez_df['Paper'] = 'Nightlights'

# Eubank - already summarized
eubank_df = eubank_summary
eubank_df['Paper'] = 'VRA'




# Martinez Data Adjustment
martinez_df.rename(columns={'Categorical Aware Lower Bound': 'Lower Bound'}, inplace=True)
martinez_df['AMIP'] = martinez_df['AMIP']  # Ensure there is an 'AMIP' column if missing
martinez_df = martinez_df[COLUMNS_TO_KEEP]

eubank_df.rename(columns={'Categorical Aware Lower Bound': 'Lower Bound'}, inplace=True)
eubank_df = eubank_df[COLUMNS_TO_KEEP]


# ohie_summary_df = ohie_summary_df[COLUMNS_TO_KEEP]

# Filtering Cash Transfers data for plotting
plot_cash_transfers_df = cash_transfers_df[cash_transfers_df['Log Hectares'] == True]

# Concatenate all dataframes for plotting
all_data = pd.concat([martinez_df, plot_cash_transfers_df, ohie_summary_df])

# Plotting code as previously described, using the 'all_data' DataFrame
fig, ax = plt.subplots(figsize=(12, 10), dpi=500)
COLUMNS_TO_PLOT = ["AMIP", "KZC21", "Lower Bound"]
colors = ['cornflowerblue', 'blue', 'red']
papers = all_data['Paper'].unique()
x_pos = 0
width = 0.35
group_width = width * len(COLUMNS_TO_PLOT) + 0.1
group_gap = 1.5
positions = []
labels = []

# Lists for minor ticks and their labels
minor_positions = []
minor_labels = []

for paper in papers:
    paper_data = all_data[all_data['Paper'] == paper]
    start_pos = x_pos
    for i in range(len(paper_data)):
        ax.bar(x_pos, paper_data.iloc[i][COLUMNS_TO_PLOT[0]], width, color=colors[0])
        ax.bar(x_pos + width, paper_data.iloc[i][COLUMNS_TO_PLOT[1]], width, color=colors[1])
        ax.bar(x_pos + (2 * width), paper_data.iloc[i][COLUMNS_TO_PLOT[2]], width, color=colors[2])
        x_pos += group_width
        minor_positions.append(x_pos)
        n, d = map(int, paper_data.iloc[i]['dimension'].strip('()').split(', '))
        dimension_label = f"({n},{d})"
        minor_labels.append(dimension_label)
    end_pos = x_pos - group_width + width
    positions.append((start_pos + end_pos) / 2)
    labels.append(paper)

    x_pos += group_gap

ax.set_ylabel(r'$k_{\text{sign}}$', fontsize=28)
ax.set_title('Removals for Sign Change', fontsize=32)
# Setting major ticks and labels
ax.set_xticks(positions)
ax.set_xticklabels(labels, ha='center', va='top', fontsize=20, rotation=0)
ax.tick_params(axis='x', which='major', pad=45, length=0)  # Increase padding for major ticks and remove tick lines

# Adding minor ticks and labels
ax.set_xticks(minor_positions, minor=True)
ax.set_xticklabels(minor_labels, minor=True, rotation=-30, ha='right', va='top', fontsize=12)  # Adjust minor tick labels
ax.tick_params(axis='x', which='minor', pad=0, length=0, direction='out', bottom=True)  # Remove minor tick lines

ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.set_yscale('log')
ax.legend(['AMIP (Upper Bound)', "KZC21 (Upper Bound)", 'OHARE (Lower Bound)'], loc='upper right', fontsize=20, shadow=True, fancybox=True, framealpha=1, edgecolor='black')
plt.grid(which='major', linestyle='-', axis='y')
plt.grid(which='minor', linestyle='--', axis='y')
plt.tight_layout()
plt.ylim(1, plt.ylim()[1])

plt.savefig(current_file_path / "results" / "figures" / "bar_plot.png")



# Function to generate LaTeX table
def generate_latex_table(all_data, papers, COLUMNS_TO_PLOT, COLUMNS_FOR_TABLE):
    latex_lines = ["\\begin{tabular}{lcccccc" + ("c"*len(COLUMNS_FOR_TABLE)) + "}", "\\hline"]
    header = ["Paper", "Regression", "n", "d"] + COLUMNS_TO_PLOT[:-1] + ["OHARE"] + COLUMNS_FOR_TABLE
    latex_lines.append(" & ".join(header) + " \\\\")
    latex_lines.append("\\hline")

    for paper in papers:
        paper_data = all_data[all_data['Paper'] == paper]
        paper_label_added = False
        for i in range(len(paper_data)):
            if not paper_label_added:
                paper_label = paper
                paper_label_added = True
            else:
                paper_label = ""
            n, d = map(int, paper_data.iloc[i]['dimension'].strip('()').split(', '))
            regression = paper_data.iloc[i].get('Regression', '')
            amip = int(paper_data.iloc[i][COLUMNS_TO_PLOT[0]])
            kzc = paper_data.iloc[i][COLUMNS_TO_PLOT[1]]
            kzc = int(kzc)
            ohare = int(paper_data.iloc[i][COLUMNS_TO_PLOT[2]])
            additional_params = [str(paper_data.iloc[i].get(col, '')) for col in COLUMNS_FOR_TABLE]
            row = [paper_label, regression, n, d, amip, kzc, f"\\textbf{{{ohare}}}"] + additional_params
            latex_lines.append(" & ".join(map(str, row)) + " \\\\")
        latex_lines.append("\\hline")

    latex_lines.append("\\end{tabular}")

    return "\n".join(latex_lines)


# Function to convert memory in MiB to a human-readable format
def convert_memory_to_human_readable(mem_mib: float) -> str:
    if mem_mib < 1024:
        return f"{mem_mib:.2f} MiB"
    elif mem_mib < 1024 ** 2:
        return f"{(mem_mib / 1024):.2f} GiB"
    else:
        return f"{(mem_mib / (1024 ** 2)):.2f} TiB"


# Update the "Memory" column with human-readable values
all_data["Memory"] = all_data["Memory"].apply(lambda x: convert_memory_to_human_readable(x))

# Function to convert runtime in seconds to a human-readable format
def convert_runtime_to_human_readable(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    if minutes > 0:
        return f"{minutes} m {int(np.ceil(remaining_seconds))} s"
    else:
        return f"{int(np.ceil(remaining_seconds))} s"

all_data["Runtime"] = all_data["Runtime"].apply(lambda x: convert_runtime_to_human_readable(x))


# Generate LaTeX table
latex_table = generate_latex_table(all_data, papers, COLUMNS_TO_PLOT, COLUMNS_FOR_TABLE)

# Save LaTeX table to file
latex_table_path = current_file_path / "results" / "figures" / "table.tex"
with open(latex_table_path, "w") as f:
    f.write(latex_table)

# Print the file path to confirm
print(f"LaTeX table saved to: {latex_table_path}")
