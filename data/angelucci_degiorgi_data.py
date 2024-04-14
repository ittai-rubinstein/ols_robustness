from pathlib import Path
from typing import List, Optional, Tuple, Dict, NamedTuple

import statsmodels.api as sm
import numpy as np
import pandas as pd

CURRENT_FILE = Path(__file__).resolve()

# Define the namedtuple type
class AngelucciDataset(NamedTuple):
    treatment: str
    time_period: int
    data: pd.DataFrame

# Irregular feature-value pairs in Angelucci and De Giorgio's study that are purged in some of Broderick et al.'s analysis.
VALUES_TO_PURGE = [
    ('hhhage', '97 y más'),
    ('hhhage', 'nr'),
    ('hhhalpha', 'nr'),
    ('p16', 'nr'),
    ('hhhage', np.nan),
    ('hhhalpha', np.nan),
    ('hhhsex', np.nan),
    ('hhhspouse', np.nan),
    ('p16', np.nan),
    ('region_1', np.nan)
]
# Conversion of 'hhhage' to numerical
HHHAGE_REPLACEMENTS= {
    '97 y más': 97,  # Replace '97 y más' with 97
    'no sabe': 98,  # Replace 'no sabe' with NaN
    'nr': 99  # Replace 'nr' with NaN
}

# List of categorical features in Angelucci and De Giorgio's dataset
CATEGORICAL_FEATURES = ['hhhsex', 'p16', 'hhhalpha', 'hhhspouse', 'region_1']

# Additional categorical columns that need to be removed to avoid collinearity
BAD_CATEGORICAL_COLUMNS = {
    'hhhspouse_nr',
	'hhhsex_9.0',
	'region_1_nan',
	'p16_nr',
    'hhhalpha_nr',
    'hhhsex_nan',
    'hhhalpha_nan',
    'p16_nan',
    'hhhspouse_nan'
}
# Columns used in the regression of Angelucci and De Giorgio
COLUMNS_OF_INTEREST = [
    "avg_shock", "hhhage", "hhhsex", "p16", "hhhalpha", "hhhspouse",
    "yycali_1", "region_1", "indice", "hectareas", "vhhnum"
]
LABEL = "Cind"
def load_angelucci_data_raw(
        purge_bad_values: bool = False,
        purge_bad_columns: bool = True,
        keep_all_categories: Optional[str] = 'region_1'
) -> Tuple[List[AngelucciDataset], Dict[str, List[str]]]:
    """
    Loads the data from study by Angelucci and De Giorgio (2009) on the effects of cash stimulus in Mexico.
    The tables in this study contain many odd values (e.g., the head-of-household sex is sometimes listed as 9.0,
        and the age is sometimes "no sabe"=don't know or "97 y mas"=97 or more). We could not find a single
        strategy for dealing with unexpected values (e.g., retaining all samples yields regressions consistent
        with those reported by Broderick et al. (2023), but getting the number of samples reported in this same
        paper requires purging only some unexpected values, which in turn give us inconsistent regression results).

    Moreover, for our algorithms that explicitly deal with one-hot encodings, it is sometimes useful not to remove
        one of the categories (we typically do remove one of the categories to avoid collinearity issues).
        Leaving all categories for one feature is useful for our one-hot aware analysis.

    Another correction we do is that removing some specific columns from our regression seems to be necessary to avoid
        collinearity issues (e.g., sometimes we do not purge samples with "hhhspouse" = "nr", but do want to remove the
        column for this category in the one-hot encoding to avoid collinearity if this is a rare answer).

    Angelucci and De Giorgio report 6 regression results (based on direct / indirect treatment effect and 3 time periods).
    We return a list of 6 pandas dataframes, whose label is "Cind", and the rest of the columns are the regressors.
    We also return a mapping from the names of categorical features to the list of columns used for their one-hot encoding.
    :param purge_bad_values: should we purge the odd values that are purged in some of the analysis of Broderick et al?
    :param purge_bad_columns: should we remove additional columns that could lead to collinearity issues?
    :param keep_all_categories: should we use python conventions and keep all the categories of some features for 1-hot-aware analysis?
    :return: list of 6 pandas dataframes, whose label is "Cind", and the rest of the columns are the regressors, and
            mapping from the names of categorical features to the list of columns used for their one-hot encoding.
    """

    # Load the data from the .dta file
    data = pd.read_stata(os.path.join(CURRENT_FILE, 'data_files/angelucci_table1.dta'))
    # Filter data for Cind < 10000
    filtered_data = data[data['Cind'] < 10000]

    if purge_bad_values:
        for column, value in VALUES_TO_PURGE:
            if pd.isna(value):
                filtered_data = filtered_data.dropna(subset=[column])
            else:
                filtered_data = filtered_data[filtered_data[column] != value]

    filtered_data.loc[:, 'hhhage'] = filtered_data['hhhage'].replace(HHHAGE_REPLACEMENTS)
    filtered_data['hhhage_float'] = pd.to_numeric(filtered_data['hhhage'], errors='coerce').astype(float)

    one_hot_encoded_frames = []
    feature_to_columns_mapping = {}

    for feature in CATEGORICAL_FEATURES:
        one_hot = pd.get_dummies(filtered_data[feature], prefix=feature, dummy_na=True,
                                 dtype=float)  # Exclude NaN dummy column

        # Drop the column corresponding to the purged value, if present
        for column, value in VALUES_TO_PURGE:
            if column == feature:
                purge_column = f"{feature}_{value}"
                one_hot.drop(columns=[purge_column], errors='ignore', inplace=True)

        if purge_bad_columns:
            # Compute the intersection of BAD_CATEGORICAL_COLUMNS and the actual columns of one_hot to ensure we only attempt to drop existing columns
            columns_to_drop = BAD_CATEGORICAL_COLUMNS.intersection(one_hot.columns)
            # Drop the columns in place
            one_hot.drop(columns=list(columns_to_drop), inplace=True, errors='ignore')

        # Remove category from each 1-hot encoding to avoid collinearity.
        if feature != keep_all_categories and not one_hot.empty:
            # Find the most common category among the remaining columns by summing each column and finding the min
            most_common_category = one_hot.sum().nlargest(1).idxmax()

            # Drop the most common category to avoid collinearity
            one_hot.drop(columns=[most_common_category], inplace=True)

        # Append the one-hot encoded DataFrame to the list
        one_hot_encoded_frames.append(one_hot)
        feature_to_columns_mapping[feature] = one_hot.columns.tolist()

    # Concatenate all one-hot encoded DataFrames
    one_hot_encoded_data = pd.concat(one_hot_encoded_frames, axis=1)

    # Concatenate the new one-hot encoded columns back to the original dataframe (without the original categorical columns)
    filtered_data_with_one_hot = pd.concat(
        [filtered_data.drop(columns=CATEGORICAL_FEATURES),
        one_hot_encoded_data], axis=1
    )

    # If we removed a category from each feature to avoid collinearity, then we should add an intercept to the fit.
    if keep_all_categories is None:
        filtered_data_with_one_hot = sm.add_constant(filtered_data_with_one_hot)

    # Initialize an empty list to store the updated columns of interest with one-hot encoded columns
    columns_of_interest_one_hot = [LABEL]

    # Iterate over the columns of interest
    for column in COLUMNS_OF_INTEREST:
        # Check if the column is in the mapping (i.e., it's a categorical column that was one-hot encoded)
        if column in feature_to_columns_mapping:
            # If it's a categorical column, extend the list with its one-hot encoded columns
            columns_of_interest_one_hot.extend(feature_to_columns_mapping[column])
        else:
            # If it's not a categorical column, just append it to the list
            columns_of_interest_one_hot.append(column)



    angelucci_datasets = [
        AngelucciDataset(
            treatment=treatment_type,
            time_period=time_period,
            data=filtered_data_with_one_hot[
                columns_of_interest_one_hot + [f'treat{treatment_type}']
            ][filtered_data_with_one_hot['t'] == time_period].dropna()
        ) for treatment_type in ["np", "p"] for time_period in range(8, 11)
    ]

    return angelucci_datasets, feature_to_columns_mapping





