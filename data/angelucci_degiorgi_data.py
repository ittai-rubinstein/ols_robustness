from pathlib import Path
from typing import List, Optional, Tuple, Dict, NamedTuple

import statsmodels.api as sm
import numpy as np
import os
import pandas as pd

from src.utils.linear_regression import LinearRegression

CURRENT_FILE = Path(__file__).resolve()

FORMULA1 = 'Cind ~ treatment + avg_shock + hhhage + C(hhhsex) + C(p16) + C(hhhalpha) + C(hhhspouse) + yycali_1 + C(region_1) + indice + hectareas + vhhnum'
FORMULA2 = 'Cind ~ treatment + hhhage + C(hhhsex) + C(p16) + C(hhhalpha) + yycali_1 + indice + hectareas + vhhnum + avg_shock'

# Define the namedtuple type
class AngelucciDataset(NamedTuple):
    treatment: str
    time_period: int
    # data: pd.DataFrame
    regression: LinearRegression


# Conversion of 'hhhage' to numerical
HHHAGE_REPLACEMENTS= {
    '97 y más': 97,  # Replace '97 y más' with 97
    'no sabe': 98,  # Replace 'no sabe' with NaN
    'nr': 99  # Replace 'nr' with NaN
}

def load_angelucci_data(
        which_regression: int = 1
) -> List[AngelucciDataset]:
    """
    Loads the data from Angelucci and De Giorgi's study on the effect of cash transfers to poor households in Mexico, to
    measure the direct and indirect effects of the cash transfer program.

    There appears to be some minor confusion regarding the exact fit used by Angelucci and De Giorgi.
    On the one hand, in their paper they describe a regression formula similar to FORMULA2, and fitting with this
    formula reproduces their results.

    On the other hand, their reproduction materials use FORMULA1 which also controls for region and marital status.
    This is also the regression formula used in the AMIP benchmark of the same data.

    The differences in the regression are fairly minor, and we support both cases for a more complete benchmark.
    """

    # Load the data from the .dta file
    data = pd.read_stata( CURRENT_FILE.parent / "data_files" / "angelucci_table1.dta")
    # Filter data for Cind < 10000
    filtered_data = data[data['Cind'] < 10000]

    filtered_data.loc[:, 'hhhage'] = filtered_data['hhhage'].replace(HHHAGE_REPLACEMENTS)
    filtered_data = filtered_data.copy()
    filtered_data['hhhage_float'] = pd.to_numeric(filtered_data['hhhage'], errors='coerce').astype(float)

    # Drop the original 'hhhage' column
    filtered_data = filtered_data.drop(columns=['hhhage'])

    # Rename the 'hhhage_float' column to 'hhhage'
    filtered_data = filtered_data.rename(columns={'hhhage_float': 'hhhage'})

    if which_regression == 1:
        formula = FORMULA1
    else:
        formula = FORMULA2

    regressions = []
    for t in range(8, 11):
        current_samples = filtered_data[filtered_data["t"] == t]
        for treatment in ["p", "np"]:
            treated_samples = current_samples.copy()
            treated_samples["treatment"] = treated_samples[f"treat{treatment}"]
            regressions.append(
                AngelucciDataset(
                    treatment=treatment,
                    time_period=t,
                    regression=LinearRegression(
                        data=treated_samples,
                        formula=formula,
                        column_of_interest="treatment",
                        special_categorical="region_1" if which_regression == 1 else None
                    )
                )
            )
    return regressions



    # one_hot_encoded_frames = []
    # feature_to_columns_mapping = {}

    # for feature in CATEGORICAL_FEATURES:
    #     one_hot = pd.get_dummies(filtered_data[feature], prefix=feature, dummy_na=True,
    #                              dtype=float)  # Exclude NaN dummy column
    #
    #     # Drop the column corresponding to the purged value, if present
    #     for column, value in VALUES_TO_PURGE:
    #         if column == feature:
    #             purge_column = f"{feature}_{value}"
    #             one_hot.drop(columns=[purge_column], errors='ignore', inplace=True)
    #
    #     if purge_bad_columns:
    #         # Compute the intersection of BAD_CATEGORICAL_COLUMNS and the actual columns of one_hot to ensure we only attempt to drop existing columns
    #         columns_to_drop = BAD_CATEGORICAL_COLUMNS.intersection(one_hot.columns)
    #         # Drop the columns in place
    #         one_hot.drop(columns=list(columns_to_drop), inplace=True, errors='ignore')
    #
    #     # Remove category from each 1-hot encoding to avoid collinearity.
    #     if feature != keep_all_categories and not one_hot.empty:
    #         # Find the most common category among the remaining columns by summing each column and finding the min
    #         most_common_category = one_hot.sum().nlargest(1).idxmax()
    #
    #         # Drop the most common category to avoid collinearity
    #         one_hot.drop(columns=[most_common_category], inplace=True)
    #
    #     # Append the one-hot encoded DataFrame to the list
    #     one_hot_encoded_frames.append(one_hot)
    #     feature_to_columns_mapping[feature] = one_hot.columns.tolist()
    #
    # # Concatenate all one-hot encoded DataFrames
    # one_hot_encoded_data = pd.concat(one_hot_encoded_frames, axis=1)
    #
    # # Concatenate the new one-hot encoded columns back to the original dataframe (without the original categorical columns)
    # filtered_data_with_one_hot = pd.concat(
    #     [filtered_data.drop(columns=CATEGORICAL_FEATURES),
    #     one_hot_encoded_data], axis=1
    # )
    #
    # # If we removed a category from each feature to avoid collinearity, then we should add an intercept to the fit.
    # if keep_all_categories is None:
    #     filtered_data_with_one_hot = sm.add_constant(filtered_data_with_one_hot)
    #
    # # Initialize an empty list to store the updated columns of interest with one-hot encoded columns
    # columns_of_interest_one_hot = [LABEL]
    #
    # # Iterate over the columns of interest
    # for column in COLUMNS_OF_INTEREST:
    #     # Check if the column is in the mapping (i.e., it's a categorical column that was one-hot encoded)
    #     if column in feature_to_columns_mapping:
    #         # If it's a categorical column, extend the list with its one-hot encoded columns
    #         columns_of_interest_one_hot.extend(feature_to_columns_mapping[column])
    #     else:
    #         # If it's not a categorical column, just append it to the list
    #         columns_of_interest_one_hot.append(column)
    #
    #
    #
    # angelucci_datasets = [
    #     AngelucciDataset(
    #         treatment=treatment_type,
    #         time_period=time_period,
    #         data=filtered_data_with_one_hot[
    #             columns_of_interest_one_hot + [f'treat{treatment_type}']
    #         ][filtered_data_with_one_hot['t'] == time_period].dropna()
    #     ) for treatment_type in ["np", "p"] for time_period in range(8, 11)
    # ]
    #
    # return angelucci_datasets, feature_to_columns_mapping





