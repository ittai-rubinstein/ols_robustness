from pathlib import Path
from typing import Tuple, List
import statsmodels.api as sm

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data.data_loading_utils import DataConventions

CURRENT_FILE = Path(__file__).resolve()

def load_martinez_raw(keep_all_country_categories: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Loads the martinez dataset as a pandas dataframe.
    :param keep_all_country_categories: whether to add a column for all countries in one-hot encoding (set True if using
        one-hot aware analysis).
    :return:
    """
    martinez = pd.read_csv(CURRENT_FILE.parent / 'data_files' / 'martinez.csv')

    # The martinez table was originally generated in the R convention that one value of each one-hot encoding
    # should be omitted and that we add an intercept variable to avoid collinearity issues
    if keep_all_country_categories:
        # Initialize the column with 0
        martinez['_Icountryco_1'] = 0

        # Set to 1 for rows where 'countrycode' is 'AFG'
        martinez.loc[martinez['countrycode'] == 'AFG', '_Icountryco_1'] = 1

        # Reorder columns to place '_Icountryco_1' before '_Icountryco_2'
        col_order = martinez.columns.tolist()
        icountryco_2_index = col_order.index('_Icountryco_2')
        col_order.insert(icountryco_2_index, col_order.pop(col_order.index('_Icountryco_1')))
        martinez = martinez[col_order]
    else:
        martinez = sm.add_constant(martinez)

    label = "lngdp14"
    main_feature = "lndn13_fiw"

    # grab only the columns we care about, and
    # reorder columns so that lndn13_fiw is last since
    # this is the coefficient whose sign we care about.
    # we are following Martinez, equation 6
    keys = martinez.columns.to_list()

    # Remove parameters that are linearly dependent on other parameters of the fit.
    keys.remove('Dfree_avg9213')
    keys.remove('Dpfree_avg9213')
    keys.remove('Dnfree_avg9213')
    # Remove irrelevant columns
    keys.remove('countrycode')
    keys.remove('countryname')
    keys.remove('year')
    # Remove label and main feature:
    keys.remove(main_feature)
    keys.remove(label)
    keys.append(main_feature)
    keys = [label] + keys

    country_category_columns = [c for c in martinez.columns if c.startswith("_Icountryco")]
    return martinez[keys], country_category_columns


def load_martinez_XY():
    martinez, _ = load_martinez_raw(keep_all_country_categories=False)
    keys = martinez.columns
    Y = martinez[keys[0]].to_numpy()
    X = martinez[keys[1:]].to_numpy()
    return X, Y
    # martinez = pd.read_csv(CURRENT_FILE.parent / 'data_files' / 'martinez.csv')
    #
    # # The martinez table was originally generated in the R convention that one value of each one-hot encoding
    # # should be omitted and that we add an intercept variable to avoid collinearity issues.
    # # the python convention is to assign encodings to all values and use psuedo inverse / Ridge regularization to deal with collinearity.
    # if data_conventions == DataConventions.PYTHON:
    #     # Check for the '_Icountryco_1' column and add it if not present
    #     if '_Icountryco_1' not in martinez.columns:
    #         # Initialize the column with 0
    #         martinez['_Icountryco_1'] = 0
    #
    #         # Set to 1 for rows where 'countrycode' is 'AFG'
    #         martinez.loc[martinez['countrycode'] == 'AFG', '_Icountryco_1'] = 1
    #
    #         # Reorder columns to place '_Icountryco_1' before '_Icountryco_2'
    #         col_order = martinez.columns.tolist()
    #         icountryco_2_index = col_order.index('_Icountryco_2')
    #         col_order.insert(icountryco_2_index, col_order.pop(col_order.index('_Icountryco_1')))
    #         martinez = martinez[col_order]
    #
    #     # Check for the '_Iyear_1992' column and add it if not present
    #     if '_Iyear_1992' not in martinez.columns:
    #         # Initialize the column with 0
    #         martinez['_Iyear_1992'] = 0
    #
    #         # Set to 1 for rows where 'year' is 1992
    #         martinez.loc[martinez['year'] == 1992, '_Iyear_1992'] = 1
    #
    #         # Reorder columns to place '_Iyear_1992' before '_Iyear_1993'
    #         col_order = martinez.columns.tolist()
    #         iyear_1993_index = col_order.index('_Iyear_1993')
    #         col_order.insert(iyear_1993_index, col_order.pop(col_order.index('_Iyear_1992')))
    #         martinez = martinez[col_order]
    # elif data_conventions == DataConventions.R:
    #     # The dataset is already almost in the R convention.
    #     # This convention is that a one-hot encoding should have a column for each value but one
    #     # and an offset column of ones. This is done in order to avoid collinearity issues with multiple
    #     # one-hot encodings.
    #     # However, the R code does not automatically save this offset column when generating a csv file.
    #     martinez.insert(0, 'offset', 1)
    # else:
    #     raise NotImplementedError(f"Unrecognized data conventions: {data_conventions}")
    #
    # label = "lngdp14"
    # main_feature = "lndn13_fiw"
    #
    # # grab only the columns we care about, and
    # # reorder columns so that lndn13_fiw is last since
    # # this is the coefficient whose sign we care about.
    # # we are following Martinez, equation 6
    # keys = martinez.columns.to_list()
    #
    # # Ittai's change: these two parameters are linearly dependent on other parameters of the fit.
    # keys.remove('Dfree_avg9213')
    # keys.remove('Dpfree_avg9213')
    # keys.remove('Dnfree_avg9213')
    # # Remove irrelevant columns
    # keys.remove('countrycode')
    # keys.remove('countryname')
    # keys.remove('year')
    # # Remove label and main feature:
    # keys.remove(main_feature)
    # keys.remove(label)
    # keys.append(main_feature)
    # X = martinez[keys].to_numpy()
    # Y = martinez[label].to_numpy()
    # return X, Y