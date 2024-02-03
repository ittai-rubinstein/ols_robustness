import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from tqdm import notebook as ntqdm

import enum

class DataConventions(enum.Enum):
    PYTHON = 1
    R = 2


def load_martinez_XY(data_conventions: DataConventions = DataConventions.PYTHON):
    martinez = pd.read_csv('../robustness_auditing-main/data/martinez.csv')

    # The martinez table was originally generated in the R convention that one value of each one-hot encoding
    # should be omitted and that we add an intercept variable to avoid colinearity issues.
    # the python convention is to assign encodings to all values and use psuedo inverse / Ridge regularization to deal with colinearity.
    if data_conventions == DataConventions.PYTHON:
        # Check for the '_Icountryco_1' column and add it if not present
        if '_Icountryco_1' not in martinez.columns:
            # Initialize the column with 0
            martinez['_Icountryco_1'] = 0

            # Set to 1 for rows where 'countrycode' is 'AFG'
            martinez.loc[martinez['countrycode'] == 'AFG', '_Icountryco_1'] = 1

            # Reorder columns to place '_Icountryco_1' before '_Icountryco_2'
            col_order = martinez.columns.tolist()
            icountryco_2_index = col_order.index('_Icountryco_2')
            col_order.insert(icountryco_2_index, col_order.pop(col_order.index('_Icountryco_1')))
            martinez = martinez[col_order]

        # Check for the '_Iyear_1992' column and add it if not present
        if '_Iyear_1992' not in martinez.columns:
            # Initialize the column with 0
            martinez['_Iyear_1992'] = 0

            # Set to 1 for rows where 'year' is 1992
            martinez.loc[martinez['year'] == 1992, '_Iyear_1992'] = 1

            # Reorder columns to place '_Iyear_1992' before '_Iyear_1993'
            col_order = martinez.columns.tolist()
            iyear_1993_index = col_order.index('_Iyear_1993')
            col_order.insert(iyear_1993_index, col_order.pop(col_order.index('_Iyear_1992')))
            martinez = martinez[col_order]

    Y = martinez["lngdp14"].to_numpy()
    # grab only the columns we care about, and
    # reorder columns so that lndn13_fiw is last since
    # this is the coefficient whose sign we care about.
    # we are following Martinez, equation 6
    keys = martinez.columns.to_list()
    keys.remove("lndn13_fiw")
    if data_conventions:
        # Ittai's change: these two parameters are linearly dependent on other parameters of the fit.
        keys.remove('Dfree_avg9213')
        keys.remove('Dpfree_avg9213')
        keys.remove('Dnfree_avg9213')
    keys = keys[4:] + ["lndn13_fiw"]
    X = martinez[keys].to_numpy()
    return X, Y