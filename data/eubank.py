from pathlib import Path

import pandas as pd

from src.utils.linear_regression import LinearRegression

CURRENT_FILE = Path(__file__).resolve()
EUBANK_DATA_PATH = CURRENT_FILE.parent / "data_files" / "Eubank_black_perc.csv"

REGRESSION_FORMULA = "st_icpsr_MI_rate_black ~ vra2_x_post1965 + st_census_urban_perc + C(state) + C(year)"

def load_eubank_regression() -> LinearRegression:
    data = pd.read_csv(EUBANK_DATA_PATH)
    return LinearRegression(
        data,
        REGRESSION_FORMULA,
        # Select which categorical column is in more dire need of special treatment based on the size of its smallest bucket.
        special_categorical="year" if data.groupby('year').size().min() < data.groupby('state').size().min() else "state"
    )