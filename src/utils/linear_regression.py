from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper
from formulaic import model_matrix
from typing import List, Optional, Any

from src.categorical_data.categorical_data import split_and_normalize, perform_regression_and_append_residuals, \
    normalize_and_split_dataframes

def drop_zero_std_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate standard deviation of all columns
    std_devs = df.std()

    # Filter columns where standard deviation is not zero or the column name is 'Intercept'
    cols_to_keep = std_devs[(std_devs != 0) | (std_devs.index == 'Intercept')].index

    # Select only the columns that we want to keep in the DataFrame
    filtered_df = df[cols_to_keep]

    return filtered_df

@dataclass
class CategoricalAware:
    beta_e: float
    beta_e_sign: int
    X: np.ndarray
    residuals: np.ndarray
    axis_of_interest_normalized: np.ndarray
    num_samples: int
    dimension: int
    split_X: List[np.ndarray]
    split_R: List[np.ndarray]


# @dataclass
# class CategoricalAware:
#     column_name: str
#     split_X: List[np.ndarray] = field(default_factory=list)
#     split_R: List[np.ndarray] = field(default_factory=list)
#     split_Z: List[np.ndarray] = field(default_factory=list)

@dataclass
class RegressionArrays:
    X: np.ndarray
    # Y: np.ndarray
    Z: np.ndarray
    R: np.ndarray

class LinearRegression:
    data: pd.DataFrame
    formula: str
    column_of_interest: str
    special_categorical: Optional[str]
    regression_arrays: RegressionArrays
    model: RegressionResultsWrapper
    residuals: np.ndarray
    feature_array: pd.DataFrame
    labels: pd.Series
    Z: pd.Series
    # axis_of_interest: np.ndarray
    categorical_aware: Optional[CategoricalAware] = None

    def __init__(self, data: pd.DataFrame, formula: str, column_of_interest: str = None,
                 special_categorical: str = None):
        self.data = data
        self.formula = formula

        # Extract the first feature from the formula
        features_part = formula.split('~')[1].strip()
        first_feature = features_part.split('+')[0].strip()
        if '(' in first_feature:
            first_feature = first_feature.split('(')[-1].split(')')[0].strip()

        column_of_interest = column_of_interest or first_feature
        self.column_of_interest = column_of_interest
        self.special_categorical = special_categorical

        # Perform regression using statsmodels
        self.model = smf.ols(formula=self.formula, data=self.data).fit()
        self.residuals = self.model.resid.values  # np.ndarray of residuals

        # Extract X and Y using formulaic
        self.labels, self.feature_array = model_matrix(formula, data=self.data)

        self.feature_array = self.feature_array[[column_of_interest] + [c for c in self.feature_array.columns if c != column_of_interest]]
        self.feature_array = drop_zero_std_columns(self.feature_array)
        self.indices = self.feature_array.index  # Capture the indices of the rows retained in X


        # # Ensure the column of interest is the first column in X
        # col_index = list(self.feature_array.columns).index(self.column_of_interest)
        # self.feature_array = np.hstack([self.feature_array[:, col_index:col_index + 1], self.feature_array[:, :col_index], self.feature_array[:, col_index + 1:]])

        # Compute Z and axis_of_interest
        self.Z = self.feature_array[column_of_interest]  # First column is now the column of interest

        self.regression_arrays = RegressionArrays(
            X=self.feature_array.drop(columns='Intercept').to_numpy(),
            R=self.residuals, Z=self.Z.to_numpy()
        )

        if self.special_categorical:
            self._extract_categorical_aware()

    def _extract_categorical_aware(self):
        assert f'C({self.special_categorical})' in self.formula, "The special categorical column must be included in the formula."

        # Remove the intercept and categorical variable from the formula
        modified_formula = self.formula.replace(f'C({self.special_categorical}) +', '').replace(
            f'+ C({self.special_categorical})', '').replace(f'C({self.special_categorical})', '') + ' - 1'

        # Generate X_prime without the special categorical and intercept
        _, X_prime_df = model_matrix(modified_formula, data=self.data.loc[self.indices])
        X_prime_df = X_prime_df.drop(columns=['Intercept'], errors='ignore')  # Drop the intercept column if it exists
        X_prime_df = drop_zero_std_columns(X_prime_df)

        # Add one-hot encoding of the special categorical column to X_prime
        cat_encoded = pd.get_dummies(self.data.loc[self.indices, self.special_categorical],
                                     prefix=self.special_categorical)
        df = pd.concat([X_prime_df, cat_encoded], axis=1)

        # Extract the label column to the DataFrame
        label = self.formula.split("~")[0].strip()
        df[label] = self.data.loc[self.indices, label]

        # Split and normalize the DataFrame based on the categorical columns
        bucket_dfs = split_and_normalize(df, list(cat_encoded.columns))

        # Perform linear regression and append residuals
        coefficients = perform_regression_and_append_residuals(bucket_dfs, label)
        beta_e = np.abs(coefficients[self.column_of_interest])
        beta_e_sign = np.sign(coefficients[self.column_of_interest])

        # Normalize data and extract numpy arrays
        split_X, split_R, axis_of_interest_normalized = normalize_and_split_dataframes(
            bucket_dfs,
            self.column_of_interest,
            sign=beta_e_sign
        )
        X = np.vstack(split_X)
        residuals = np.concatenate(split_R)
        num_samples, dimension = X.shape

        # Store results in CategoricalAware
        self.categorical_aware = CategoricalAware(
            beta_e=beta_e,
            beta_e_sign=beta_e_sign,
            X=X,
            residuals=residuals,
            axis_of_interest_normalized=axis_of_interest_normalized,
            num_samples=num_samples,
            dimension=dimension,
            split_X=split_X,
            split_R=split_R
        )

# Example usage:
# df = pd.read_csv('your_data.csv')
# lr = LinearRegression(data=df, formula='label ~ column1 + C(column2)', special_categorical='column2')
