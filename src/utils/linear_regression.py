from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper
from formulaic import model_matrix
from typing import List, Optional, Any


@dataclass
class CategoricalAware:
    column_name: str
    split_X: List[np.ndarray] = field(default_factory=list)
    split_R: List[np.ndarray] = field(default_factory=list)
    split_Z: List[np.ndarray] = field(default_factory=list)

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
            assert f'C({self.special_categorical})' in formula, "The special categorical column must be included in the formula."
            # Modify the formula to exclude the categorical and intercept
            modified_formula = formula.replace(f'C({self.special_categorical}) +', '').replace(
                f'+ C({self.special_categorical})', '').replace(f'C({self.special_categorical})', '') + ' - 1'
            _, X_prime = model_matrix(modified_formula, data=self.data.loc[self.indices])
            X_prime = X_prime.values

            # Fetching unique categories from the filtered data
            categories = self.data.loc[self.indices, self.special_categorical].unique()
            split_X, split_R, split_Z = [], [], []
            for category in categories:
                category_indices = self.data.loc[self.indices, self.special_categorical] == category
                split_X.append(X_prime[category_indices, :])
                split_R.append(self.residuals[category_indices])
                split_Z.append(self.Z[category_indices])

            # Storing categorical splits in a structured dataclass
            self.categorical_aware = CategoricalAware(
                column_name=self.special_categorical,
                split_X=split_X,
                split_R=split_R,
                split_Z=split_Z
            )

            self.categorical_aware = CategoricalAware(column_name=self.special_categorical, split_X=split_X,
                                                      split_R=split_R, split_Z=split_Z)

# Example usage:
# df = pd.read_csv('your_data.csv')
# lr = LinearRegression(data=df, formula='label ~ column1 + C(column2)', special_categorical='column2')
