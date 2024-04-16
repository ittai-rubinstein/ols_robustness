from dataclasses import dataclass

from src.utils.linear_regression import LinearRegression


@dataclass
class IVRegression:
    endogenous_regression: LinearRegression
    outcome_regression: LinearRegression