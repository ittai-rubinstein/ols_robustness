from typing import Union, NamedTuple

from src.utils.iv_regression import IVRegression
from src.utils.linear_regression import LinearRegression


class NamedRegression(NamedTuple):
    name: str
    regression: Union[LinearRegression, IVRegression]