from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
CURRENT_FILE = Path(__file__).resolve()
import enum

class DataConventions(enum.Enum):
    PYTHON = 1
    R = 2


