import pandas as pd
import numpy as np

from data.eubank import load_eubank_regression
from data.martinez import load_martinez_linear_regression
from src.robustness_auditor import AuditorConfig, RobustnessAuditor
from pathlib import Path

regression = load_eubank_regression()

CURRENT_DIR = Path(__file__).resolve().parent
base_dir = CURRENT_DIR / "results" / "eubank"
results = []
for categorical_aware in [False, True]:
    ra = RobustnessAuditor(
        regression,
        AuditorConfig(
            output_dir=base_dir / f"{categorical_aware=}"
        )
    )
    ra.compute_all_bounds(categorical_aware)
    ra.plot_removal_effects()

    result = ra.summary()
    result["categorical_aware"] = categorical_aware
    results.append(result)
    df = pd.DataFrame(results)
    df = df[["categorical_aware"] + [col for col in df.columns if col != "experiment"]]
    print(df)

    df.to_csv(base_dir / "results.csv")

