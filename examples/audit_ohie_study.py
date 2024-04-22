import pandas as pd
import numpy as np

from data.ohie_data import load_ohie_regressions
from src.robustness_auditor import AuditorConfig, RobustnessAuditor
from pathlib import Path
import os
from data.angelucci_degiorgi_data import load_angelucci_data

print("Loading OHIE data...", end=" ", flush=True)
ols_regressions = load_ohie_regressions(iv=False)
iv_regressions = load_ohie_regressions(iv=True)
print("Done")

CURRENT_DIR = Path(__file__).resolve().parent
base_dir = CURRENT_DIR / "results" / "ohie"
results = []
print("Running robustness auditor on OLS regressors:")
for ols_regression in ols_regressions:
    for i in range(2):
        print('*'*80)
    print(f"Running on {ols_regression.name=}")
    for i in range(2):
        print('*'*80)
    output_dir = base_dir / "ols" / ols_regression.name
    config = AuditorConfig(output_dir=output_dir)
    ra = RobustnessAuditor(ols_regression.regression, config)
    ra.compute_all_bounds()
    ra.plot_removal_effects()
    result = ra.summary()
    result["experiment"] = ols_regression.name
    results.append(result)
    df = pd.DataFrame(results)
    df = df[["experiment"] + [col for col in df.columns if col != "experiment"]]
    print(df)
    df.to_csv(base_dir / "ols" / "robustness_bounds.csv")

results = []
print("Running robustness auditor on IV regressors:")
for iv_regression in iv_regressions:
    for (name, regression) in [
        (iv_regression.name + "_end", iv_regression.regression.endogenous_regression),
        (iv_regression.name + "_out", iv_regression.regression.outcome_regression)
    ]:
        for i in range(2):
            print('*'*80)
        print(f"Running on {name=}")
        for i in range(2):
            print('*'*80)
        output_dir = base_dir / "iv" /name
        config = AuditorConfig(output_dir=output_dir)
        ra = RobustnessAuditor(regression, config)
        ra.compute_all_bounds()
        ra.plot_removal_effects()
        result = ra.summary()
        result["experiment"] = name
        results.append(result)
        df = pd.DataFrame(results)
        df = df[["experiment"] + [col for col in df.columns if col != "experiment"]]
        print(df)
        df.to_csv(base_dir / "iv" / "robustness_bounds.csv")