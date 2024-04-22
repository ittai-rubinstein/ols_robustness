import pandas as pd
import numpy as np
from src.robustness_auditor import AuditorConfig, RobustnessAuditor
from pathlib import Path
import os
from data.angelucci_degiorgi_data import load_angelucci_data



CURRENT_DIR = Path(__file__).resolve().parent
base_dir = CURRENT_DIR / "results" / "cash_transfers"
results = []
for log_hectareas in [True, False]:
    datasets = load_angelucci_data(which_regression=1, log_hectareas=log_hectareas)
    for dataset in datasets:
        name = f"t_{dataset.time_period}_treat_{dataset.treatment}_log_hectareas_{log_hectareas}"
        for i in range(2):
            print('*'*80)
        print(f"Running on {name=}")
        for i in range(2):
            print('*'*80)
        output_dir = base_dir / name
        config = AuditorConfig(output_dir=output_dir)
        ra = RobustnessAuditor(dataset.regression, config)
        ra.compute_all_bounds(categorical_aware=True)
        ra.plot_removal_effects()
        result = ra.summary()
        result["experiment"] = name
        results.append(result)
        df = pd.DataFrame(results)
        df = df[["experiment"] + [col for col in df.columns if col != "experiment"]]
        print(df)

        df.to_csv(base_dir / "results.csv")