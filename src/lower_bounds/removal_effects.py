from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.lower_bounds.amip import approximate_most_influential_pertrubation
from src.lower_bounds.lower_bounds import compute_removal_effects


@dataclass
class LowerBoundConfig:
    run_amip: bool = True
    run_kzcs21: bool = False
    verbose: bool = True


@dataclass
class LowerBoundResult:
    removal_effects: np.ndarray
    removal_sets: List[List[int]]

class RemovalEffectsLowerBound:
    _config: LowerBoundConfig
    amip: Optional[LowerBoundResult] = None
    kzcs21: Optional[LowerBoundResult] = None

    def __init__(
            self, X: np.ndarray, R: np.ndarray,
            axis_of_interest: np.ndarray, config: LowerBoundConfig
    ):
        self._config = config
        if config.run_amip:
            if config.verbose:
                print("Computing the AMIP Lower Bound...")
            amip_order, _ = approximate_most_influential_pertrubation(
                X, R, axis_of_interest
            )
            self.amip = LowerBoundResult(
                removal_effects=compute_removal_effects(
                    X, R, axis_of_interest, amip_order, verbose=config.verbose, normalized=True
                ),
                removal_sets= [(amip_order[:k]).tolist() for k in range(1, len(amip_order) + 1)]
            )
        if config.run_kzcs21:
            raise NotImplementedError()