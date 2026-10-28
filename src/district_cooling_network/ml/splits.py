"""Group-aware train/val/test splits.

Healthy and faulted twins of a pair share one exogenous load realisation, and all
pairs of a seed share another. Splitting by ``pair_id`` (default) keeps complete
simulation scenarios, including both twins, out of training; splitting by ``seed``
is stricter and also keeps each load realisation on one side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_by_group(
    frame: pd.DataFrame,
    group_col: str = "pair_id",
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    rng_seed: int = 0,
) -> dict[str, pd.DataFrame]:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    groups = np.array(sorted(frame[group_col].unique()))
    if len(groups) < 3:
        raise ValueError(
            f"need >= 3 distinct {group_col!r} values for train/val/test, found {len(groups)}"
        )
    shuffled = np.random.default_rng(rng_seed).permutation(groups)

    n = len(shuffled)
    n_val = max(1, round(fractions[1] * n))
    n_test = max(1, round(fractions[2] * n))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("not enough groups for the requested fractions")

    parts = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
    return {name: frame[frame[group_col].isin(ids)] for name, ids in parts.items()}
