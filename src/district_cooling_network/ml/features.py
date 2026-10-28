"""Leakage-safe features and labels for the pilot-style detectors.

Rules from the pilot study:
* Inputs are physical measurements only. Generation metadata (FSED name, severity,
  scenario/pair id, seed, target, fault state/onset) must never be a feature.
* Filter-specific channels exist only for the clogged-filter case, so even their
  missingness would reveal the class; they are excluded.
* A stuck-actuator scenario is faulty only from its onset, not for every timestep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MEASUREMENT_FEATURES = (
    "desired_flow_m3s",
    "realized_flow_m3s",
    "supply_pressure_m",
    "return_pressure_m",
    "ets_dp_m",
    "supply_temp_C",
    "return_temp_C",
    "deltaT_K",
    "delivered_load_W",
    "plant_head_m",
)

LEAKY_COLUMNS = (
    "scenario_id",
    "pair_id",
    "seed",
    "state",
    "fsed",
    "severity_parameter",
    "severity_value",
    "target_consumer",
    "fault_onset_s",
    "fault_active",
    "fault_class",
    "filter_flow_m3s",
    "filter_headloss_m",
)


def add_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``fault_active`` (0/1) and ``fault_class`` (FSED name or ``normal``)."""
    out = frame.copy()
    is_fault = out["state"].eq("fault")
    onset = out["fault_onset_s"].fillna(-np.inf)  # no onset -> faulty from the start
    out["fault_active"] = (is_fault & (out["time_s"] >= onset)).astype(int)
    out["fault_class"] = np.where(out["fault_active"] == 1, out["fsed"], "normal")
    return out


def feature_matrix(frame: pd.DataFrame, columns: tuple[str, ...] = MEASUREMENT_FEATURES) -> pd.DataFrame:
    leaked = set(columns) & set(LEAKY_COLUMNS)
    if leaked:
        raise ValueError(f"leaky columns requested as features: {sorted(leaked)}")
    return frame[list(columns)]
