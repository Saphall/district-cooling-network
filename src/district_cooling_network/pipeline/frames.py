"""Result -> long-table conversion for the dataset (legacy Experiment 06 schema)."""

from __future__ import annotations

import numpy as np
import pandas as pd

import district_cooling_network.bootstrap  # noqa: F401  (must precede dcngen imports)


def result_to_frame(
    result,
    times: np.ndarray,
    *,
    scenario_id: str,
    pair_id: str,
    seed: int,
    state: str,
    fsed: str,
    severity_parameter: str,
    severity_value: float,
    target: str,
    onset_s: float = np.nan,
    filter_link: str | None = None,
    plant_head: float = np.nan,
) -> pd.DataFrame:
    """Target-ETS trajectory with raw physical channels.

    Healthy-fault residuals are deliberately not computed here: the healthy twin is
    privileged simulator ground truth, not a detector input.
    """
    target = str(target)

    q = result.consumer_flow[target].to_numpy(dtype=float)
    desired = result.desired_flow[target].to_numpy(dtype=float)
    ps = result.pressure_s[target].to_numpy(dtype=float)
    pr = result.pressure_r[target].to_numpy(dtype=float)
    ts = result.supply_temp[target].to_numpy(dtype=float)
    tr = result.return_temp[target].to_numpy(dtype=float)
    delivered = result.delivered_load[target].to_numpy(dtype=float)
    unmet = result.unmet[target].to_numpy(dtype=bool)
    converged = np.asarray(result.converged, dtype=bool)

    df = pd.DataFrame(
        {
            "scenario_id": scenario_id,
            "pair_id": pair_id,
            "seed": int(seed),
            "state": state,
            "fsed": fsed,
            "severity_parameter": severity_parameter,
            "severity_value": float(severity_value),
            "target_consumer": target,
            "fault_onset_s": onset_s,
            "time_s": times.astype(float),
            "desired_flow_m3s": desired,
            "realized_flow_m3s": q,
            "supply_pressure_m": ps,
            "return_pressure_m": pr,
            "ets_dp_m": ps - pr,
            "supply_temp_C": ts,
            "return_temp_C": tr,
            "deltaT_K": np.abs(tr - ts),
            "delivered_load_W": delivered,
            "unmet": unmet,
            "converged": converged,
            "plant_head_m": plant_head,
        }
    )

    if filter_link is not None:
        df["filter_flow_m3s"] = result.pipe_flow[filter_link].to_numpy(dtype=float)
        df["filter_headloss_m"] = result.pipe_headloss[filter_link].to_numpy(dtype=float)
    else:
        df["filter_flow_m3s"] = np.nan
        df["filter_headloss_m"] = np.nan

    return df
