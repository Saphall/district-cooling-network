"""Turn a simulated pair into the files stored in its pair directory.

``target_table``  legacy pilot schema (Experiment 06): one parquet, target ETS only.
``network``       cloud-worker schema: every consumer and every pipe, per state.
"""

from __future__ import annotations

import pandas as pd

import district_cooling_network.bootstrap  # noqa: F401  (must precede dcngen imports)
from district_cooling_network.manifest import PairSpec
from district_cooling_network.pipeline.context import SeedContext, StaticContext
from district_cooling_network.pipeline.frames import result_to_frame
from district_cooling_network.pipeline.network_export import (
    export_network_observations,
    export_pipe_observations,
)
from district_cooling_network.pipeline.pairs import PairResult


def target_table(static, sc: SeedContext, pair: PairSpec, res: PairResult) -> dict[str, pd.DataFrame]:
    onset = float("nan") if res.onset_s is None else res.onset_s
    frames = []
    for state, result, head, link in (
        ("healthy", res.healthy, res.healthy_head, res.healthy_filter),
        ("fault", res.fault, res.fault_head, res.fault_filter),
    ):
        frames.append(
            result_to_frame(
                result,
                static.times,
                scenario_id=f"{pair.pair_id}_{state}",
                pair_id=pair.pair_id,
                seed=sc.scenario_seed,
                state=state,
                fsed=pair.fsed,
                severity_parameter=pair.severity_parameter,
                severity_value=pair.severity_value,
                target=sc.target,
                onset_s=onset,
                filter_link=link,
                plant_head=head,
            )
        )
    return {"data.parquet": pd.concat(frames, ignore_index=True)}


def network(static, sc: SeedContext, pair: PairSpec, res: PairResult) -> dict[str, pd.DataFrame]:
    files: dict[str, pd.DataFrame] = {}
    for state, result in (("healthy", res.healthy), ("fault", res.fault)):
        common = dict(
            result=result,
            scenario_id=f"{pair.pair_id}_{state}",
            pair_id=pair.pair_id,
            state=state,
            fsed=pair.fsed,
            target_consumer=sc.target,
            seed=sc.scenario_seed,
        )
        files[f"{state}_consumers.parquet"] = export_network_observations(
            **common, onset_s=res.onset_s
        )
        files[f"{state}_pipes.parquet"] = export_pipe_observations(**common)
    return files


EXPORTERS = {"target_table": target_table, "network": network}


def export_pair(
    static: StaticContext, sc: SeedContext, pair: PairSpec, res: PairResult
) -> dict[str, pd.DataFrame]:
    fmt = static.spec.get("export", "target_table")
    return EXPORTERS[fmt](static, sc, pair, res)
