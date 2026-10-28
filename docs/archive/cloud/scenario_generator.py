
from pathlib import Path
import json
import numpy as np

from dcngen.topology.dcnifier import build_dcn
from dcngen.orchestrate.scenario import run_eps_ntu_scenario

from fault_runner import run_fault, run_filter_pair
from network_exporter import (
    export_network_observations,
    export_pipe_observations,
)


def generate_pair(
    net, cfg, loads, ua_healthy, pipe_U,
    times, target, seed, fsed, severity, output_root,
):
    """Generate and save one matched healthy/fault pair."""

    output_root = Path(output_root)
    n_steps = len(times)

    pair_id = (
        f"seed{seed}_target{target}_"
        f"{fsed}_{severity:g}"
    )

    pair_dir = output_root / pair_id

    if (pair_dir / "_SUCCESS").exists():
        return {"pair_id": pair_id, "status": "skipped"}

    if fsed == "clogged_filter":
        healthy, fault = run_filter_pair(
            net, cfg, loads, ua_healthy, pipe_U,
            target=target,
            k_clogged=severity,
            n_steps=n_steps,
        )
        onset_s = None

    else:
        healthy_dcn = build_dcn(net, cfg)

        healthy = run_eps_ntu_scenario(
            healthy_dcn,
            cfg,
            loads,
            n_steps=n_steps,
            ua=ua_healthy,
            pipe_U=pipe_U,
        )

        fault = run_fault(
            net, cfg, loads, ua_healthy, pipe_U,
            times, target, fsed, severity,
        )

        onset_s = (
            float(times[
                max(
                    1,
                    min(
                        int(round(severity * (n_steps - 1))),
                        n_steps - 1,
                    ),
                )
            ])
            if fsed == "stuck_actuator"
            else None
        )

    if not np.all(healthy.converged):
        raise RuntimeError(f"{pair_id}: healthy did not converge")

    if not np.all(fault.converged):
        raise RuntimeError(f"{pair_id}: fault did not converge")

    pair_dir.mkdir(parents=True, exist_ok=True)

    for state, sim in [
        ("healthy", healthy),
        ("fault", fault),
    ]:
        scenario_id = f"{pair_id}_{state}"

        common = dict(
            result=sim,
            scenario_id=scenario_id,
            pair_id=pair_id,
            state=state,
            fsed=fsed,
            target_consumer=target,
            seed=seed,
        )

        consumer_df = export_network_observations(
            **common,
            onset_s=onset_s,
        )

        pipe_df = export_pipe_observations(**common)

        consumer_df.to_parquet(
            pair_dir / f"{state}_consumers.parquet",
            index=False,
        )

        pipe_df.to_parquet(
            pair_dir / f"{state}_pipes.parquet",
            index=False,
        )

    metadata = {
        "pair_id": pair_id,
        "seed": int(seed),
        "target_consumer": str(target),
        "fsed": fsed,
        "severity": float(severity),
        "n_steps": int(n_steps),
        "onset_s": onset_s,
        "status": "complete",
    }

    (pair_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    (pair_dir / "_SUCCESS").write_text("complete\n")

    return {"pair_id": pair_id, "status": "complete"}
