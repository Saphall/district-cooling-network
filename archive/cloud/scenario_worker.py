
import os
import sys
import shutil
import uuid
from pathlib import Path

import numpy as np

from parallel_launcher import REQUIRED_FILES, scenario_complete


_SEED_CACHE = {}


def run_scenario(project_root, scenario, output_root, n_steps=144):
    """Execute one scenario, publishing output only after completion."""

    project_root = Path(project_root).resolve()
    output_root = Path(output_root).resolve()

    src = project_root / "code/dcngen/src"
    experiments = project_root / "rq1_experiments"

    for path in (src, experiments):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from cloud_config import load_cloud_config
    from controlled_runner import run_controlled_scenario
    from diagnostics.writer_with_desired_flow import write_scenario as diagnostic_writer
    from dcngen.orchestrate import poc
    from dcngen.loads.archetypes import ARCHETYPE_NAMES

    name = scenario["name"]
    final_folder = output_root / "scenarios" / name

    if scenario_complete(final_folder) and (final_folder / "_SUCCESS").is_file():
        return {"name": name, "status": "skipped"}

    if final_folder.exists():
        raise RuntimeError(
            f"Incomplete output already exists: {final_folder}. "
            "Manual inspection required; nothing overwritten."
        )

    seed = int(scenario["seed"])
    cache_key = (str(project_root), seed)

    if cache_key not in _SEED_CACHE:
        cfg = load_cloud_config(project_root)

        net = poc.load_ditec_network(
            Path(cfg.paths.ditec_data) / cfg.poc.network,
            cfg.poc.static_scenario_id,
        )

        static_seed, sensor_seed = (
            int(s)
            for s in np.random.default_rng(seed).integers(
                2**63, size=2
            )
        )

        consumers = sorted(
            j for j in net.junctions
            if net.draw.base_demand.get(j, 0.0) > 0.0
        )

        static_rng = np.random.default_rng(static_seed)
        archetypes = {
            j: str(a)
            for j, a in zip(
                consumers,
                static_rng.choice(
                    ARCHETYPE_NAMES,
                    size=len(consumers),
                ),
            )
        }

        mask_rng = np.random.default_rng(sensor_seed)
        dcn_for_mask = poc.build_dcn(net, cfg)
        sensor_mask = poc.draw_sensor_mask(
            dcn_for_mask, cfg, mask_rng
        )

        # Retain only one seed per worker to limit memory use.
        _SEED_CACHE.clear()
        _SEED_CACHE[cache_key] = (
            cfg, net, archetypes, sensor_mask
        )
        print(
            f"[CACHE] Loaded seed {seed}",
            flush=True
        )
    else:
        print(
            f"[CACHE] Reusing seed {seed}",
            flush=True
        )

    cfg, net, archetypes, sensor_mask = _SEED_CACHE[cache_key]

    staging_root = output_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)

    staging_name = f"{name}_{uuid.uuid4().hex}"
    staging_parent = staging_root / staging_name
    staging_parent.mkdir()

    try:
        result = run_controlled_scenario(
            name=name,
            seed=seed,
            net=net,
            archetypes=archetypes,
            cfg=cfg,
            out_dir=staging_parent,
            scenario_id=int(scenario["scenario_id"]),
            n_steps=int(n_steps),
            junction=scenario["junction"],
            side=scenario["side"],
            severity=scenario["severity"],
            onset=scenario["onset"],
            sensor_mask=sensor_mask,
            writer=diagnostic_writer,
        )

        staged_folder = staging_parent / name

        if not scenario_complete(staged_folder):
            raise RuntimeError(
                f"Scenario output incomplete: {staged_folder}"
            )

        final_folder.parent.mkdir(parents=True, exist_ok=True)

        if final_folder.exists():
            raise RuntimeError(
                f"Destination appeared during execution: {final_folder}"
            )

        # Mark the staged output complete before publishing it.
        # Staging and final folders must share a filesystem.
        marker = staged_folder / "_SUCCESS"
        marker.write_text(
            f"scenario_id={scenario['scenario_id']}\\n"
            f"name={name}\\n",
            encoding="utf-8"
        )

        # Publish only fully written output.
        os.rename(staged_folder, final_folder)

        return {
            "name": name,
            "status": "complete",
            "validation_passed": bool(result["report"].passed),
        }

    finally:
        if staging_parent.exists():
            shutil.rmtree(staging_parent)
