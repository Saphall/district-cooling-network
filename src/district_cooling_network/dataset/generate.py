"""Resumable dataset generation.

Each counterfactual pair is computed independently and committed atomically to
``datasets/<id>/pairs/<pair_id>/`` (``data.parquet`` + ``meta.json``). A re-run
computes only pairs that are not already committed, so interruptions, restarts
(Colab disconnects) and dataset extensions never regenerate finished work.
"""

from __future__ import annotations

import copy
import os
import platform
import shutil
import socket
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from district_cooling_network.config import dcngen_master_seed, load_dcngen_lock
from district_cooling_network.fsutil import (
    atomic_path,
    canonical_json,
    read_json,
    sha256_file,
    sha256_text,
    write_json_atomic,
)
from district_cooling_network.manifest import (
    PairSpec,
    assert_writable,
    build_pair_specs,
    derive_seeds,
    failures_dir,
    global_params,
    pairs_dir,
    seeds_dir,
    select_pending,
    write_manifest,
)
from district_cooling_network.paths import Workspace, get_workspace


class Runner(Protocol):
    """The simulation side; swapped for a fake in tests."""

    def static_context(self, spec: dict[str, Any], static_seed: int, ws: Workspace) -> Any: ...

    def seed_context(self, static: Any, scenario_seed: int, cached_target: str | None) -> Any: ...

    def run_pair(
        self, static: Any, sc: Any, pair: PairSpec
    ) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
        """Files (name -> table) to store in the pair directory, plus extra meta."""
        ...


class DcngenRunner:
    """Default runner: dcngen via ``district_cooling_network.pipeline`` (imported lazily)."""

    def static_context(self, spec, static_seed, ws):
        from district_cooling_network.pipeline.context import build_static_context

        return build_static_context(spec, static_seed, ws)

    def seed_context(self, static, scenario_seed, cached_target):
        from district_cooling_network.pipeline.context import build_seed_context

        return build_seed_context(static, scenario_seed, cached_target)

    def run_pair(self, static, sc, pair):
        from district_cooling_network.pipeline.export import export_pair
        from district_cooling_network.pipeline.pairs import run_pair

        result = run_pair(static, sc, pair)
        extra = {} if result.onset_s is None else {"fault_onset_s": result.onset_s}
        return export_pair(static, sc, pair, result), extra


@dataclass
class GenerationReport:
    requested: int = 0
    completed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    skipped_existing: int = 0

    @property
    def ok(self) -> bool:
        return not self.failed


def qa_checks(files: dict[str, pd.DataFrame], n_steps: int, export: str) -> dict[str, Any]:
    simulated = [f for f in files.values() if "converged" in f]
    checks: dict[str, Any] = {
        "all_converged": all(bool(f["converged"].all()) for f in simulated),
        "both_states": {s for s in ("healthy", "fault") if any(n.startswith(s) for n in files)}
        == {"healthy", "fault"}
        if export == "network"
        else set(simulated[0]["state"].unique()) == {"healthy", "fault"},
    }
    if export == "network":
        checks["row_count"] = all(len(f) % n_steps == 0 and len(f) > 0 for f in files.values())
    else:
        checks["row_count"] = len(files["data.parquet"]) == 2 * n_steps
    return {"passed": all(checks.values()), "checks": checks}


def _commit_pair(
    ws: Workspace,
    spec: dict[str, Any],
    pair: PairSpec,
    files: dict[str, pd.DataFrame],
    extra: dict[str, Any],
    duration_s: float,
    log: Callable[[str], None],
) -> bool:
    """Atomically publish a pair directory. Returns False if another worker won."""
    dataset_id = spec["dataset_id"]
    final = pairs_dir(ws, dataset_id) / pair.pair_id
    assert_writable(ws, final)
    final.parent.mkdir(parents=True, exist_ok=True)

    lock = load_dcngen_lock(ws)
    qa = qa_checks(files, int(spec["n_steps"]), spec.get("export", "target_table"))
    meta = {
        "dataset_id": dataset_id,
        "pair_id": pair.pair_id,
        "spec_hash": pair.spec_hash,
        "fsed": pair.fsed,
        "severity_parameter": pair.severity_parameter,
        "severity_value": pair.severity_value,
        "seed_index": pair.seed_index,
        "scenario_seed": pair.scenario_seed,
        "target_consumer": str(next(iter(files.values()))["target_consumer"].iloc[0]),
        "export": spec.get("export", "target_table"),
        "rows": sum(len(f) for f in files.values()),
        "qa": qa,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "duration_s": round(duration_s, 3),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dcngen_commit": lock.commit,
        "dcngen_patches": {p.file: p.sha256 for p in lock.patches},
        **extra,
    }

    tmp = final.parent / f".tmp-{pair.pair_id}-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        meta["files"] = {}
        for name, table in files.items():
            table.to_parquet(tmp / name, index=False)
            meta["files"][name] = sha256_file(tmp / name)
        write_json_atomic(tmp / "meta.json", meta)
        (tmp / "_SUCCESS").write_text("complete\n")  # written last

        existing_meta = final / "meta.json"
        if (
            existing_meta.exists()
            and (final / "_SUCCESS").exists()
            and read_json(existing_meta).get("spec_hash") == pair.spec_hash
        ):
            log(f"  {pair.pair_id}: another worker committed first; discarding duplicate")
            return False
        if final.exists():
            # Stale pair being rebuilt: archive, never delete.
            archive = ws.dataset_dir(dataset_id) / "pairs_stale"
            archive.mkdir(parents=True, exist_ok=True)
            final.rename(archive / f"{pair.pair_id}-{time.strftime('%Y%m%dT%H%M%S')}")
        try:
            tmp.rename(final)
        except OSError:
            log(f"  {pair.pair_id}: another worker committed first; discarding duplicate")
            return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    (failures_dir(ws, dataset_id) / f"{pair.pair_id}.json").unlink(missing_ok=True)
    return True


def _seed_target(
    ws: Workspace, spec: dict[str, Any], seed_index: int, scenario_seed: int
) -> tuple[Path, str | None, str]:
    path = seeds_dir(ws, spec["dataset_id"]) / f"seed{seed_index:02d}.json"
    ghash = sha256_text(canonical_json(global_params(spec, ws)))
    if path.exists():
        cached = read_json(path)
        if cached.get("scenario_seed") == scenario_seed and cached.get("global_hash") == ghash:
            return path, cached["target_consumer"], ghash
    return path, None, ghash


def generate(
    spec: dict[str, Any],
    ws: Workspace | None = None,
    *,
    runner: Runner | None = None,
    shard: tuple[int, int] = (0, 1),
    only_fsed: set[str] | None = None,
    limit: int | None = None,
    retry_failed: bool = True,
    rebuild_stale: bool = False,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
) -> GenerationReport:
    ws = ws or get_workspace()
    dataset_id = spec["dataset_id"]
    assert_writable(ws, ws.dataset_dir(dataset_id))

    pairs = {p.pair_id: p for p in build_pair_specs(spec, ws)}
    frame = write_manifest(spec, ws)
    todo = select_pending(
        frame,
        shard=shard,
        only_fsed=only_fsed,
        include_failed=retry_failed,
        include_stale=rebuild_stale,
    )
    if limit is not None:
        todo = todo[:limit]

    report = GenerationReport(requested=len(todo))
    report.skipped_existing = int((frame["status"] == "done").sum())
    stale = int((frame["status"] == "stale").sum())
    log(
        f"[{dataset_id}] {len(frame)} pairs: {report.skipped_existing} done, "
        f"{len(todo)} to run (shard {shard[0]}/{shard[1]})"
        + (f", {stale} stale (use --rebuild-stale)" if stale and not rebuild_stale else "")
    )
    if dry_run or not todo:
        return report

    runner = runner or DcngenRunner()
    static_seed, _ = derive_seeds(dcngen_master_seed(ws), int(spec["n_seeds"]))
    static = None
    by_seed: dict[int, list[str]] = {}
    for pair_id in todo:
        by_seed.setdefault(pairs[pair_id].seed_index, []).append(pair_id)

    for seed_index, ids in by_seed.items():
        scenario_seed = pairs[ids[0]].scenario_seed
        if static is None:
            static = runner.static_context(spec, static_seed, ws)
        seed_path, cached, ghash = _seed_target(ws, spec, seed_index, scenario_seed)
        explicit = all(pairs[i].target is not None for i in ids)
        if explicit:  # no auto-target needed, so skip the reference run
            cached = pairs[ids[0]].target
        sc = runner.seed_context(static, scenario_seed, cached)
        if cached is None:
            write_json_atomic(
                seed_path,
                {
                    "scenario_seed": scenario_seed,
                    "target_consumer": sc.target,
                    "global_hash": ghash,
                },
            )
        log(f"seed {seed_index:02d}: {len(ids)} pair(s)")

        for pair_id in ids:
            pair = pairs[pair_id]
            started = time.monotonic()
            pair_sc = sc
            if pair.target is not None:
                pair_sc = copy.copy(sc)
                pair_sc.target = pair.target
            try:
                data, extra = runner.run_pair(static, pair_sc, pair)
                if _commit_pair(ws, spec, pair, data, extra, time.monotonic() - started, log):
                    report.completed.append(pair_id)
                    log(f"  done   {pair_id} ({time.monotonic() - started:.1f}s)")
            except Exception as exc:  # keep going; the pair is retried on the next run
                report.failed[pair_id] = f"{type(exc).__name__}: {exc}"
                write_json_atomic(
                    failures_dir(ws, dataset_id) / f"{pair_id}.json",
                    {
                        "pair_id": pair_id,
                        "error": report.failed[pair_id],
                        "traceback": traceback.format_exc(),
                        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    },
                )
                log(f"  FAILED {pair_id}: {report.failed[pair_id]}")

    write_manifest(spec, ws)
    return report


def _kind(filename: str) -> str:
    """data.parquet -> dataset; healthy_consumers.parquet -> consumers."""
    stem = filename.removesuffix(".parquet")
    return "dataset" if stem == "data" else stem.split("_", 1)[1]


def consolidate(
    spec: dict[str, Any], ws: Workspace | None = None, *, include_qa_failed: bool = False
) -> dict[str, Path]:
    """Concatenate committed pairs into ``consolidated/`` (dataset, manifest, summary)."""
    ws = ws or get_workspace()
    dataset_id = spec["dataset_id"]
    frame = write_manifest(spec, ws)
    wanted = {"done", "qa_failed"} if include_qa_failed else {"done"}
    chosen = frame[frame["status"].isin(wanted)]
    if chosen.empty:
        raise RuntimeError("no completed pairs to consolidate")

    base = pairs_dir(ws, dataset_id)
    kinds: dict[str, list[pd.DataFrame]] = {}
    for pid in chosen["pair_id"]:
        for path in sorted((base / pid).glob("*.parquet")):
            kinds.setdefault(_kind(path.name), []).append(pd.read_parquet(path))
    tables = {k: pd.concat(v, ignore_index=True) for k, v in kinds.items()}
    dataset = tables.get("dataset", next(iter(tables.values())))
    summary = (
        dataset.groupby(["fsed", "state"], dropna=False)
        .agg(rows=("time_s", "size"), scenarios=("scenario_id", "nunique"), seeds=("seed", "nunique"))
        .reset_index()
    )

    out = ws.dataset_dir(dataset_id) / "consolidated"
    out.mkdir(parents=True, exist_ok=True)
    paths = {name: out / f"{name}.parquet" for name in tables}
    paths["manifest"] = out / "scenario_manifest.csv"
    paths["summary"] = out / "dataset_summary.csv"
    for name, table in tables.items():
        with atomic_path(paths[name]) as tmp:
            table.to_parquet(tmp, index=False)
    with atomic_path(paths["manifest"]) as tmp:
        chosen.to_csv(tmp, index=False)
    with atomic_path(paths["summary"]) as tmp:
        summary.to_csv(tmp, index=False)
    write_json_atomic(
        out / "export.json",
        {
            "pairs": chosen["pair_id"].tolist(),
            "files": {name: sha256_file(paths[name]) for name in tables},
            "rows": {name: len(table) for name, table in tables.items()},
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    return paths
