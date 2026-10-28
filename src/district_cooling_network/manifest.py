"""Scenario manifest: the dataset grid, per-pair status, and the legacy registry.

Completion is decided by the filesystem, not by a mutable table: a counterfactual
pair is *done* when ``datasets/<id>/pairs/<pair_id>/_SUCCESS`` exists (written last,
and the pair directory itself appears atomically). ``manifest.csv`` is a
human-readable view regenerated from the spec plus that on-disk state, so
concurrent shards and crashed runs cannot corrupt it.
"""

from __future__ import annotations

import csv
import io
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from district_cooling_network.config import (
    dcngen_config_fingerprint,
    dcngen_master_seed,
    load_dcngen_lock,
)
from district_cooling_network.fsutil import (
    canonical_json,
    is_within,
    read_json,
    sha256_file,
    sha256_text,
    write_text_atomic,
)
from district_cooling_network.paths import Workspace, get_workspace

FSED_ORDER = (
    "stuck_actuator",
    "poor_balancing",
    "excess_control_margin",
    "clogged_filter",
    "hx_fouling",
)

STATUSES = ("pending", "done", "qa_failed", "failed", "stale")

MANIFEST_COLUMNS = [
    "pair_id",
    "seed_index",
    "scenario_seed",
    "fsed",
    "severity_parameter",
    "severity_value",
    "spec_hash",
    "status",
    "target_consumer",
    "rows",
    "completed_at",
]


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    seed_index: int
    scenario_seed: int
    fsed: str
    level_index: int
    severity_parameter: str
    severity_value: float
    spec_hash: str
    target: str | None = None  # None = auto-selected per seed


def pair_id_for(
    seed_index: int, fsed: str, level_index: int, level: float, target: str | None = None
) -> str:
    """Auto-target ids match the legacy pilot naming (Experiment 06)."""
    prefix = f"seed{seed_index:02d}" + (f"_target{target}" if target is not None else "")
    match fsed:
        case "stuck_actuator":
            return f"{prefix}_stuck_{level_index + 1:02d}"
        case "poor_balancing":
            return f"{prefix}_balancing_{level:.2f}"
        case "excess_control_margin":
            return f"{prefix}_control_{level:.2f}"
        case "clogged_filter":
            return f"{prefix}_filter_K{level:g}"
        case "hx_fouling":
            return f"{prefix}_fouling_{level:.2f}"
    raise ValueError(f"unknown FSED {fsed!r}")


def derive_seeds(master_seed: int, n_seeds: int) -> tuple[int, list[int]]:
    """Same draw order as the legacy script, so extending ``n_seeds`` keeps old seeds."""
    rng = np.random.default_rng(master_seed)
    static_seed = int(rng.integers(2**63))
    return static_seed, [int(rng.integers(2**63)) for _ in range(n_seeds)]


def global_params(spec: dict[str, Any], ws: Workspace) -> dict[str, Any]:
    """Everything that, if changed, invalidates already generated pairs."""
    lock = load_dcngen_lock(ws)
    return {
        "n_steps": spec["n_steps"],
        "healthy_ua_factor": spec["healthy_ua_factor"],
        "pump_curve": spec["pump_curve"],
        "filter": spec["filter"],
        "export": spec.get("export", "target_table"),
        "dcngen_commit": lock.commit,
        "dcngen_patches": {p.file: p.sha256 for p in lock.patches},
        "dcngen_config": dcngen_config_fingerprint(ws),
    }


def build_pair_specs(spec: dict[str, Any], ws: Workspace | None = None) -> list[PairSpec]:
    ws = ws or get_workspace()
    shared = global_params(spec, ws)
    _, seeds = derive_seeds(dcngen_master_seed(ws), int(spec["n_seeds"]))
    targets = spec.get("targets", "auto")
    target_list: list[str | None] = [None] if targets == "auto" else [str(t) for t in targets]
    pairs: list[PairSpec] = []
    for seed_index, scenario_seed in enumerate(seeds):
        for target in target_list:
            for fsed in FSED_ORDER:
                block = spec["fsed"].get(fsed)
                if not block:
                    continue
                for level_index, level in enumerate(block["levels"]):
                    level = float(level)
                    digest = sha256_text(
                        canonical_json(
                            {
                                "global": shared,
                                "seed_index": seed_index,
                                "scenario_seed": scenario_seed,
                                "target": target,
                                "fsed": fsed,
                                "parameter": block["severity_parameter"],
                                "level": level,
                            }
                        )
                    )
                    pairs.append(
                        PairSpec(
                            pair_id=pair_id_for(seed_index, fsed, level_index, level, target),
                            seed_index=seed_index,
                            scenario_seed=scenario_seed,
                            fsed=fsed,
                            level_index=level_index,
                            severity_parameter=block["severity_parameter"],
                            severity_value=level,
                            spec_hash=digest,
                            target=target,
                        )
                    )
    ids = [p.pair_id for p in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate pair ids in dataset spec (levels collide after formatting)")
    return pairs


# --- on-disk layout ------------------------------------------------------------------


def pairs_dir(ws: Workspace, dataset_id: str) -> Path:
    return ws.dataset_dir(dataset_id) / "pairs"


def failures_dir(ws: Workspace, dataset_id: str) -> Path:
    return ws.dataset_dir(dataset_id) / "failures"


def seeds_dir(ws: Workspace, dataset_id: str) -> Path:
    return ws.dataset_dir(dataset_id) / "seeds"


def pair_status(ws: Workspace, dataset_id: str, pair: PairSpec) -> str:
    pair_dir = pairs_dir(ws, dataset_id) / pair.pair_id
    meta_path = pair_dir / "meta.json"
    if pair_dir.exists() and not (pair_dir / "_SUCCESS").exists():
        return "failed"  # incomplete directory: archived and redone on the next run
    if meta_path.exists():
        meta = read_json(meta_path)
        if meta.get("spec_hash") != pair.spec_hash:
            return "stale"
        return "done" if meta.get("qa", {}).get("passed", False) else "qa_failed"
    if (failures_dir(ws, dataset_id) / f"{pair.pair_id}.json").exists():
        return "failed"
    return "pending"


def build_manifest(
    spec: dict[str, Any], ws: Workspace | None = None, pairs: list[PairSpec] | None = None
) -> pd.DataFrame:
    ws = ws or get_workspace()
    dataset_id = spec["dataset_id"]
    pairs = pairs if pairs is not None else build_pair_specs(spec, ws)
    rows = []
    for pair in pairs:
        status = pair_status(ws, dataset_id, pair)
        meta: dict[str, Any] = {}
        meta_path = pairs_dir(ws, dataset_id) / pair.pair_id / "meta.json"
        if meta_path.exists():
            meta = read_json(meta_path)
        rows.append(
            {
                **{k: v for k, v in asdict(pair).items() if k != "level_index"},
                "status": status,
                "target_consumer": meta.get("target_consumer", ""),
                "rows": meta.get("rows", ""),
                "completed_at": meta.get("completed_at", ""),
            }
        )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def write_manifest(spec: dict[str, Any], ws: Workspace | None = None) -> pd.DataFrame:
    """Refresh ``manifest.csv`` and keep a snapshot of the spec it was built from."""
    ws = ws or get_workspace()
    base = ws.dataset_dir(spec["dataset_id"])
    frame = build_manifest(spec, ws)
    write_text_atomic(base / "manifest.csv", frame.to_csv(index=False))

    clean = {k: v for k, v in spec.items() if not k.startswith("_")}
    text = yaml.safe_dump(clean, sort_keys=True)
    current = base / "spec.yaml"
    if current.exists() and current.read_text() != text:
        history = base / "spec_history"
        history.mkdir(exist_ok=True)
        write_text_atomic(history / f"{sha256_text(current.read_text())[:12]}.yaml", current.read_text())
    write_text_atomic(current, text)
    return frame


def select_pending(
    frame: pd.DataFrame,
    *,
    shard: tuple[int, int] = (0, 1),
    only_fsed: set[str] | None = None,
    include_failed: bool = True,
    include_stale: bool = False,
) -> list[str]:
    """Pair ids still to run for this shard (grid order, round-robin shards)."""
    index, count = shard
    if not 0 <= index < count:
        raise ValueError(f"invalid shard {index}/{count}")
    wanted = {"pending"}
    if include_failed:
        wanted.add("failed")
    if include_stale:
        wanted.add("stale")
    out = []
    for position, row in enumerate(frame.itertuples(index=False)):
        if position % count != index:
            continue
        if row.status in wanted and (only_fsed is None or row.fsed in only_fsed):
            out.append(row.pair_id)
    return out


# --- legacy outputs -------------------------------------------------------------------

REGISTRY_FIELDS = ["label", "root", "relpath", "size", "sha256", "registered_at"]


def _read_registry(ws: Workspace) -> list[dict[str, str]]:
    if not ws.legacy_registry.exists():
        return []
    with ws.legacy_registry.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _write_registry(ws: Workspace, rows: list[dict[str, Any]]) -> None:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=REGISTRY_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_text_atomic(ws.legacy_registry, buf.getvalue())


def legacy_roots(ws: Workspace) -> list[Path]:
    return sorted({Path(r["root"]) for r in _read_registry(ws)})


def import_legacy(ws: Workspace, root: Path, label: str, pattern: str = "*") -> tuple[int, int]:
    """Register existing outputs (path, size, sha256) without touching them.

    Returns ``(newly_registered, already_registered)``."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = _read_registry(ws)
    known = {(r["root"], r["relpath"]) for r in rows}
    new = skipped = 0
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for path in sorted(p for p in root.rglob(pattern) if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if (str(root), rel) in known:
            skipped += 1
            continue
        rows.append(
            {
                "label": label,
                "root": str(root),
                "relpath": rel,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "registered_at": stamp,
            }
        )
        new += 1
    _write_registry(ws, rows)
    return new, skipped


def verify_legacy(ws: Workspace) -> dict[str, list[str]]:
    """Compare registered files with disk: ``missing`` / ``changed`` / ``ok`` lists."""
    result: dict[str, list[str]] = {"ok": [], "missing": [], "changed": []}
    for row in _read_registry(ws):
        path = Path(row["root"]) / row["relpath"]
        if not path.exists():
            result["missing"].append(str(path))
        elif sha256_file(path) != row["sha256"]:
            result["changed"].append(str(path))
        else:
            result["ok"].append(str(path))
    return result


def assert_writable(ws: Workspace, path: Path) -> None:
    """Refuse writes into the pristine dcngen clone or any registered legacy output root."""
    protected = [ws.dcngen_pristine, *legacy_roots(ws)]
    for root in protected:
        if is_within(path, root):
            raise PermissionError(f"{path} is inside protected location {root}")
