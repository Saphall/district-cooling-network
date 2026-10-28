"""Validation experiments (simulator physics checks). Run via ``dcn validate``."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from district_cooling_network.config import dcngen_config_fingerprint, load_dcngen_lock
from district_cooling_network.fsutil import (
    read_json,
    sha256_file,
    sha256_text,
    write_json_atomic,
)
from district_cooling_network.paths import Workspace, get_workspace

# name -> (module, output directory under results/validation)
EXPERIMENTS: dict[str, tuple[str, str]] = {
    "01_stuck_valve": ("district_cooling_network.validation.exp01_stuck_valve", "01_stuck_valve"),
    "02_excess_control_margin": (
        "district_cooling_network.validation.exp02_excess_control_margin",
        "02_excess_control_margin",
    ),
    "03_poor_balancing": ("district_cooling_network.validation.exp03_poor_balancing", "03_poor_balancing"),
    "04_clogged_filter": ("district_cooling_network.validation.exp04_clogged_filter", "04_clogged_filter"),
    "05_hx_fouling": ("district_cooling_network.validation.exp05_hx_fouling", "05_hx_fouling"),
}
# Cheap post-processing of the five experiments; always re-run.
DERIVED: dict[str, tuple[str, str]] = {
    "summary": ("district_cooling_network.validation.summary", "validation_summary"),
    "figures": ("district_cooling_network.validation.figures", "validation_figures"),
}
ALL_NAMES = [*EXPERIMENTS, *DERIVED]


@dataclass
class ValidationOutcome:
    name: str
    status: str  # passed | failed | skipped
    report: Path


def _module_file(module: str) -> Path:
    return Path(__file__).resolve().parents[2] / (module.replace(".", "/") + ".py")


def _fingerprint(ws: Workspace, module: str) -> str:
    lock = load_dcngen_lock(ws)
    parts = [
        sha256_file(_module_file(module)),
        lock.commit,
        *(p.sha256 for p in lock.patches),
        dcngen_config_fingerprint(ws),
    ]
    return sha256_text("|".join(parts))


def run_validation(name: str, ws: Workspace | None = None, force: bool = False) -> ValidationOutcome:
    ws = ws or get_workspace()
    table = {**EXPERIMENTS, **DERIVED}
    if name not in table:
        raise KeyError(f"unknown validation {name!r}; choose from {', '.join(ALL_NAMES)}")
    module, out_name = table[name]
    report_dir = ws.reports_dir / "validation"
    report_path = report_dir / f"{name}.json"
    fingerprint = _fingerprint(ws, module)

    if name in EXPERIMENTS and not force and report_path.exists():
        previous = read_json(report_path)
        if previous.get("status") == "passed" and previous.get("fingerprint") == fingerprint:
            return ValidationOutcome(name, "skipped", report_path)

    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / f"{name}.log"
    started = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(
            [sys.executable, "-m", module], stdout=log, stderr=subprocess.STDOUT, text=True
        )
    out_dir = ws.validation_root / out_name
    outputs = (
        {p.relative_to(ws.root).as_posix(): sha256_file(p) for p in sorted(out_dir.rglob("*")) if p.is_file()}
        if out_dir.exists()
        else {}
    )
    lock = load_dcngen_lock(ws)
    status = "passed" if proc.returncode == 0 else "failed"
    write_json_atomic(
        report_path,
        {
            "name": name,
            "status": status,
            "returncode": proc.returncode,
            "fingerprint": fingerprint,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
            "duration_s": round(time.time() - started, 2),
            "dcngen_commit": lock.commit,
            "dcngen_patches": {p.file: p.sha256 for p in lock.patches},
            "python": sys.version.split()[0],
            "log": log_path.relative_to(ws.root).as_posix(),
            "outputs": outputs,
        },
    )
    return ValidationOutcome(name, status, report_path)


def load_reports(ws: Workspace | None = None) -> list[dict]:
    ws = ws or get_workspace()
    report_dir = ws.reports_dir / "validation"
    return [json.loads(p.read_text()) for p in sorted(report_dir.glob("*.json"))]
