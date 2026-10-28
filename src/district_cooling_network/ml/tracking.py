"""ML run directories: ``results/ml/<name>/<config-hash>/``.

The directory is keyed by the config hash, so re-running an identical config
finds its finished run (``Run.is_complete``) instead of recomputing it.
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from district_cooling_network.fsutil import (
    canonical_json,
    read_json,
    sha256_text,
    write_json_atomic,
    write_text_atomic,
)
from district_cooling_network.paths import Workspace, get_workspace


@dataclass
class Run:
    path: Path
    config: dict[str, Any]

    @property
    def is_complete(self) -> bool:
        status = self.path / "status.json"
        return status.exists() and read_json(status).get("state") == "complete"

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        write_json_atomic(self.path / "metrics.json", metrics)

    def finish(self) -> None:
        write_json_atomic(
            self.path / "status.json",
            {"state": "complete", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
        )


def start_run(name: str, config: dict[str, Any], ws: Workspace | None = None) -> Run:
    """Create (or reopen) the run directory for ``config`` and snapshot it."""
    ws = ws or get_workspace()
    digest = sha256_text(canonical_json(config))[:12]
    path = ws.ml_root / name / digest
    path.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path / "config.yaml", yaml.safe_dump(config, sort_keys=True))
    write_json_atomic(
        path / "env.json",
        {"python": sys.version.split()[0], "platform": platform.platform()},
    )
    if not (path / "status.json").exists():
        write_json_atomic(
            path / "status.json",
            {"state": "started", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
        )
    return Run(path=path, config=config)
