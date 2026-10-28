"""Config loading and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from district_cooling_network.fsutil import sha256_text, write_text_atomic
from district_cooling_network.paths import Workspace, get_workspace


@dataclass(frozen=True)
class PatchRef:
    file: str
    sha256: str


@dataclass(frozen=True)
class DcngenLock:
    repo: str
    commit: str
    patches: tuple[PatchRef, ...]


def load_dcngen_lock(ws: Workspace | None = None) -> DcngenLock:
    ws = ws or get_workspace()
    raw = yaml.safe_load(ws.dcngen_lock.read_text())
    return DcngenLock(
        repo=raw["repo"],
        commit=raw["commit"],
        patches=tuple(PatchRef(p["file"], p["sha256"]) for p in raw.get("patches", [])),
    )


def render_dcngen_config(ws: Workspace | None = None) -> Path:
    """Write ``configs/resolved/dcngen.yaml``: the pinned base config with this
    machine's DiTEC data path. Only ``paths.ditec_data`` differs from the base."""
    ws = ws or get_workspace()
    raw = yaml.safe_load(ws.dcngen_base_config.read_text())
    raw["paths"]["ditec_data"] = str(ws.data_dir)
    text = yaml.safe_dump(raw, sort_keys=False)
    target = ws.resolved_dcngen_config
    if not target.exists() or target.read_text() != text:
        write_text_atomic(target, text)
    return target


def dcngen_config_fingerprint(ws: Workspace | None = None) -> str:
    """Hash of the dcngen config *without* machine-specific paths."""
    ws = ws or get_workspace()
    raw = yaml.safe_load(ws.dcngen_base_config.read_text())
    raw.pop("paths", None)
    return sha256_text(yaml.safe_dump(raw, sort_keys=True))


def dcngen_master_seed(ws: Workspace | None = None) -> int:
    ws = ws or get_workspace()
    return int(yaml.safe_load(ws.dcngen_base_config.read_text())["seed"])


def load_dataset_spec(name_or_path: str, ws: Workspace | None = None) -> dict[str, Any]:
    """Load ``configs/datasets/<name>.yaml`` (or an explicit path)."""
    ws = ws or get_workspace()
    path = Path(name_or_path)
    if not path.suffix:
        path = ws.configs_dir / "datasets" / f"{name_or_path}.yaml"
    spec = yaml.safe_load(path.read_text())
    spec["_path"] = str(path)
    return spec
