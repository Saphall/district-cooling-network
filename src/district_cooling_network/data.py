"""Raw input data (DiTEC-WDN parquet files)."""

from __future__ import annotations

from pathlib import Path

import yaml

from district_cooling_network.paths import Workspace, get_workspace


def _inputs(ws: Workspace) -> dict:
    return yaml.safe_load((ws.configs_dir / "inputs.yaml").read_text())["ditec_wdn"]


def network_dir(ws: Workspace | None = None) -> Path:
    ws = ws or get_workspace()
    return ws.data_dir / _inputs(ws)["network"]


def missing_inputs(ws: Workspace | None = None) -> list[str]:
    ws = ws or get_workspace()
    folder = network_dir(ws)
    return [f for f in _inputs(ws)["files"] if not (folder / f).exists()]


def download_inputs(ws: Workspace | None = None, log=print) -> list[str]:
    """Download only the missing files straight into the data directory."""
    from huggingface_hub import hf_hub_download

    ws = ws or get_workspace()
    cfg = _inputs(ws)
    missing = missing_inputs(ws)
    for i, name in enumerate(missing, 1):
        log(f"[{i}/{len(missing)}] downloading {name}")
        hf_hub_download(
            repo_id=cfg["repo_id"],
            repo_type=cfg["repo_type"],
            filename=f"{cfg['network']}/{name}",
            local_dir=ws.data_dir,
        )
    return missing
