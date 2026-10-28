"""Workspace layout shared by Google Colab and Linux servers.

Code and small configs live in the project checkout. Everything bulky or
generated lives in the *workspace*, a directory that can sit anywhere
(Google Drive on Colab, a data disk on a server). It is selected with
``SWIM_WORKSPACE`` and defaults to the project checkout itself.

Workspace layout::

    <workspace>/
      data/ditec-wdn/<network>/*.parquet   raw inputs (read-only)
      external/dcngen/                     pristine, pinned upstream repo (read-only)
      external/dcngen_swim/                derived build: pristine + patches/
      configs/resolved/dcngen.yaml         rendered config (machine paths filled in)
      datasets/<dataset_id>/               manifest, per-pair outputs, consolidated exports
      datasets/_legacy/registry.csv        integrity registry of pre-existing outputs
      results/validation/<experiment>/     validation CSVs and figures
      results/ml/<run_id>/                 ML run directories
      reports/                             validation / dataset status reports

Optional overrides: ``SWIM_DATA_DIR`` (the directory that *contains* the network
folders) and ``SWIM_DCNGEN_DIR`` (an existing dcngen clone to use as the pristine
source; it is only ever read).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ENV_WORKSPACE = "SWIM_WORKSPACE"
ENV_PROJECT_ROOT = "SWIM_PROJECT_ROOT"
ENV_DATA_DIR = "SWIM_DATA_DIR"
ENV_DCNGEN_DIR = "SWIM_DCNGEN_DIR"


def is_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def find_project_root() -> Path:
    """Directory holding ``pyproject.toml`` and ``configs/`` of this project."""
    env = os.environ.get(ENV_PROJECT_ROOT)
    if env:
        return Path(env).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").is_dir():
            return parent
    raise RuntimeError(
        f"Cannot locate the project root from {__file__}; set {ENV_PROJECT_ROOT}."
    )


@dataclass(frozen=True)
class Workspace:
    root: Path
    project_root: Path
    data_dir: Path
    dcngen_pristine: Path

    @property
    def dcngen_derived(self) -> Path:
        return self.root / "external" / "dcngen_swim"

    @property
    def resolved_dcngen_config(self) -> Path:
        return self.root / "configs" / "resolved" / "dcngen.yaml"

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def legacy_registry(self) -> Path:
        return self.datasets_dir / "_legacy" / "registry.csv"

    @property
    def validation_root(self) -> Path:
        return self.root / "results" / "validation"

    @property
    def ml_root(self) -> Path:
        return self.root / "results" / "ml"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    # Source-controlled inputs --------------------------------------------------

    @property
    def configs_dir(self) -> Path:
        return self.project_root / "configs"

    @property
    def patches_dir(self) -> Path:
        return self.project_root / "patches"

    @property
    def dcngen_lock(self) -> Path:
        return self.configs_dir / "dcngen.lock.yaml"

    @property
    def dcngen_base_config(self) -> Path:
        return self.configs_dir / "dcngen_default.yaml"

    # Helpers -------------------------------------------------------------------

    def dataset_dir(self, dataset_id: str) -> Path:
        return self.datasets_dir / dataset_id

    def validation_dir(self, experiment: str) -> Path:
        path = self.validation_root / experiment
        path.mkdir(parents=True, exist_ok=True)
        return path

    def dcngen_config(self) -> Path:
        """Resolved dcngen config; rendered on first use."""
        from swim_project.config import render_dcngen_config

        return render_dcngen_config(self)


@lru_cache(maxsize=1)
def get_workspace() -> Workspace:
    project_root = find_project_root()
    root = Path(os.environ.get(ENV_WORKSPACE, project_root)).expanduser().resolve()
    data_dir = Path(os.environ.get(ENV_DATA_DIR, root / "data" / "ditec-wdn")).expanduser()
    pristine = Path(os.environ.get(ENV_DCNGEN_DIR, root / "external" / "dcngen")).expanduser()
    return Workspace(
        root=root,
        project_root=project_root,
        data_dir=data_dir.resolve(),
        dcngen_pristine=pristine.resolve(),
    )


def reset_workspace_cache() -> None:
    """Re-read environment variables (tests, notebooks that change them)."""
    get_workspace.cache_clear()
