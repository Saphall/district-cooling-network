"""Pristine dcngen -> derived (patched) dcngen build, and import bootstrap.

The pinned upstream clone is only ever *read* (``git archive`` of the pinned
commit). SWiM-specific changes live in ``configs/patches/*.patch`` and are applied
to the derived copy ``external/dcngen_swim`` only.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import yaml

from district_cooling_network.config import DcngenLock, load_dcngen_lock
from district_cooling_network.fsutil import sha256_file, write_json_atomic
from district_cooling_network.paths import Workspace, get_workspace

BUILD_MARKER = ".swim_build.json"


class DcngenSetupError(RuntimeError):
    pass


def _git(*args: str, cwd: Path | None = None) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return out.stdout.strip()


def ensure_pristine(ws: Workspace, lock: DcngenLock) -> str:
    """Clone the pinned commit if absent; otherwise only verify it. Returns notes."""
    repo = ws.dcngen_pristine
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", lock.repo, str(repo)], check=True)
        _git("checkout", "--quiet", "--detach", lock.commit, cwd=repo)
        return f"cloned {lock.repo} at {lock.commit[:7]}"

    try:
        _git("cat-file", "-e", f"{lock.commit}^{{commit}}", cwd=repo)
    except subprocess.CalledProcessError as exc:
        raise DcngenSetupError(
            f"{repo} does not contain pinned commit {lock.commit}. "
            "It is left untouched; point SWIM_DCNGEN_DIR at a clone that has it."
        ) from exc

    head = _git("rev-parse", "HEAD", cwd=repo)
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no", cwd=repo))
    notes = [f"using existing clone (HEAD {head[:7]})"]
    if head != lock.commit:
        notes.append(f"HEAD differs from pin; building from pinned commit {lock.commit[:7]}")
    if dirty:
        notes.append("working tree has local edits; they are ignored (read from the commit)")
    return "; ".join(notes)


def _read_patches(ws: Workspace, lock: DcngenLock) -> list[Path]:
    paths = []
    for ref in lock.patches:
        path = ws.patches_dir / ref.file
        if not path.exists():
            raise DcngenSetupError(f"patch missing: {path}")
        actual = sha256_file(path)
        if actual != ref.sha256:
            raise DcngenSetupError(
                f"patch {ref.file} sha256 {actual} != locked {ref.sha256}; "
                "update configs/dcngen.lock.yaml if the change is intentional"
            )
        paths.append(path)
    return paths


def build_fingerprint(lock: DcngenLock) -> dict:
    return {"commit": lock.commit, "patches": {p.file: p.sha256 for p in lock.patches}}


def is_built(ws: Workspace, lock: DcngenLock) -> bool:
    marker = ws.dcngen_derived / BUILD_MARKER
    if not marker.exists():
        return False
    return json.loads(marker.read_text()) == build_fingerprint(lock)


def build_derived(ws: Workspace | None = None, force: bool = False) -> str:
    ws = ws or get_workspace()
    lock = load_dcngen_lock(ws)
    notes = ensure_pristine(ws, lock)
    patches = _read_patches(ws, lock)

    derived = ws.dcngen_derived
    if is_built(ws, lock) and not force:
        return f"{notes}; derived build up to date"

    if derived.exists():
        if not (derived / BUILD_MARKER).exists():
            raise DcngenSetupError(f"{derived} exists but is not a dcn build; refusing to delete")
        shutil.rmtree(derived)
    derived.mkdir(parents=True)

    archive = subprocess.run(
        ["git", "archive", "--format=tar", lock.commit],
        cwd=ws.dcngen_pristine,
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(derived, filter="data")

    # Inside an enclosing git repo, `git apply` silently skips paths outside the cwd.
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(derived.parent)}
    for patch in patches:
        subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(patch)],
            cwd=derived,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        applied = subprocess.run(
            ["git", "apply", "--reverse", "--check", str(patch)],
            cwd=derived,
            capture_output=True,
            text=True,
            env=env,
        )
        if applied.returncode != 0:
            shutil.rmtree(derived)
            raise DcngenSetupError(f"patch {patch.name} did not take effect: {applied.stderr.strip()}")
    write_json_atomic(derived / BUILD_MARKER, build_fingerprint(lock))
    return f"{notes}; built {derived} with {len(patches)} patch(es)"


def extract_patch(ws: Workspace, modified_dir: Path, name: str) -> tuple[Path, list[str]]:
    """Capture the edits of an in-place modified dcngen clone as the project patch.

    Read-only on ``modified_dir`` (``git diff`` against the pinned commit). The
    lock file is rewritten to reference exactly this one patch. Returns the patch
    path and warnings (e.g. untracked source files that a diff cannot carry).
    """
    lock = load_dcngen_lock(ws)
    diff = subprocess.run(
        ["git", "diff", "--binary", lock.commit, "--", "src"],
        cwd=modified_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if not diff.strip():
        raise DcngenSetupError(f"{modified_dir} has no changes under src/ relative to {lock.commit[:7]}")

    untracked = _git("ls-files", "--others", "--exclude-standard", "--", "src", cwd=modified_dir)
    warnings = [
        f"untracked file not captured: {f}"
        for f in untracked.splitlines()
        if ".before_" not in f and not f.endswith((".pyc", ".bak"))
    ]

    ws.patches_dir.mkdir(parents=True, exist_ok=True)
    patch_path = ws.patches_dir / name
    patch_path.write_text(diff)
    digest = sha256_file(patch_path)
    ws.dcngen_lock.write_text(
        "# Pinned upstream dcngen. The upstream clone is never modified; the project\n"
        "# builds a derived copy (external/dcngen_swim) = pristine commit + configs/patches/*.patch.\n"
        + yaml.safe_dump(
            {
                "repo": lock.repo,
                "commit": lock.commit,
                "patches": [{"file": name, "sha256": digest}],
            },
            sort_keys=False,
        )
    )
    return patch_path, warnings


def ensure_dcngen_importable(ws: Workspace | None = None) -> Path:
    """Put the derived dcngen ``src`` first on ``sys.path``; build it if missing."""
    ws = ws or get_workspace()
    lock = load_dcngen_lock(ws)
    if not is_built(ws, lock):
        build_derived(ws)
    src = ws.dcngen_derived / "src"
    loaded = sys.modules.get("dcngen")
    if loaded is not None:
        loaded_from = Path(getattr(loaded, "__file__", "") or "").resolve()
        if src.resolve() not in loaded_from.parents:
            raise DcngenSetupError(
                f"dcngen already imported from {loaded_from}, not the SWiM build {src}; "
                "uninstall other dcngen copies (pip uninstall dcngen)."
            )
        return src
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src
