"""Pristine -> derived dcngen build, using a tiny throwaway git repo."""

import hashlib
import subprocess
import sys

import pytest
import yaml

from district_cooling_network import dcngen_env
from district_cooling_network.paths import get_workspace, reset_workspace_cache


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def fake_project(tmp_path, monkeypatch):
    # Upstream repo with one commit.
    upstream = tmp_path / "upstream"
    (upstream / "src" / "dcngen").mkdir(parents=True)
    (upstream / "src" / "dcngen" / "__init__.py").write_text("VALUE = 1\n")
    git("init", "-q", cwd=upstream)
    git("-c", "user.email=a@b", "-c", "user.name=t", "add", ".", cwd=upstream)
    git("-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "init", cwd=upstream)
    commit = git("rev-parse", "HEAD", cwd=upstream)

    # Patch produced from a scratch edit.
    (upstream / "src" / "dcngen" / "__init__.py").write_text("VALUE = 2\n")
    patch = git("diff", cwd=upstream) + "\n"
    git("checkout", "--", ".", cwd=upstream)

    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='x'\n")
    (project / "configs" / "patches").mkdir(parents=True)
    (project / "configs" / "patches" / "0001.patch").write_text(patch)
    (project / "configs" / "dcngen.lock.yaml").write_text(
        yaml.safe_dump(
            {
                "repo": str(upstream),
                "commit": commit,
                "patches": [
                    {"file": "0001.patch", "sha256": hashlib.sha256(patch.encode()).hexdigest()}
                ],
            }
        )
    )

    monkeypatch.setenv("SWIM_PROJECT_ROOT", str(project))
    monkeypatch.setenv("SWIM_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("SWIM_DCNGEN_DIR", str(upstream))
    reset_workspace_cache()
    yield upstream
    reset_workspace_cache()
    sys.modules.pop("dcngen", None)


def test_derived_build_patches_copy_and_leaves_pristine_untouched(fake_project):
    ws = get_workspace()
    before = (fake_project / "src" / "dcngen" / "__init__.py").read_text()

    note = dcngen_env.build_derived(ws)
    assert "built" in note
    assert (ws.dcngen_derived / "src" / "dcngen" / "__init__.py").read_text() == "VALUE = 2\n"
    assert (fake_project / "src" / "dcngen" / "__init__.py").read_text() == before == "VALUE = 1\n"
    assert git("status", "--porcelain", cwd=fake_project) == ""

    assert "up to date" in dcngen_env.build_derived(ws)  # idempotent


def test_patch_applies_when_workspace_is_inside_a_git_repo(fake_project):
    ws = get_workspace()
    ws.root.mkdir(parents=True)
    git("init", "-q", cwd=ws.root)

    dcngen_env.build_derived(ws)
    assert (ws.dcngen_derived / "src" / "dcngen" / "__init__.py").read_text() == "VALUE = 2\n"


def test_local_edits_in_existing_clone_are_ignored_not_reverted(fake_project):
    ws = get_workspace()
    edited = fake_project / "src" / "dcngen" / "__init__.py"
    edited.write_text("VALUE = 99  # someone's in-place edit\n")

    dcngen_env.build_derived(ws)
    # Built from the pinned commit, so the edit neither leaks in nor gets reverted.
    assert (ws.dcngen_derived / "src" / "dcngen" / "__init__.py").read_text() == "VALUE = 2\n"
    assert edited.read_text().startswith("VALUE = 99")


def test_extract_patch_from_modified_clone_is_read_only_and_rebuildable(fake_project):
    ws = get_workspace()
    edited = fake_project / "src" / "dcngen" / "__init__.py"
    edited.write_text("VALUE = 7\n")
    (fake_project / "src" / "dcngen" / "x.py.before_edit").write_text("backup")

    path, warnings = dcngen_env.extract_patch(ws, fake_project, "0002-from-drive.patch")
    assert warnings == []
    assert "+VALUE = 7" in path.read_text()
    assert edited.read_text() == "VALUE = 7\n"  # source repo untouched

    dcngen_env.build_derived(ws, force=True)
    assert (ws.dcngen_derived / "src" / "dcngen" / "__init__.py").read_text() == "VALUE = 7\n"


def test_modified_patch_is_rejected(fake_project):
    ws = get_workspace()
    (ws.patches_dir / "0001.patch").write_text("tampered\n")
    with pytest.raises(dcngen_env.DcngenSetupError, match="sha256"):
        dcngen_env.build_derived(ws)


def test_refuses_to_delete_foreign_directory(fake_project):
    ws = get_workspace()
    ws.dcngen_derived.mkdir(parents=True)
    (ws.dcngen_derived / "precious.txt").write_text("keep")
    with pytest.raises(dcngen_env.DcngenSetupError, match="refusing"):
        dcngen_env.build_derived(ws)
    assert (ws.dcngen_derived / "precious.txt").exists()
