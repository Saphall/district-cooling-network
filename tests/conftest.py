import pytest

from district_cooling_network.paths import reset_workspace_cache


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """Workspace in a temp dir; project root stays the real checkout (real configs)."""
    monkeypatch.setenv("SWIM_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.delenv("SWIM_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("SWIM_DATA_DIR", raising=False)
    monkeypatch.delenv("SWIM_DCNGEN_DIR", raising=False)
    reset_workspace_cache()
    from district_cooling_network.paths import get_workspace

    yield get_workspace()
    reset_workspace_cache()
