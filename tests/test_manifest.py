import copy

import pytest

from district_cooling_network.config import load_dataset_spec
from district_cooling_network.manifest import (
    assert_writable,
    build_pair_specs,
    derive_seeds,
    import_legacy,
    verify_legacy,
)


def test_pilot_grid_matches_legacy_pilot(ws):
    spec = load_dataset_spec("pilot_v1", ws)
    pairs = build_pair_specs(spec, ws)
    assert len(pairs) == 45  # 3 seeds x 5 FSEDs x 3 levels
    ids = [p.pair_id for p in pairs]
    assert len(set(ids)) == 45
    assert ids[0] == "seed00_stuck_01"
    assert "seed00_balancing_0.90" in ids
    assert "seed01_control_1.20" in ids
    assert "seed02_filter_K20" in ids
    assert "seed02_fouling_0.50" in ids


def test_extending_seeds_keeps_existing_pairs_unchanged(ws):
    spec = load_dataset_spec("pilot_v1", ws)
    bigger = copy.deepcopy(spec)
    bigger["n_seeds"] = 5
    old = {p.pair_id: p.spec_hash for p in build_pair_specs(spec, ws)}
    new = {p.pair_id: p.spec_hash for p in build_pair_specs(bigger, ws)}
    assert len(new) == 75
    assert all(new[k] == v for k, v in old.items())


def test_appending_a_level_keeps_existing_pairs_unchanged(ws):
    spec = load_dataset_spec("pilot_v1", ws)
    bigger = copy.deepcopy(spec)
    bigger["fsed"]["hx_fouling"]["levels"].append(0.30)
    old = {p.pair_id: p.spec_hash for p in build_pair_specs(spec, ws)}
    new = {p.pair_id: p.spec_hash for p in build_pair_specs(bigger, ws)}
    assert len(new) == 48
    assert all(new[k] == v for k, v in old.items())


def test_changing_a_global_parameter_invalidates_hashes(ws):
    spec = load_dataset_spec("pilot_v1", ws)
    changed = copy.deepcopy(spec)
    changed["healthy_ua_factor"] = 1.08
    old = {p.pair_id: p.spec_hash for p in build_pair_specs(spec, ws)}
    new = {p.pair_id: p.spec_hash for p in build_pair_specs(changed, ws)}
    assert old.keys() == new.keys()
    assert all(old[k] != new[k] for k in old)


def test_seed_stream_is_prefix_stable():
    _, three = derive_seeds(42, 3)
    _, six = derive_seeds(42, 6)
    assert six[:3] == three


def test_legacy_registry_detects_modification_and_blocks_writes(ws, tmp_path):
    legacy = tmp_path / "old_outputs"
    (legacy / "sub").mkdir(parents=True)
    (legacy / "a.csv").write_text("x,y\n1,2\n")
    (legacy / "sub" / "b, odd name.csv").write_text("z\n")

    assert import_legacy(ws, legacy, "scenarios105") == (2, 0)
    assert import_legacy(ws, legacy, "scenarios105") == (0, 2)  # idempotent
    assert len(verify_legacy(ws)["ok"]) == 2

    with pytest.raises(PermissionError):
        assert_writable(ws, legacy / "new.csv")
    assert_writable(ws, ws.dataset_dir("pilot_v1") / "x")

    (legacy / "a.csv").write_text("tampered")
    (legacy / "sub" / "b, odd name.csv").unlink()
    result = verify_legacy(ws)
    assert len(result["changed"]) == 1 and len(result["missing"]) == 1
