"""Resumability of dataset generation, using a fake simulation runner (no dcngen needed)."""

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from district_cooling_network.config import load_dataset_spec
from district_cooling_network.dataset.generate import consolidate, generate
from district_cooling_network.manifest import pairs_dir, select_pending, write_manifest


class FakeRunner:
    def __init__(self, fail: set[str] | None = None):
        self.calls: list[str] = []
        self.static_builds = 0
        self.reference_runs = 0
        self.fail = fail or set()

    def static_context(self, spec, static_seed, ws):
        self.static_builds += 1
        return SimpleNamespace(n_steps=int(spec["n_steps"]))

    def seed_context(self, static, scenario_seed, cached_target):
        if cached_target is None:
            self.reference_runs += 1
        return SimpleNamespace(scenario_seed=scenario_seed, target=cached_target or "J7")

    def run_pair(self, static, sc, pair):
        self.calls.append(pair.pair_id)
        if pair.pair_id in self.fail:
            raise RuntimeError("solver blew up")
        n = static.n_steps
        rows = []
        for state in ("healthy", "fault"):
            rows.append(
                pd.DataFrame(
                    {
                        "scenario_id": f"{pair.pair_id}_{state}",
                        "pair_id": pair.pair_id,
                        "seed": sc.scenario_seed,
                        "state": state,
                        "fsed": pair.fsed,
                        "target_consumer": sc.target,
                        "time_s": np.arange(n) * 600.0,
                        "converged": True,
                        "unmet": False,
                    }
                )
            )
        return {"data.parquet": pd.concat(rows, ignore_index=True)}, {}


@pytest.fixture
def spec(ws):
    s = load_dataset_spec("pilot_v1", ws)
    s["n_steps"] = 4
    return s


def quiet(_):
    pass


def test_second_run_regenerates_nothing(ws, spec):
    first = FakeRunner()
    report = generate(spec, ws, runner=first, log=quiet)
    assert len(report.completed) == 45 and report.ok
    assert first.static_builds == 1 and first.reference_runs == 3

    second = FakeRunner()
    report = generate(spec, ws, runner=second, log=quiet)
    assert second.calls == [] and second.static_builds == 0 and report.completed == []


def test_resume_after_interruption_only_runs_missing_pairs(ws, spec):
    generate(spec, ws, runner=FakeRunner(), limit=10, log=quiet)
    frame = write_manifest(spec, ws)
    assert (frame["status"] == "done").sum() == 10

    resumed = FakeRunner()
    generate(spec, ws, runner=resumed, log=quiet)
    assert len(resumed.calls) == 35
    done_before = set(frame.loc[frame["status"] == "done", "pair_id"])
    assert done_before.isdisjoint(resumed.calls)
    # seed targets were cached, so only the seeds not yet seen need a reference run
    assert resumed.reference_runs <= 2


def test_failed_pair_is_recorded_and_retried(ws, spec):
    bad = "seed00_balancing_0.80"
    report = generate(spec, ws, runner=FakeRunner(fail={bad}), log=quiet)
    assert list(report.failed) == [bad] and not report.ok
    assert write_manifest(spec, ws).set_index("pair_id").loc[bad, "status"] == "failed"

    retry = FakeRunner()
    report = generate(spec, ws, runner=retry, log=quiet)
    assert retry.calls == [bad] and report.ok


def test_extending_dataset_only_adds_new_pairs(ws, spec):
    generate(spec, ws, runner=FakeRunner(), log=quiet)
    bigger = copy.deepcopy(spec)
    bigger["n_seeds"] = 4
    extend = FakeRunner()
    generate(bigger, ws, runner=extend, log=quiet)
    assert len(extend.calls) == 15
    assert all(c.startswith("seed03_") for c in extend.calls)


def test_changed_spec_marks_pairs_stale_and_never_overwrites_silently(ws, spec):
    generate(spec, ws, runner=FakeRunner(), log=quiet)
    changed = copy.deepcopy(spec)
    changed["healthy_ua_factor"] = 1.08

    untouched = FakeRunner()
    generate(changed, ws, runner=untouched, log=quiet)
    assert untouched.calls == []
    assert (write_manifest(changed, ws)["status"] == "stale").all()

    rebuild = FakeRunner()
    generate(changed, ws, runner=rebuild, rebuild_stale=True, limit=2, log=quiet)
    assert len(rebuild.calls) == 2
    archived = list((ws.dataset_dir("pilot_v1") / "pairs_stale").iterdir())
    assert len(archived) == 2


def test_shards_partition_the_work(ws, spec):
    frame = write_manifest(spec, ws)
    shards = [set(select_pending(frame, shard=(i, 3))) for i in range(3)]
    assert sum(len(s) for s in shards) == 45
    assert not (shards[0] & shards[1] or shards[1] & shards[2] or shards[0] & shards[2])


def test_crash_while_writing_leaves_no_partial_pair(ws, spec, monkeypatch):
    def boom(self, *a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    report = generate(spec, ws, runner=FakeRunner(), limit=1, log=quiet)
    assert len(report.failed) == 1

    root = pairs_dir(ws, "pilot_v1")
    assert list(root.iterdir()) == []  # neither a pair dir nor a leftover tmp dir
    assert (write_manifest(spec, ws)["status"] == "failed").sum() == 1


def test_consolidate_and_qa_exclusion(ws, spec):
    generate(spec, ws, runner=FakeRunner(), limit=5, log=quiet)
    paths = consolidate(spec, ws)
    data = pd.read_parquet(paths["dataset"])
    assert data["pair_id"].nunique() == 5 and set(data["state"]) == {"healthy", "fault"}
    assert paths["summary"].exists() and paths["manifest"].exists()


def test_pair_directory_has_success_marker_and_file_hashes(ws, spec):
    generate(spec, ws, runner=FakeRunner(), limit=1, log=quiet)
    pair_dir = next(p for p in pairs_dir(ws, "pilot_v1").iterdir())
    assert (pair_dir / "_SUCCESS").exists() and (pair_dir / "data.parquet").exists()
    from district_cooling_network.fsutil import read_json

    assert list(read_json(pair_dir / "meta.json")["files"]) == ["data.parquet"]


def test_directory_without_success_marker_is_archived_and_redone(ws, spec):
    generate(spec, ws, runner=FakeRunner(), limit=1, log=quiet)
    pair_dir = next(p for p in pairs_dir(ws, "pilot_v1").iterdir())
    (pair_dir / "_SUCCESS").unlink()
    assert write_manifest(spec, ws).set_index("pair_id").loc[pair_dir.name, "status"] == "failed"

    redo = FakeRunner()
    generate(spec, ws, runner=redo, log=quiet, limit=1)
    assert redo.calls == [pair_dir.name] or pair_dir.name in redo.calls
    assert (pair_dir / "_SUCCESS").exists()
    assert len(list((ws.dataset_dir("pilot_v1") / "pairs_stale").iterdir())) == 1


def test_explicit_targets_expand_the_grid_and_skip_reference_runs(ws, spec):
    spec = copy.deepcopy(spec)
    spec["targets"] = ["25", "7"]
    spec["n_seeds"] = 1
    runner = FakeRunner()
    report = generate(spec, ws, runner=runner, log=quiet)
    assert len(report.completed) == 30 and runner.reference_runs == 0
    assert "seed00_target25_stuck_01" in report.completed
    data = pd.read_parquet(pairs_dir(ws, "pilot_v1") / "seed00_target7_stuck_01" / "data.parquet")
    assert set(data["target_consumer"]) == {"7"}
