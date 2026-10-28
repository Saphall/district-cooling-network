import pandas as pd
import pytest

from district_cooling_network.ml.splits import split_by_group
from district_cooling_network.ml.tracking import start_run


def test_seed_split_keeps_groups_disjoint():
    frame = pd.DataFrame({"seed": [s for s in range(10) for _ in range(3)], "x": range(30)})
    parts = split_by_group(frame, "seed", rng_seed=1)
    seeds = {k: set(v["seed"]) for k, v in parts.items()}
    assert seeds["train"].isdisjoint(seeds["val"]) and seeds["train"].isdisjoint(seeds["test"])
    assert seeds["val"].isdisjoint(seeds["test"])
    assert sum(len(v) for v in parts.values()) == 30
    assert all(len(v) for v in parts.values())


def test_too_few_groups_is_an_error():
    with pytest.raises(ValueError):
        split_by_group(pd.DataFrame({"seed": [0, 0, 1, 1]}), "seed")


def test_run_directory_is_keyed_by_config(ws):
    cfg = {"model": "x", "seed": 1}
    run = start_run("baseline", cfg, ws)
    assert not run.is_complete
    run.write_metrics({"f1": 0.5})
    run.finish()

    again = start_run("baseline", dict(reversed(list(cfg.items()))), ws)
    assert again.path == run.path and again.is_complete
    assert start_run("baseline", {**cfg, "seed": 2}, ws).path != run.path
