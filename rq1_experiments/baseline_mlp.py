"""RQ1 topology-agnostic baseline: MLP on target-ETS measurements, scenario-level split.

Reads ``datasets/<id>/consolidated/dataset.parquet`` (``export: target_table``) and writes
``results/ml/rq1_baseline_mlp/<config-hash>/``. Finished runs are skipped unless ``--force``.

    uv run --extra ml python rq1_experiments/baseline_mlp.py --dataset pilot_v1
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from district_cooling_network.ml.features import add_labels, feature_matrix
from district_cooling_network.ml.splits import split_by_group
from district_cooling_network.ml.tracking import start_run
from district_cooling_network.paths import get_workspace


def fit_eval(train: pd.DataFrame, test: pd.DataFrame, target: str, seed: int) -> dict:
    scaler = StandardScaler().fit(feature_matrix(train))
    model = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500, random_state=seed)
    model.fit(scaler.transform(feature_matrix(train)), train[target])
    pred = model.predict(scaler.transform(feature_matrix(test)))
    return {
        "accuracy": float(accuracy_score(test[target], pred)),
        "macro_f1": float(f1_score(test[target], pred, average="macro")),
        "n_train": len(train),
        "n_test": len(test),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="pilot_v1")
    parser.add_argument("--split", default="pair_id", choices=["pair_id", "seed"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ws = get_workspace()
    config = {"dataset": args.dataset, "model": "mlp_64x64", "split": args.split, "seed": args.seed}
    run = start_run("rq1_baseline_mlp", config, ws)
    if run.is_complete and not args.force:
        print(f"already complete: {run.path}")
        return

    data = add_labels(pd.read_parquet(ws.dataset_dir(args.dataset) / "consolidated" / "dataset.parquet"))
    parts = split_by_group(data, args.split, rng_seed=args.seed)
    train, test = parts["train"], parts["test"]

    metrics = {
        "fault_type_given_fault": fit_eval(
            train[train.fault_active == 1], test[test.fault_active == 1], "fault_class", args.seed
        ),
        "normal_vs_fault": fit_eval(train, test, "fault_active", args.seed),
    }
    run.write_metrics(metrics)
    run.finish()
    print(metrics, "->", run.path)


if __name__ == "__main__":
    main()
