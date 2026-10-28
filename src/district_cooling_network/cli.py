"""Command line entry point: ``dcn <command>`` (or ``python -m district_cooling_network.cli``)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from district_cooling_network.config import load_dataset_spec, load_dcngen_lock
from district_cooling_network.paths import get_workspace, is_colab

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _shard(text: str) -> tuple[int, int]:
    index, _, count = text.partition("/")
    return int(index), int(count)


def cmd_paths(args) -> int:
    ws = get_workspace()
    for key, value in [
        ("project_root", ws.project_root),
        ("workspace", ws.root),
        ("data_dir", ws.data_dir),
        ("dcngen_pristine", ws.dcngen_pristine),
        ("dcngen_derived", ws.dcngen_derived),
        ("datasets", ws.datasets_dir),
        ("validation", ws.validation_root),
        ("ml", ws.ml_root),
        ("reports", ws.reports_dir),
    ]:
        print(f"{key:16s} {value}")
    return 0


def cmd_doctor(args) -> int:
    from district_cooling_network.data import missing_inputs
    from district_cooling_network.dcngen_env import is_built
    from district_cooling_network.manifest import verify_legacy

    ws = get_workspace()
    problems = 0

    def check(ok: bool, label: str, hint: str = "") -> None:
        nonlocal problems
        print(f"[{'ok' if ok else '!!'}] {label}" + ("" if ok else f"  -> {hint}"))
        problems += not ok

    check(sys.version_info >= (3, 12), f"python {sys.version.split()[0]} (>=3.12)", "use Python 3.12+")
    print(f"[..] environment: {'Google Colab' if is_colab() else 'local/server'}; workspace {ws.root}")
    for mod in ("numpy", "pandas", "scipy", "pyarrow", "networkx", "yaml", "wntr"):
        check(importlib.util.find_spec(mod) is not None, f"module {mod}", "pip install -e .")
    missing = missing_inputs(ws)
    check(not missing, f"DiTEC inputs ({len(missing)} missing)", "dcn download-data")
    check(ws.dcngen_pristine.exists(), f"pristine dcngen at {ws.dcngen_pristine}", "dcn setup-dcngen")
    check(is_built(ws, load_dcngen_lock(ws)), "derived dcngen build is current", "dcn setup-dcngen")
    legacy = verify_legacy(ws)
    check(
        not legacy["changed"] and not legacy["missing"],
        f"legacy outputs intact ({len(legacy['ok'])} ok)",
        f"changed={legacy['changed'][:3]} missing={legacy['missing'][:3]}",
    )
    return 1 if problems else 0


def cmd_setup_dcngen(args) -> int:
    from district_cooling_network.dcngen_env import build_derived

    print(build_derived(force=args.force))
    return 0


def cmd_extract_patch(args) -> int:
    from district_cooling_network.dcngen_env import extract_patch

    path, warnings = extract_patch(get_workspace(), args.source, args.name)
    print(f"wrote {path} and updated configs/dcngen.lock.yaml")
    for w in warnings:
        print(f"warning: {w}")
    return 0


def cmd_download_data(args) -> int:
    from district_cooling_network.data import download_inputs

    fetched = download_inputs()
    print(f"downloaded {len(fetched)} file(s)" if fetched else "all inputs already present")
    return 0


def cmd_manifest(args) -> int:
    from district_cooling_network import manifest as mf

    ws = get_workspace()
    if args.action == "import-legacy":
        new, skipped = mf.import_legacy(ws, args.root, args.label, args.pattern)
        done = sum(1 for _ in args.root.expanduser().rglob("_SUCCESS"))
        print(f"registered {new} file(s), {skipped} already registered -> {ws.legacy_registry}")
        print(f"{done} scenario folder(s) with a _SUCCESS marker under {args.root}")
        return 0
    if args.action == "verify-legacy":
        result = mf.verify_legacy(ws)
        print({k: len(v) for k, v in result.items()})
        for key in ("changed", "missing"):
            for path in result[key]:
                print(f"{key}: {path}")
        return 1 if result["changed"] or result["missing"] else 0

    spec = load_dataset_spec(args.dataset)
    frame = mf.write_manifest(spec, ws)
    print(frame["status"].value_counts().reindex(mf.STATUSES, fill_value=0).to_string())
    print(f"manifest: {ws.dataset_dir(spec['dataset_id']) / 'manifest.csv'}")
    return 0


def _generate_parallel(args) -> int:
    """One process per shard, BLAS pinned to one thread each (as in the cloud pilot)."""
    import os
    import subprocess

    spec = load_dataset_spec(args.dataset)
    log_dir = get_workspace().dataset_dir(spec["dataset_id"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **{v: "1" for v in THREAD_VARS}}
    procs = []
    for i in range(args.workers):
        cmd = [sys.executable, "-m", "district_cooling_network.cli", "generate", args.dataset]
        cmd += ["--shard", f"{i}/{args.workers}"]
        if args.fsed:
            cmd += ["--fsed", *args.fsed]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        for flag, on in (
            ("--rebuild-stale", args.rebuild_stale),
            ("--no-retry-failed", args.no_retry_failed),
            ("--dry-run", args.dry_run),
        ):
            if on:
                cmd.append(flag)
        log = (log_dir / f"shard-{i}-of-{args.workers}.log").open("a")
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    codes = [proc.wait() for proc, _ in procs]
    for _, log in procs:
        log.close()
    from district_cooling_network.manifest import write_manifest

    write_manifest(spec)
    print(f"{args.workers} workers finished, exit codes {codes}; logs in {log_dir}")
    return int(any(codes))


def cmd_generate(args) -> int:
    from district_cooling_network.dataset.generate import generate

    if args.workers > 1:
        return _generate_parallel(args)
    spec = load_dataset_spec(args.dataset)
    report = generate(
        spec,
        shard=args.shard,
        only_fsed=set(args.fsed) if args.fsed else None,
        limit=args.limit,
        retry_failed=not args.no_retry_failed,
        rebuild_stale=args.rebuild_stale,
        dry_run=args.dry_run,
    )
    print(f"completed {len(report.completed)}, failed {len(report.failed)}")
    return 0 if report.ok else 1


def cmd_consolidate(args) -> int:
    from district_cooling_network.dataset.generate import consolidate

    spec = load_dataset_spec(args.dataset)
    for name, path in consolidate(spec, include_qa_failed=args.include_qa_failed).items():
        print(f"{name}: {path}")
    return 0


def cmd_validate(args) -> int:
    from district_cooling_network.validation.runner import ALL_NAMES, run_validation

    names = args.names or ALL_NAMES
    worst = 0
    for name in names:
        outcome = run_validation(name, force=args.force)
        print(f"{outcome.status:8s} {name}  ({outcome.report})")
        worst |= outcome.status == "failed"
    return int(worst)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dcn", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("paths", help="print resolved paths").set_defaults(func=cmd_paths)
    sub.add_parser("doctor", help="check environment, inputs and builds").set_defaults(
        func=cmd_doctor
    )

    p = sub.add_parser("setup-dcngen", help="clone pinned dcngen and build the patched copy")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_setup_dcngen)

    sub.add_parser("download-data", help="fetch missing DiTEC-WDN inputs").set_defaults(
        func=cmd_download_data
    )

    p = sub.add_parser(
        "extract-patch", help="capture an in-place modified dcngen clone as the project patch"
    )
    p.add_argument("source", type=Path, help="modified dcngen clone (read only)")
    p.add_argument("--name", default="0001-swim-fsed-extensions.patch")
    p.set_defaults(func=cmd_extract_patch)

    p = sub.add_parser("manifest", help="scenario manifest and legacy registry")
    p.add_argument("action", choices=["build", "status", "import-legacy", "verify-legacy"])
    p.add_argument("dataset", nargs="?", default="pilot_v1", help="dataset spec name or path")
    p.add_argument("--root", type=Path, help="import-legacy: output folder")
    p.add_argument("--label", default="legacy", help="import-legacy: registry label")
    p.add_argument("--pattern", default="*", help="import-legacy: glob inside --root")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("generate", help="resumable dataset generation")
    p.add_argument("dataset", nargs="?", default="pilot_v1")
    p.add_argument("--shard", type=_shard, default=(0, 1), metavar="I/N", help="this worker's shard")
    p.add_argument("--workers", type=int, default=1, help="parallel shard processes")
    p.add_argument("--fsed", nargs="+", help="restrict to these FSEDs")
    p.add_argument("--limit", type=int, help="run at most this many pairs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rebuild-stale", action="store_true")
    p.add_argument("--no-retry-failed", action="store_true")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("consolidate", help="merge committed pairs into one parquet")
    p.add_argument("dataset", nargs="?", default="pilot_v1")
    p.add_argument("--include-qa-failed", action="store_true")
    p.set_defaults(func=cmd_consolidate)

    p = sub.add_parser("validate", help="run physics validation experiments")
    p.add_argument("names", nargs="*")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "manifest" and args.action == "import-legacy" and args.root is None:
        raise SystemExit("manifest import-legacy requires --root")
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
