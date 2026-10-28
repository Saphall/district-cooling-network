#!/usr/bin/env bash
# Resumable parallel generation on a server. Safe to re-run or interrupt at any time.
# Usage: scripts/run_generation.sh [dataset] [workers] [workspace_dir]
set -euo pipefail

cd "$(dirname "$0")/.."
DATASET="${1:-pilot_v1}"
WORKERS="${2:-$(nproc)}"
export SWIM_WORKSPACE="${3:-${SWIM_WORKSPACE:-$PWD}}"

uv run dcn generate "$DATASET" --workers "$WORKERS"
uv run dcn consolidate "$DATASET"
uv run dcn manifest status "$DATASET"
