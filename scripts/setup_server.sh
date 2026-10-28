#!/usr/bin/env bash
# One-time setup on a Linux CPU server. Usage: scripts/setup_server.sh [workspace_dir]
# The workspace holds data/, datasets/, results/, reports/ (default: the project directory).
set -euo pipefail

cd "$(dirname "$0")/.."
export SWIM_WORKSPACE="${1:-${SWIM_WORKSPACE:-$PWD}}"
mkdir -p "$SWIM_WORKSPACE"

command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/"; exit 1; }
uv sync --all-extras

uv run --all-extras dcn setup-dcngen
uv run --all-extras dcn download-data
uv run --all-extras dcn doctor
echo "workspace: $SWIM_WORKSPACE"
