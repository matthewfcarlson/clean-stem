#!/usr/bin/env bash
# Run a clean-stem command on a faster machine over SSH, e.g.
#   CLEAN_STEM_REMOTE=me@studio scripts/remote.sh bench run --jobs 8 --label try1
#
# Syncs this working tree (uncommitted changes and samples/ included) to
# ~/clean-stem there, runs `uv run clean-stem ARGS` and copies the bench
# results back. The remote needs uv, ffmpeg and, for the sung fill,
# scripts/setup_seedvc.sh run once; the bench corpus is built there with
# `scripts/remote.sh bench corpus` or synced from here.
set -euo pipefail

host="${CLEAN_STEM_REMOTE:?set CLEAN_STEM_REMOTE to user@host}"
here="$(cd "$(dirname "$0")/.." && pwd)"
bench="${CLEAN_STEM_BENCH:-$HOME/.cache/clean-stem/bench}"

rsync -az --delete --exclude .venv --exclude __pycache__ --exclude .pytest_cache --exclude .DS_Store "$here/" "${host}:clean-stem/"
args=$(printf '%q ' "$@")
ssh -t "${host}" "export PATH=\$HOME/.local/bin:\$PATH CLEAN_STEM_SEEDVC=\${CLEAN_STEM_SEEDVC:-\$HOME/.cache/clean-stem/seed-vc}; cd ~/clean-stem && uv run --quiet clean-stem ${args}"
if [ "${1:-}" = bench ]; then
  mkdir -p "$bench"
  rsync -az "${host}:.cache/clean-stem/bench/results" "$bench/"
fi
