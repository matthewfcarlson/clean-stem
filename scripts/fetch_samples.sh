#!/usr/bin/env bash
# Copy the sample tracks listed in samples/manifest.tsv from a local music library.
# Usage: scripts/fetch_samples.sh [MUSIC_ROOT]   (default: /Volumes/Media/Music)
set -euo pipefail

root="${1:-${MUSIC_ROOT:-/Volumes/Media/Music}}"
here="$(cd "$(dirname "$0")/.." && pwd)"
dest="$here/samples"

grep -v '^#' "$dest/manifest.tsv" | while IFS=$'\t' read -r id src _expect _notes; do
  [ -n "$id" ] || continue
  out="$dest/$id.${src##*.}"
  if [ -f "$out" ]; then
    echo "skip  $id (already present)"
  else
    cp "$root/$src" "$out"
    echo "copy  $id"
  fi
done
