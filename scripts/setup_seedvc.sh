#!/usr/bin/env bash
# Set up Seed-VC for `clean-stem clean --fill sung`, in its own environment.
# Usage: scripts/setup_seedvc.sh [DIR]   (default: ~/.cache/clean-stem/seed-vc)
# Then: export CLEAN_STEM_SEEDVC=DIR   (or pass --seedvc DIR)
#
# Seed-VC (GPL-3.0) is cloned at a pinned commit; its Python 3.10 / torch 2.4
# environment goes in DIR/.venv. Models (~1.5 GB) download on first use into
# DIR/checkpoints. Off macOS, the replacement words are spoken with espeak-ng,
# so install that too (e.g. apt install espeak-ng).
set -euo pipefail

SEEDVC_REPO=https://github.com/Plachtaa/seed-vc.git
SEEDVC_COMMIT=51383efd921027683c89e5348211d93ff12ac2a8

dir="${1:-$HOME/.cache/clean-stem/seed-vc}"
if [ ! -d "$dir/.git" ]; then
  git clone "$SEEDVC_REPO" "$dir"
fi
git -C "$dir" fetch --quiet origin "$SEEDVC_COMMIT" 2>/dev/null || true
git -C "$dir" checkout --quiet "$SEEDVC_COMMIT"

uv venv --allow-existing -p 3.10 "$dir/.venv"
VIRTUAL_ENV="$dir/.venv" uv pip install \
  torch==2.4.0 torchaudio==2.4.0 scipy==1.13.1 librosa==0.10.2 \
  "huggingface-hub>=0.28.1,<1" munch==4.0.0 einops==0.8.0 \
  descript-audio-codec==1.0.0 transformers==4.46.3 soundfile==0.12.1 \
  numpy==1.26.4 pyyaml hydra-core==1.3.2 accelerate pyworld "setuptools<80"

if ! command -v say >/dev/null && ! command -v espeak-ng >/dev/null && ! command -v espeak >/dev/null; then
  echo "warning: no text-to-speech engine found; install espeak-ng" >&2
fi
echo "Seed-VC ready. Use: export CLEAN_STEM_SEEDVC=$dir"
