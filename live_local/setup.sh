#!/bin/bash
# BahuSanket live demo: one-time setup on a Mac (about 20-30 minutes, mostly downloads). Run: bash setup.sh
set -e
cd "$(dirname "$0")"
echo "== 1/4 The BahuSanket model bundle"
if [ ! -f bundle/model_point_indices_v2.json ]; then
  ZIP=""
  for z in bahusanket_bundle.zip ~/Downloads/bahusanket_bundle.zip; do [ -f "$z" ] && ZIP="$z" && break; done
  if [ -z "$ZIP" ]; then      # Safari often unzips downloads by itself: use the folder then
    F="$(find bahusanket_bundle ~/Downloads/bahusanket_bundle -name model_point_indices_v2.json 2>/dev/null | head -1)"
    if [ -n "$F" ]; then mkdir -p bundle && cp -R "$(dirname "$F")"/. bundle/ && echo "copied the unzipped bundle folder"; fi
  fi
  if [ -z "$ZIP" ] && [ ! -f bundle/model_point_indices_v2.json ]; then
    echo "Could not find bahusanket_bundle.zip. Download it from your Google Drive (BahuSanket/live_bundle/) into this folder"
    echo "or into Downloads, then run: bash setup.sh"; exit 1
  fi
  [ -n "$ZIP" ] && mkdir -p bundle && unzip -q -o "$ZIP" -d bundle && echo "unpacked $ZIP"
else
  echo "already unpacked"
fi
echo "== 2/4 uv (installs the right Python for this project only)"
if ! command -v uv >/dev/null 2>&1; then
  [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "== 3/4 Python 3.12 and the libraries (into .venv in this folder)"
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
echo "== 4/4 The Hugging Face models (about 4.5 GB, once)"
.venv/bin/python download_models.py
echo
echo "Setup finished. Start the demo with: bash run.sh"
