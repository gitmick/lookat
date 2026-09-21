#!/usr/bin/env bash
# Raspberry Pi OS Bookworm (64-bit) setup. Run from the project root.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== system packages =="
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev python3-pip \
    libgl1 libglib2.0-0 libatlas-base-dev \
    python3-picamera2 libcap-dev

echo "== virtualenv (with system site packages, so picamera2 stays visible) =="
python3 -m venv --system-site-packages .venv
./.venv/bin/pip install --upgrade pip wheel
./.venv/bin/pip install -e .

echo "== face landmark model =="
./scripts/fetch_model.sh

echo
echo "Done. Test it with:"
echo "  ./.venv/bin/python run.py --windowed --debug"
