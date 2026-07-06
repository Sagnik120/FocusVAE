#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python src/train.py --config configs/vanilla.yaml
