#!/usr/bin/env bash
set -euo pipefail

MUJOCO_GL=glx PYTHONPATH=. .venv/bin/python scripts/train_online_rl.py \
  --config config/online_rl_coffee_vc1.json \
  "$@"
