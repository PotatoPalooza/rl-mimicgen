#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_dppo_bc_to_rl.sh --task TASK [options]

Runs the official low-dim DPPO pipeline for one MimicGen task:
1. prepare task artifacts/configs
2. pretrain BC
3. sweep saved BC checkpoints
4. eval the best swept BC checkpoint
5. finetune RL from that best checkpoint

Options:
  --task TASK             Required task id or path to a task spec YAML.
  --every-n N             Optional sweep checkpoint stride override. Default: 2
  --n-episodes N          Optional sweep completed-episode override. Default: 10
  --checkpoint-dir PATH   Optional checkpoint directory to sweep.
  --group GROUP           Optional wandb group tag (default: variant suffix of
                          --task, e.g. stack_d0 -> d0). Links BC + RL runs.
  --help                  Show this help.

Examples:
  scripts/run_dppo_bc_to_rl.sh --task coffee_d1
  scripts/run_dppo_bc_to_rl.sh --task square_d0 --every-n 2 --n-episodes 20
EOF
}

TASK=""
EVERY_N="2"
N_EPISODES="10"
CHECKPOINT_DIR=""
GROUP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"
      shift 2
      ;;
    --every-n)
      EVERY_N="$2"
      shift 2
      ;;
    --n-episodes)
      N_EPISODES="$2"
      shift 2
      ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --group)
      GROUP="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$TASK" ]]; then
  echo "--task is required" >&2
  usage >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
LAUNCHER="$REPO_ROOT/scripts/run_official_dppo_mimicgen.py"

if [[ -n "$GROUP" ]]; then
  export WANDB_RUN_GROUP="$GROUP"
  echo "==> wandb group: $GROUP"
fi

task_name="$(basename "$TASK")"
task_name="${task_name%.yaml}"
dataset_id="$task_name"

echo "==> Preparing $TASK"
(
  cd "$REPO_ROOT"
  "$PYTHON_BIN" "$LAUNCHER" prepare --task "$TASK"
)

echo "==> Pretraining BC for $TASK"
(
  cd "$REPO_ROOT"
  "$PYTHON_BIN" "$LAUNCHER" pretrain --task "$TASK"
)

if [[ -z "$CHECKPOINT_DIR" ]]; then
  latest_checkpoint="$(
    find "$REPO_ROOT/logs/official_dppo/mimicgen/pretrain/$dataset_id" -path '*/checkpoint/state_*.pt' 2>/dev/null \
      | sort | tail -1
  )"
  if [[ -z "$latest_checkpoint" ]]; then
    echo "Failed to find a pretrain checkpoint directory for task $dataset_id." >&2
    exit 1
  fi
  CHECKPOINT_DIR="$(dirname "$latest_checkpoint")"
fi

sweep_args=(sweep --task "$TASK")
if [[ -n "$EVERY_N" ]]; then
  sweep_args+=(--every-n "$EVERY_N")
fi
if [[ -n "$N_EPISODES" ]]; then
  sweep_args+=(--n-episodes "$N_EPISODES")
fi
if [[ -n "$CHECKPOINT_DIR" ]]; then
  sweep_args+=(--checkpoint-dir "$CHECKPOINT_DIR")
fi

echo "==> Sweeping BC checkpoints for $TASK"
tmp_sweep_json="$(mktemp)"
(
  cd "$REPO_ROOT"
  "$PYTHON_BIN" "$LAUNCHER" "${sweep_args[@]}" | tee "$tmp_sweep_json"
)

latest_sweep_best="$(
  "$PYTHON_BIN" - <<'PY' "$tmp_sweep_json"
import json
import sys
from pathlib import Path

payload_text = Path(sys.argv[1]).read_text(encoding="utf-8")
start = payload_text.rfind("\n{")
if start != -1:
    payload_text = payload_text[start + 1 :]
payload = json.loads(payload_text)
print(payload.get("best_checkpoint_copy") or payload.get("best_checkpoint") or "")
PY
)"
rm -f "$tmp_sweep_json"

if [[ -z "$latest_sweep_best" ]]; then
  echo "Failed to find best_checkpoint.pt for task $dataset_id after sweep." >&2
  exit 1
fi

echo "==> Evaluating best BC checkpoint"
(
  cd "$REPO_ROOT"
  "$PYTHON_BIN" "$LAUNCHER" eval-bc --task "$TASK" --checkpoint "$latest_sweep_best"
)

echo "==> Finetuning RL from best BC checkpoint"
(
  cd "$REPO_ROOT"
  "$PYTHON_BIN" "$LAUNCHER" finetune --task "$TASK" --checkpoint "$latest_sweep_best"
)

echo "Best checkpoint:"
echo "  $latest_sweep_best"
