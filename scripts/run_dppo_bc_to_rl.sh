#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_dppo_bc_to_rl.sh --task TASK [options]

Runs the official low-dim DPPO pipeline for one MimicGen task:
1. prepare task artifacts/configs
2. pretrain BC (async checkpoint-eval picks the best state and writes
   best_checkpoint.pt into the run dir)
3. eval the best BC checkpoint
4. finetune RL from that best checkpoint

eval-bc and finetune omit --checkpoint and let the launcher auto-resolve
the most recent pretrain run's best_checkpoint.pt (falling back to the
latest state_*.pt if async eval was disabled).

Options:
  --task TASK             Required task id or path to a task spec YAML.
  --group GROUP           Optional wandb group tag (default: variant suffix of
                          --task, e.g. stack_d0 -> d0). Links BC + RL runs.
  --mujoco-gl BACKEND     MuJoCo GL backend for every stage (exports MUJOCO_GL).
                          One of glx|egl|osmesa. Use osmesa on WSL.
  --help                  Show this help.

Examples:
  scripts/run_dppo_bc_to_rl.sh --task coffee_d1
  scripts/run_dppo_bc_to_rl.sh --task square_d0
  MUJOCO_GL=osmesa scripts/run_dppo_bc_to_rl.sh --task stack_d0     # WSL
EOF
}

TASK=""
GROUP=""
MUJOCO_GL_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"
      shift 2
      ;;
    --group)
      GROUP="$2"
      shift 2
      ;;
    --mujoco-gl)
      MUJOCO_GL_FLAG="$2"
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

if [[ -n "$MUJOCO_GL_FLAG" ]]; then
  export MUJOCO_GL="$MUJOCO_GL_FLAG"
fi
if [[ -n "${MUJOCO_GL:-}" ]]; then
  echo "==> MUJOCO_GL=$MUJOCO_GL"
fi

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

echo "==> Evaluating best BC checkpoint"
(
  cd "$REPO_ROOT"
  "$PYTHON_BIN" "$LAUNCHER" eval-bc --task "$TASK"
)

echo "==> Finetuning RL from best BC checkpoint"
(
  cd "$REPO_ROOT"
  "$PYTHON_BIN" "$LAUNCHER" finetune --task "$TASK"
)
