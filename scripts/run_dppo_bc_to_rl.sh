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
  --auto-skip-pretrain    Reuse completed pretrain outputs when present, otherwise run pretrain.
  --skip-pretrain         Require completed pretrain outputs and do not run BC pretrain.
  --every-n N             Optional sweep checkpoint stride override. Default: 2
  --n-episodes N          Optional sweep completed-episode override. Default: 40
  --checkpoint-dir PATH   Optional checkpoint directory to sweep.
  --help                  Show this help.

Examples:
  scripts/run_dppo_bc_to_rl.sh --task coffee_d1
  scripts/run_dppo_bc_to_rl.sh --task coffee_d1 --auto-skip-pretrain
  scripts/run_dppo_bc_to_rl.sh --task square_d0 --every-n 2 --n-episodes 40
EOF
}

TASK=""
EVERY_N="2"
N_EPISODES="40"
CHECKPOINT_DIR=""
AUTO_SKIP_PRETRAIN=0
SKIP_PRETRAIN=0

find_completed_pretrain_checkpoint() {
  local task="$1"
  local task_name
  task_name="$(basename "$task")"
  task_name="${task_name%.yaml}"
  "$PYTHON_BIN" - "$REPO_ROOT" "$task_name" <<PY
from pathlib import Path
import re
import sys
from omegaconf import OmegaConf


repo_root = Path(sys.argv[1])
dataset_id = sys.argv[2]
log_root = repo_root / "logs" / "official_dppo" / "mimicgen"
pretrain_root = log_root / "pretrain" / dataset_id


def _run_dir_sort_token(path: Path) -> str:
    name = path.name
    matches = re.findall(r"(\\d{8}_\\d{6}|\\d{4}-\\d{2}-\\d{2}_\\d{2}-\\d{2}-\\d{2})", name)
    if matches:
        return matches[-1].replace("-", "")
    return name


run_dirs = sorted(
    (path for path in pretrain_root.glob("*/*") if path.is_dir()),
    key=_run_dir_sort_token,
    reverse=True,
)

for run_dir in run_dirs:
    best_checkpoint = run_dir / "best_checkpoint.pt"
    if best_checkpoint.exists():
        print(best_checkpoint.as_posix())
        raise SystemExit(0)

    run_config_path = run_dir / "run_config.yaml"
    if not run_config_path.exists():
        continue

    try:
        cfg = OmegaConf.load(run_config_path)
    except Exception:
        continue

    train_cfg = cfg.get("train")
    if not train_cfg:
        continue

    n_epochs = train_cfg.get("n_epochs")
    if n_epochs is None:
        continue
    try:
        n_epochs_int = int(n_epochs)
    except (TypeError, ValueError):
        continue

    final_checkpoint = run_dir / "checkpoint" / f"state_{n_epochs_int}.pt"
    if final_checkpoint.exists():
        print(final_checkpoint.as_posix())
        raise SystemExit(0)

print("")
PY
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --task)
      TASK="$2"
      shift 2
      ;;
    --auto-skip-pretrain)
      AUTO_SKIP_PRETRAIN=1
      shift
      ;;
    --skip-pretrain)
      SKIP_PRETRAIN=1
      shift
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

task_name="$(basename "$TASK")"
task_name="${task_name%.yaml}"
dataset_id="$task_name"

echo "==> Preparing $TASK"
(
  cd "$REPO_ROOT"
  "$PYTHON_BIN" "$LAUNCHER" prepare --task "$TASK"
)

if [[ "$SKIP_PRETRAIN" -eq 1 ]]; then
  echo "==> Skipping BC pretrain for $TASK because --skip-pretrain was set."
  if [[ -z "$CHECKPOINT_DIR" ]]; then
    completed_pretrain_checkpoint="$(find_completed_pretrain_checkpoint "$TASK")"
    if [[ -z "$completed_pretrain_checkpoint" ]]; then
      echo "Failed to find a completed pretrain checkpoint directory for task $dataset_id while --skip-pretrain is set." >&2
      exit 1
    fi
  fi
elif [[ "$AUTO_SKIP_PRETRAIN" -eq 1 ]]; then
  completed_pretrain_checkpoint="$(find_completed_pretrain_checkpoint "$TASK")"
  if [[ -n "$completed_pretrain_checkpoint" ]]; then
    echo "==> Reusing completed BC pretrain outputs for $TASK"
  else
    echo "==> No completed BC pretrain found for $TASK. Running BC pretrain."
    (
      cd "$REPO_ROOT"
      "$PYTHON_BIN" "$LAUNCHER" pretrain --task "$TASK"
    )
  fi
else
  echo "==> Pretraining BC for $TASK"
  (
    cd "$REPO_ROOT"
    "$PYTHON_BIN" "$LAUNCHER" pretrain --task "$TASK"
  )
fi

if [[ -z "$CHECKPOINT_DIR" ]]; then
  if [[ "$AUTO_SKIP_PRETRAIN" -eq 1 || "$SKIP_PRETRAIN" -eq 1 ]]; then
    latest_checkpoint="$(find_completed_pretrain_checkpoint "$TASK")"
  else
    latest_checkpoint="$(
      find "$REPO_ROOT/logs/official_dppo/mimicgen/pretrain/$dataset_id" -path '*/checkpoint/state_*.pt' 2>/dev/null \
        | sort | tail -1
    )"
  fi
  if [[ -z "$latest_checkpoint" ]]; then
    if [[ "$SKIP_PRETRAIN" -eq 1 ]]; then
      echo "Failed to find a completed pretrain checkpoint directory for task $dataset_id while --skip-pretrain is set." >&2
    elif [[ "$AUTO_SKIP_PRETRAIN" -eq 1 ]]; then
      echo "Failed to find a completed pretrain checkpoint directory for task $dataset_id." >&2
    else
      echo "Failed to find a pretrain checkpoint directory for task $dataset_id." >&2
    fi
    exit 1
  fi
  CHECKPOINT_DIR="$(dirname "$latest_checkpoint")"
fi

sweep_args=(sweep --task "$TASK")
if [[ "$AUTO_SKIP_PRETRAIN" -eq 1 ]]; then
  sweep_args+=(--auto-skip-pretrain)
fi
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
