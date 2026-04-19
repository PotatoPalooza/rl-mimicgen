#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/install_mimicgen_datasets.sh [options]

Installs released MimicGen core datasets into the default local dataset root:
  runs/datasets/core/

Options:
  --task TASK             Download one dataset family, for example square or coffee.
  --variant VARIANT       Limit to one or more variants, for example D0 or D1. Repeatable.
  --all-core              Download all released core dataset families.
  --data-dir PATH         Dataset root. Default: <repo>/runs/datasets
  --modality MODALITY     Modality to configure during download. Default: low_dim
  --help                  Show this help.

Examples:
  bash scripts/install_mimicgen_datasets.sh --task square --variant D0
  bash scripts/install_mimicgen_datasets.sh --task coffee --variant D1 --variant D2
  bash scripts/install_mimicgen_datasets.sh --all-core
EOF
}

TASK=""
ALL_CORE=0
DATA_DIR=""
MODALITY="low_dim"
VARIANTS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"
      shift 2
      ;;
    --variant)
      VARIANTS+=("$2")
      shift 2
      ;;
    --all-core)
      ALL_CORE=1
      shift
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --modality)
      MODALITY="$2"
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

if [[ -n "$TASK" && "$ALL_CORE" -eq 1 ]]; then
  echo "Use either --task or --all-core, not both." >&2
  exit 2
fi

if [[ -z "$TASK" && "$ALL_CORE" -ne 1 ]]; then
  echo "Specify --task or --all-core." >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/runs/datasets}"

TASKS=()
if [[ "$ALL_CORE" -eq 1 ]]; then
  TASKS=(
    stack
    stack_three
    square
    threading
    three_piece_assembly
    coffee
    coffee_preparation
    nut_assembly
    pick_place
    hammer_cleanup
    mug_cleanup
    kitchen
  )
else
  TASKS=("$TASK")
fi

for task_name in "${TASKS[@]}"; do
  cmd=(
    "$PYTHON_BIN"
    -m rl_mimicgen.mimicgen.paper_bc_one_task
    --task "$task_name"
    --modality "$MODALITY"
    --data-dir "$DATA_DIR"
    --no-run-training
  )
  for variant in "${VARIANTS[@]}"; do
    cmd+=(--variant "$variant")
  done
  echo "Installing datasets for task=$task_name into $DATA_DIR"
  (
    cd "$REPO_ROOT"
    "${cmd[@]}"
  )
done

echo "Datasets available under: $DATA_DIR/core"
