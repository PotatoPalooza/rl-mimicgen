#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/mimicgen_train_bc.sh --task TASK [options]

Runs one MimicGen task as independent variant x modality jobs using
`rl_mimicgen.mimicgen.paper_bc_one_task`.

Options:
  --task TASK                  Required task name.
  --variant NAME              Limit to a specific variant. Repeatable.
  --modality NAME             Limit to a specific modality. Repeatable. Default: low_dim,image
  --algo NAME                 Training algorithm. Repeatable. Default: bc. Supported: bc,diffusion_policy
  --max-parallel N            Maximum concurrent jobs. Default: 2
  --run-root-base PATH        Base directory for per-job run roots.
  --data-dir PATH             Shared dataset directory.
  --no-download-datasets      Reuse datasets already present under DATA_DIR/core.
  --dry-run                   Print commands without running them.
  --help                      Show this help.

Examples:
  scripts/mimicgen_train_bc.sh --task square
  scripts/mimicgen_train_bc.sh --task square --algo diffusion_policy --variant D0 --modality low_dim
EOF
}

variants_for_task() {
  case "$1" in
    stack|stack_three) echo "D0 D1" ;;
    square|threading|three_piece_assembly|coffee) echo "D0 D1 D2" ;;
    coffee_preparation|hammer_cleanup|kitchen) echo "D0 D1" ;;
    nut_assembly|pick_place) echo "D0" ;;
    mug_cleanup) echo "D0 D1 O1 O2" ;;
    *)
      echo "Unknown task: $1" >&2
      exit 2
      ;;
  esac
}

contains() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

cleanup() {
  trap - INT TERM EXIT
  local pids
  pids=$(jobs -pr)
  if [[ -n "$pids" ]]; then
    kill $pids 2>/dev/null || true
    wait $pids 2>/dev/null || true
  fi
}

wait_for_slot() {
  local limit="$1"
  while true; do
    local running
    running=$(jobs -pr | wc -l | tr -d ' ')
    if (( running < limit )); then
      break
    fi
    wait -n
  done
}

TASK=""
MAX_PARALLEL=2
DRY_RUN=0
RUN_ROOT_BASE=""
DATA_DIR=""
RUNNER_FLAGS=()
REQUESTED_VARIANTS=()
REQUESTED_MODALITIES=()
REQUESTED_ALGOS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"
      shift 2
      ;;
    --variant)
      REQUESTED_VARIANTS+=("$2")
      shift 2
      ;;
    --modality)
      REQUESTED_MODALITIES+=("$2")
      shift 2
      ;;
    --algo)
      REQUESTED_ALGOS+=("$2")
      shift 2
      ;;
    --max-parallel)
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --run-root-base)
      RUN_ROOT_BASE="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --no-download-datasets)
      RUNNER_FLAGS+=("$1")
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-parallel must be a positive integer" >&2
  exit 2
fi

if [[ ${#REQUESTED_MODALITIES[@]} -eq 0 ]]; then
  REQUESTED_MODALITIES=("low_dim" "image")
fi
if [[ ${#REQUESTED_ALGOS[@]} -eq 0 ]]; then
  REQUESTED_ALGOS=("bc")
fi

for modality in "${REQUESTED_MODALITIES[@]}"; do
  if [[ "$modality" != "low_dim" && "$modality" != "image" ]]; then
    echo "Unsupported modality: $modality" >&2
    exit 2
  fi
done

for algo in "${REQUESTED_ALGOS[@]}"; do
  if [[ "$algo" != "bc" && "$algo" != "diffusion_policy" ]]; then
    echo "Unsupported algorithm: $algo" >&2
    exit 2
  fi
done

read -r -a VALID_VARIANTS <<<"$(variants_for_task "$TASK")"
if [[ ${#REQUESTED_VARIANTS[@]} -eq 0 ]]; then
  REQUESTED_VARIANTS=("${VALID_VARIANTS[@]}")
fi

for variant in "${REQUESTED_VARIANTS[@]}"; do
  if ! contains "$variant" "${VALID_VARIANTS[@]}"; then
    echo "Unsupported variant for task $TASK: $variant" >&2
    echo "Allowed variants: ${VALID_VARIANTS[*]}" >&2
    exit 2
  fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "$RUN_ROOT_BASE" ]]; then
  RUN_ROOT_BASE="$REPO_ROOT/runs/${TASK}_parallel"
fi
mkdir -p "$RUN_ROOT_BASE"
LOG_DIR="$RUN_ROOT_BASE/logs"
mkdir -p "$LOG_DIR"
trap cleanup INT TERM EXIT

if (( DRY_RUN )); then
  echo "Dry run. Commands:"
fi

for variant in "${REQUESTED_VARIANTS[@]}"; do
  for modality in "${REQUESTED_MODALITIES[@]}"; do
    run_root="$RUN_ROOT_BASE/${variant}_${modality}"
    log_path="$LOG_DIR/${variant}_${modality}.log"
    cmd=(python3 -m rl_mimicgen.mimicgen.paper_bc_one_task --task "$TASK" --variant "$variant" --modality "$modality" --run-root "$run_root")
    for algo in "${REQUESTED_ALGOS[@]}"; do
      cmd+=(--algo "$algo")
    done
    if [[ -n "$DATA_DIR" ]]; then
      cmd+=(--data-dir "$DATA_DIR")
    fi
    if (( DRY_RUN )); then
      printf '%q ' "${cmd[@]}" "${RUNNER_FLAGS[@]}"
      printf '\n'
      continue
    fi
    wait_for_slot "$MAX_PARALLEL"
    echo "Launching $TASK $variant $modality -> $log_path"
    (
      cd "$REPO_ROOT"
      "${cmd[@]}" "${RUNNER_FLAGS[@]}" >"$log_path" 2>&1
    ) &
  done
done

if (( ! DRY_RUN )); then
  wait
fi
