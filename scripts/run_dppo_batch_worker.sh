#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_dppo_batch_worker.sh [options]

Starts a worker that claims TODO tasks from the batch queue and runs the pipeline.

Options:
  --auto-skip-pretrain    Pass --auto-skip-pretrain to run_dppo_bc_to_rl.sh.
  --skip-pretrain         Legacy alias for --auto-skip-pretrain.
  --no-auto-skip-pretrain Explicitly disable auto pretrain skipping for this worker.
  --help                  Show this help.

Environment:
  PIPELINE_SCRIPT   Pipeline command override. Defaults scripts/run_dppo_bc_to_rl.sh.
  TASK_FILE         Task queue file path.
  LOCK_FILE         Queue lock file path.
  WORKER_ID         Worker identifier.
  RUN_ID            Output log directory suffix.
  LOG_ROOT          Worker log root directory.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_SCRIPT="$REPO_ROOT/scripts/run_dppo_bc_to_rl.sh"
TASK_FILE="${TASK_FILE:-$REPO_ROOT/scripts/dppo_batch_tasks.txt}"
LOCK_FILE="${LOCK_FILE:-$TASK_FILE.lock}"
WORKER_ID="${WORKER_ID:-$(hostname)-$$}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${WORKER_ID}}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/official_dppo/mimicgen/batch}"
LOG_DIR="$LOG_ROOT/$RUN_ID"
AUTO_SKIP_PRETRAIN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto-skip-pretrain)
      AUTO_SKIP_PRETRAIN=1
      shift
      ;;
    --skip-pretrain)
      AUTO_SKIP_PRETRAIN=1
      shift
      ;;
    --no-auto-skip-pretrain)
      AUTO_SKIP_PRETRAIN=0
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

mkdir -p "$LOG_DIR"
touch "$TASK_FILE"
exec 9>"$LOCK_FILE"

timestamp() {
  date --iso-8601=seconds
}

log() {
  printf '[%s] [%s] %s\n' "$(timestamp)" "$WORKER_ID" "$*"
}

claim_task() {
  local claimed=""
  local tmp
  tmp="$(mktemp)"

  flock 9
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ -z "$claimed" && "$line" =~ ^TODO[[:space:]]+([^[:space:]]+)$ ]]; then
      claimed="${BASH_REMATCH[1]}"
      printf 'RUNNING %s %s %s\n' "$claimed" "$WORKER_ID" "$(timestamp)" >> "$tmp"
    else
      printf '%s\n' "$line" >> "$tmp"
    fi
  done < "$TASK_FILE"
  mv "$tmp" "$TASK_FILE"
  flock -u 9

  printf '%s' "$claimed"
}

mark_task() {
  local target_task="$1"
  local target_status="$2"
  local tmp
  tmp="$(mktemp)"

  flock 9
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^RUNNING[[:space:]]+${target_task}([[:space:]]|$) ]]; then
      printf '%s %s %s %s\n' "$target_status" "$target_task" "$WORKER_ID" "$(timestamp)" >> "$tmp"
    else
      printf '%s\n' "$line" >> "$tmp"
    fi
  done < "$TASK_FILE"
  mv "$tmp" "$TASK_FILE"
  flock -u 9
}

count_pending() {
  flock 9
  awk 'BEGIN { n = 0 } /^TODO[[:space:]]+/ { n += 1 } END { print n }' "$TASK_FILE"
  flock -u 9
}

log "Queue file: $TASK_FILE"
log "Lock file: $LOCK_FILE"
log "Run log dir: $LOG_DIR"

had_failure=0

while true; do
  task="$(claim_task)"
  if [[ -z "$task" ]]; then
    pending="$(count_pending)"
    if [[ "$pending" == "0" ]]; then
      log "No TODO tasks remain. Exiting."
      exit "$had_failure"
    fi
    log "No claimable task found. Exiting."
    exit "$had_failure"
  fi

  task_log="$LOG_DIR/${task}.log"
  log "Starting task: $task"
  log "Task log: $task_log"

  if (
    echo "[start] $(timestamp) worker=$WORKER_ID task=$task"
    if [[ "$AUTO_SKIP_PRETRAIN" -eq 1 ]]; then
      bash "$PIPELINE_SCRIPT" --task "$task" --auto-skip-pretrain
    else
      bash "$PIPELINE_SCRIPT" --task "$task"
    fi
    echo "[done] $(timestamp) worker=$WORKER_ID task=$task"
  ) 2>&1 | tee "$task_log"; then
    mark_task "$task" "DONE"
    log "Completed task: $task"
  else
    mark_task "$task" "FAILED"
    had_failure=1
    log "Task failed: $task"
    log "See log: $task_log"
    log "Continuing to next TODO task."
  fi
done
