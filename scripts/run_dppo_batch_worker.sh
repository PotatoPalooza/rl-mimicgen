#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_dppo_batch_worker.sh [options]

Starts a worker that claims TODO tasks from the batch queue and runs the pipeline.

Options:
  --auto-skip-pretrain    Pass --auto-skip-pretrain to run_dppo_bc_to_rl.sh.
  --skip-pretrain         Pass --skip-pretrain to run_dppo_bc_to_rl.sh.
  --no-auto-skip-pretrain Explicitly disable auto pretrain skipping for this worker.
  --help                  Show this help.

Environment:
  PIPELINE_SCRIPT   Pipeline command override. Defaults scripts/run_dppo_bc_to_rl.sh.
  TASK_FILE         Task queue file path.
  LOCK_FILE         Queue lock file path.
  WORKER_ID         Worker identifier.
  RUN_ID            Output log directory suffix.
  LOG_ROOT          Worker log root directory.
  MAX_RETRIES       Maximum restart count after failures. Default: 3.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-$REPO_ROOT/scripts/run_dppo_bc_to_rl.sh}"
TASK_FILE="${TASK_FILE:-$REPO_ROOT/scripts/dppo_batch_tasks.txt}"
LOCK_FILE="${LOCK_FILE:-$TASK_FILE.lock}"
HOSTNAME_VALUE="$(hostname)"
WORKER_ID="${WORKER_ID:-${HOSTNAME_VALUE}-$$}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${WORKER_ID}}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/official_dppo/mimicgen/batch}"
LOG_DIR="$LOG_ROOT/$RUN_ID"
MAX_RETRIES="${MAX_RETRIES:-3}"
AUTO_SKIP_PRETRAIN=0
SKIP_PRETRAIN=0
CURRENT_TASK=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto-skip-pretrain)
      AUTO_SKIP_PRETRAIN=1
      shift
      ;;
    --skip-pretrain)
      SKIP_PRETRAIN=1
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

worker_is_live() {
  local worker="$1"
  if [[ ! "$worker" =~ ^(.+)-([0-9]+)$ ]]; then
    return 1
  fi

  local worker_host="${BASH_REMATCH[1]}"
  local worker_pid="${BASH_REMATCH[2]}"
  if [[ "$worker_host" != "$HOSTNAME_VALUE" ]]; then
    return 0
  fi
  kill -0 "$worker_pid" 2>/dev/null
}

recover_stale_tasks() {
  local tmp recovered exhausted
  tmp="$(mktemp)"
  recovered=0
  exhausted=0

  flock 9
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^RUNNING[[:space:]]+([^[:space:]]+)[[:space:]]+([^[:space:]]+)[[:space:]]+([^[:space:]]+)([[:space:]]+([0-9]+))?$ ]]; then
      local task="${BASH_REMATCH[1]}"
      local worker="${BASH_REMATCH[2]}"
      local retry="${BASH_REMATCH[5]:-0}"
      if worker_is_live "$worker"; then
        printf '%s\n' "$line" >> "$tmp"
        continue
      fi

      if (( retry < MAX_RETRIES )); then
        local next_retry=$((retry + 1))
        printf 'TODO %s %s\n' "$task" "$next_retry" >> "$tmp"
        recovered=$((recovered + 1))
      else
        printf 'FAILED %s %s %s %s\n' "$task" "$WORKER_ID" "$(timestamp)" "$retry" >> "$tmp"
        exhausted=$((exhausted + 1))
      fi
    else
      printf '%s\n' "$line" >> "$tmp"
    fi
  done < "$TASK_FILE"
  mv "$tmp" "$TASK_FILE"
  flock -u 9

  if (( recovered > 0 )); then
    log "Recovered $recovered stale RUNNING task(s) back to TODO."
  fi
  if (( exhausted > 0 )); then
    log "Marked $exhausted stale RUNNING task(s) FAILED after exhausting retries."
  fi
}

claim_task() {
  local claimed="" claimed_retry="0"
  local tmp
  tmp="$(mktemp)"

  flock 9
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ -z "$claimed" && "$line" =~ ^TODO[[:space:]]+([^[:space:]]+)([[:space:]]+([0-9]+))?$ ]]; then
      claimed="${BASH_REMATCH[1]}"
      claimed_retry="${BASH_REMATCH[3]:-0}"
      printf 'RUNNING %s %s %s %s\n' "$claimed" "$WORKER_ID" "$(timestamp)" "$claimed_retry" >> "$tmp"
    else
      printf '%s\n' "$line" >> "$tmp"
    fi
  done < "$TASK_FILE"
  mv "$tmp" "$TASK_FILE"
  flock -u 9

  if [[ -n "$claimed" ]]; then
    printf '%s %s\n' "$claimed" "$claimed_retry"
  fi
}

resolve_task() {
  local target_task="$1"
  local outcome="$2"
  local tmp
  tmp="$(mktemp)"

  flock 9
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^RUNNING[[:space:]]+${target_task}[[:space:]]+([^[:space:]]+)[[:space:]]+([^[:space:]]+)([[:space:]]+([0-9]+))?$ ]]; then
      local retry="${BASH_REMATCH[4]:-0}"
      if [[ "$outcome" == "success" ]]; then
        printf 'DONE %s %s %s %s\n' "$target_task" "$WORKER_ID" "$(timestamp)" "$retry" >> "$tmp"
      else
        if (( retry < MAX_RETRIES )); then
          local next_retry=$((retry + 1))
          printf 'TODO %s %s\n' "$target_task" "$next_retry" >> "$tmp"
        else
          printf 'FAILED %s %s %s %s\n' "$target_task" "$WORKER_ID" "$(timestamp)" "$retry" >> "$tmp"
        fi
      fi
    else
      printf '%s\n' "$line" >> "$tmp"
    fi
  done < "$TASK_FILE"
  mv "$tmp" "$TASK_FILE"
  flock -u 9
}

task_is_failed() {
  local target_task="$1"
  flock 9
  if grep -Eq "^FAILED[[:space:]]+${target_task}([[:space:]]|$)" "$TASK_FILE"; then
    flock -u 9
    return 0
  fi
  flock -u 9
  return 1
}

count_pending() {
  flock 9
  awk 'BEGIN { n = 0 } /^TODO[[:space:]]+/ { n += 1 } END { print n }' "$TASK_FILE"
  flock -u 9
}

cleanup_on_exit() {
  local status="$?"
  if [[ "$status" -ne 0 && -n "$CURRENT_TASK" ]]; then
    log "Worker exiting unexpectedly while task was active: $CURRENT_TASK"
    resolve_task "$CURRENT_TASK" "failure"
  fi
}

trap cleanup_on_exit EXIT
trap 'exit 130' INT TERM HUP

log "Queue file: $TASK_FILE"
log "Lock file: $LOCK_FILE"
log "Run log dir: $LOG_DIR"
log "Max retries per task: $MAX_RETRIES"

recover_stale_tasks

had_failure=0

while true; do
  claim_output="$(claim_task)"
  if [[ -z "$claim_output" ]]; then
    pending="$(count_pending)"
    if [[ "$pending" == "0" ]]; then
      log "No TODO tasks remain. Exiting."
      exit "$had_failure"
    fi
    log "No claimable task found. Exiting."
    exit "$had_failure"
  fi
  read -r task task_retry <<< "$claim_output"

  task_log="$LOG_DIR/${task}.log"
  log "Starting task: $task"
  log "Task retry: ${task_retry:-0}/$MAX_RETRIES"
  log "Task log: $task_log"
  CURRENT_TASK="$task"

  if (
    echo "[start] $(timestamp) worker=$WORKER_ID task=$task retry=${task_retry:-0}"
    pipeline_status=0
    if [[ "$SKIP_PRETRAIN" -eq 1 ]]; then
      bash "$PIPELINE_SCRIPT" --task "$task" --skip-pretrain || pipeline_status="$?"
    elif [[ "$AUTO_SKIP_PRETRAIN" -eq 1 ]]; then
      bash "$PIPELINE_SCRIPT" --task "$task" --auto-skip-pretrain || pipeline_status="$?"
    else
      bash "$PIPELINE_SCRIPT" --task "$task" || pipeline_status="$?"
    fi
    if [[ "$pipeline_status" -eq 0 ]]; then
      echo "[done] $(timestamp) worker=$WORKER_ID task=$task retry=${task_retry:-0}"
    fi
    exit "$pipeline_status"
  ) 2>&1 | tee "$task_log"; then
    resolve_task "$task" "success"
    CURRENT_TASK=""
    log "Completed task: $task"
  else
    resolve_task "$task" "failure"
    CURRENT_TASK=""
    had_failure=1
    if task_is_failed "$task"; then
      log "Task failed permanently after exhausting retries: $task"
    else
      log "Task failed and was re-queued with incremented retry: $task"
    fi
    log "See log: $task_log"
    log "Continuing to next TODO task."
  fi
done
