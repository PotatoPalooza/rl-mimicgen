#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_ROOT="${TASK_ROOT:-$REPO_ROOT/configs/mimicgen_tasks}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/official_dppo/mimicgen}"

usage() {
  cat <<'EOF'
Usage: bash scripts/dppo_task_status.sh

Print per-task official DPPO status using on-disk run artifacts.

Environment:
  TASK_ROOT   Task spec directory. Default: configs/mimicgen_tasks
  LOG_ROOT    Official DPPO log root. Default: logs/official_dppo/mimicgen
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

shopt -s nullglob

run_dirs_for_task() {
  local stage_root="$1"
  local task="$2"
  local stage_dir="$LOG_ROOT/$stage_root/$task"
  if [[ ! -d "$stage_dir" ]]; then
    return 0
  fi
  find "$stage_dir" -mindepth 2 -maxdepth 2 -type d | sort
}

extract_yaml_scalar() {
  local key="$1"
  local path="$2"
  awk -F': *' -v key="$key" '$1 ~ ("^" key "$") { print $2; exit }' "$path"
}

has_completed_pretrain() {
  local task="$1"
  local run_dir config_path n_epochs
  while IFS= read -r run_dir; do
    config_path="$run_dir/run_config.yaml"
    [[ -f "$config_path" ]] || continue
    n_epochs="$(extract_yaml_scalar "  n_epochs" "$config_path" | tr -d '[:space:]')"
    [[ -n "$n_epochs" ]] || continue
    [[ -f "$run_dir/checkpoint/state_${n_epochs}.pt" ]] && return 0
  done < <(run_dirs_for_task "pretrain" "$task")
  return 1
}

has_started_pretrain() {
  local task="$1"
  local run_dir
  while IFS= read -r run_dir; do
    [[ -n "$run_dir" ]] && return 0
  done < <(run_dirs_for_task "pretrain" "$task")
  return 1
}

has_completed_finetune() {
  local task="$1"
  local run_dir config_path n_train_itr final_itr
  while IFS= read -r run_dir; do
    config_path="$run_dir/run_config.yaml"
    [[ -f "$config_path" ]] || continue
    n_train_itr="$(extract_yaml_scalar "  n_train_itr" "$config_path" | tr -d '[:space:]')"
    [[ -n "$n_train_itr" ]] || continue
    final_itr=$((n_train_itr - 1))
    [[ -f "$run_dir/checkpoint/state_${final_itr}.pt" ]] && return 0
  done < <(run_dirs_for_task "finetune" "$task")
  return 1
}

has_started_finetune() {
  local task="$1"
  local run_dir
  while IFS= read -r run_dir; do
    [[ -n "$run_dir" ]] && return 0
  done < <(run_dirs_for_task "finetune" "$task")
  return 1
}

overall_status() {
  local pretrain_status="$1"
  local finetune_status="$2"
  if [[ "$finetune_status" == "done" ]]; then
    printf 'finished finetuning'
  elif [[ "$pretrain_status" == "done" ]]; then
    printf 'finished pretraining'
  elif [[ "$pretrain_status" == "started" || "$finetune_status" == "started" ]]; then
    printf 'started'
  else
    printf 'no start'
  fi
}

printf '%-28s %-10s %-10s %s\n' "TASK" "PRETRAIN" "FINETUNE" "OVERALL"

for task_path in "$TASK_ROOT"/*.yaml; do
  task="$(basename "$task_path" .yaml)"

  pretrain_status="no"
  finetune_status="no"

  if has_completed_pretrain "$task"; then
    pretrain_status="done"
  elif has_started_pretrain "$task"; then
    pretrain_status="started"
  fi

  if has_completed_finetune "$task"; then
    finetune_status="done"
  elif has_started_finetune "$task"; then
    finetune_status="started"
  fi

  printf '%-28s %-10s %-10s %s\n' \
    "$task" \
    "$pretrain_status" \
    "$finetune_status" \
    "$(overall_status "$pretrain_status" "$finetune_status")"
done
