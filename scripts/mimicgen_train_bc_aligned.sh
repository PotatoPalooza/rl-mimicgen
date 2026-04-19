#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/mimicgen_train_bc_aligned.sh --task TASK --variant VARIANT --modality MODALITY [options]

Runs the sidecar aligned diffusion BC pipeline:
1. generate a diffusion_policy config with a shared runtime profile
2. disable mismatched internal rollout checkpoint selection by default
3. train robomimic BC
4. evaluate saved BC checkpoints with the RL-stack eval path
5. write the best checkpoint path under rl_aligned_eval/

Options:
  --task TASK
  --variant VARIANT
  --modality MODALITY
  --diffusion-runtime-profile NAME   Default: dppo_ddim5_ft5_act8
  --experiment-name NAME             Optional fixed experiment folder name.
  --run-root PATH
  --data-dir PATH
  --save-every-n-epochs N            Default: 200
  --eval-episodes N                  Default: 20
  --eval-num-envs N                  Default: 4
  --device DEVICE                    Default: cuda
  --no-download-datasets
  --no-run-training
  --no-select-best
  --no-disable-internal-rollout-eval
  --dry-run
  --help
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD=(python3 -m rl_mimicgen.mimicgen.aligned_bc_one_task)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help)
      usage
      exit 0
      ;;
    *)
      CMD+=("$1")
      shift
      ;;
  esac
done

cd "$REPO_ROOT"
"${CMD[@]}"
