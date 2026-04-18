# rl-mimicgen

RL + mimicgen

## Requirements

- Python 3.11
- Blackwell GPU
- `uv`
- `gitman`

## Quickstart

```bash
pip install uv gitman

gitman install
uv sync
```

This installs `rl-mimicgen` plus editable local checkouts for `mimicgen`, `robosuite`, and `robomimic`.

`uv sync` creates the project `.venv` if it does not already exist.

If you use `just`:

```bash
just bootstrap
```

## Verify

```bash
MUJOCO_GL=glx uv run python scripts/verify_blackwell_env.py
```

Or:

```bash
just verify
```

## AWAC

AWAC is implemented as a separate non-diffusion online RL path in the same `rl/`
trainer stack. The intended starting point is a robomimic BC or BC-RNN checkpoint;
diffusion policies remain on the `dppo` path.

An example Coffee config is provided at:

- `config/online_rl_coffee_awac.json`

The quickest way to launch it is:

```bash
bash scripts/run_online_rl_coffee_awac.sh
```

The full training command is:

```bash
MUJOCO_GL=glx PYTHONPATH=. .venv/bin/python scripts/train_online_rl.py \
  --config config/online_rl_coffee_awac.json
```

This config uses the Coffee BC-RNN checkpoint already referenced by the existing
Coffee PPO configs, runs `algorithm="awac"`, and writes outputs to
`logs/online_rl_coffee_awac`.

Useful overrides for longer runs or checkpoint swaps:

```bash
MUJOCO_GL=glx PYTHONPATH=. .venv/bin/python scripts/train_online_rl.py \
  --config config/online_rl_coffee_awac.json \
  --checkpoint /abs/path/to/your/coffee_bc_rnn_checkpoint.pth \
  --output_dir logs/online_rl_coffee_awac_experiment \
  --num_envs 16 \
  --total_updates 120 \
  --rollout_steps 192
```

## Train

Run the MimicGen one-task BC runner:

```bash
uv run python -m rl_mimicgen.mimicgen.paper_bc_one_task \
  --task square \
  --variant D0 \
  --modality low_dim \
  --algo bc
```

Run the parallel runner over all policies - `variant x modality` for a task:

```bash
bash scripts/mimicgen_train_bc.sh \
  --task square \
  --max-parallel 2
```

Limit the launch set or reuse an existing demonstration dataset directory:

```bash
bash scripts/mimicgen_train_bc.sh \
  --task square \
  --variant D0 \
  --variant D1 \
  --modality low_dim \
  --data-dir /path/to/datasets \
  --no-download-datasets
```

Train a diffusion policy on MimicGen data with the same automation:

```bash
bash scripts/mimicgen_train_bc.sh \
  --task square \
  --variant D0 \
  --modality low_dim \
  --algo diffusion_policy
```

You can also ask the Python runner to emit both BC and diffusion configs in one run root:

```bash
uv run python -m rl_mimicgen.mimicgen.paper_bc_one_task \
  --task square \
  --variant D0 \
  --modality low_dim \
  --algo bc \
  --algo diffusion_policy \
  --no-download-datasets
```

By default this creates one run root per job under `runs/<task>_parallel/`, for example
`runs/square_parallel/D0_low_dim`. Each job writes its logs to `runs/<task>_parallel/logs/`
and its robomimic outputs under a shallow results root inside that job directory.
For diffusion runs, generated configs set `algo_name=diffusion_policy`, `train.seq_length=16`,
and min-max action normalization so actions are scaled into the range expected by robomimic's
diffusion policy implementation.

For an aligned diffusion BC sidecar that minimizes BC-to-RL eval gap, use the dedicated runner.
This trains diffusion BC under a named runtime profile and selects the best saved checkpoint
using the RL-stack evaluator instead of robomimic rollout selection:

```bash
bash scripts/mimicgen_train_bc_aligned.sh \
  --task coffee \
  --variant D0 \
  --modality low_dim \
  --diffusion-runtime-profile dppo_ddim5_ft5_act8 \
  --experiment-name coffee_d0_aligned_bc
```

This writes the best RL-aligned checkpoint under:

- `<run-root>/core_training_results/<experiment_name>_<runtime_profile>/<timestamp>/rl_aligned_eval/best_checkpoint.json`
- `<run-root>/core_training_results/<experiment_name>_<runtime_profile>/<timestamp>/rl_aligned_eval/best_checkpoint.pth`

## Analyze Logs

Export plots, tables, and flat CSVs from one run, several runs, or an entire logs directory:

```bash
./.venv/bin/python scripts/analyze_rl_logs.py logs/online_rl --output-dir analysis/online_rl
```

You can also point it at individual run folders:

```bash
./.venv/bin/python scripts/analyze_rl_logs.py \
  logs/online_rl/coffee_d1_v1 \
  logs/online_rl/coffee_d1_v2 \
  --output-dir analysis/coffee_compare
```

The script writes:

- `summary.csv` and `summary.md`
- one `{run_name}_metrics.csv` per run
- one `{run_name}_dashboard.png` per run
- comparison PNGs for eval success, eval return, and train success

## Evaluate A Saved Policy

Run deterministic evaluation from a saved checkpoint without resuming training:

```bash
./.venv/bin/python scripts/eval_online_rl.py \
  --config config/online_rl_coffee_smoke.json \
  --checkpoint logs/online_rl_coffee_safe_parallel_research/policy_best_update_0010_success_0.5104.pth \
  --output-dir analysis/eval/coffee_best \
  --episodes 20 \
  --eval-num-envs 8
```

This writes `eval_metrics.json` under the chosen output directory and prints the metrics to stdout.

To record a video from a saved checkpoint, set `--video-path`. Video capture forces single-env eval:

```bash
./.venv/bin/python scripts/eval_online_rl.py \
  --config config/online_rl_coffee_smoke.json \
  --checkpoint logs/online_rl_coffee_safe_parallel_research/policy_best_update_0010_success_0.5104.pth \
  --output-dir analysis/eval/coffee_best_video \
  --episodes 1 \
  --video-path videos/eval.mp4
```

This writes the video to `analysis/eval/coffee_best_video/videos/eval.mp4`.

To evaluate every saved BC epoch from a finished robomimic run and choose the best checkpoint
with the RL-stack evaluator, use:

```bash
MUJOCO_GL=glx \
UV_CACHE_DIR=.uv-cache \
uv run python scripts/eval_bc_checkpoints.py \
  --config config/online_rl_coffee_v11_dppo_profile_eval.json \
  --run-dir runs/aligned_bc_YYYYMMDD_HHMMSS/core_training_results/coffee_d0_aligned_bc_dppo_ddim5_ft5_act8/20260417123456 \
  --output-dir analysis/eval_bc/coffee_d0_aligned_bc \
  --episodes 20 \
  --eval-num-envs 4
```

The sweep writes:

- `checkpoint_metrics.jsonl`
- `checkpoint_metrics.csv`
- `best_checkpoint.json`
- `best_checkpoint.txt`
- `best_checkpoint.pth`
- `best_checkpoint_success_<score>.pth`

Notes:

- `--checkpoint` can point at a saved `policy_latest.*` or `policy_best_update_*.*` artifact.
- Both plain robomimic-style `.pth` checkpoints and residual-policy `.pt` artifacts are supported.
- Use the original training config as the base `--config`, then override only eval-specific settings on the command line.
- `--eval-num-envs` speeds up metric-only evaluation, but video export uses a single environment.
