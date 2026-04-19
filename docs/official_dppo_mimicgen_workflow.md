# Official DPPO on MimicGen Low-Dim

This document describes the current workflow for running MimicGen low-dim tasks through the official vendored `dppo/` implementation in this repo.

The important boundary is:

- `dppo/` is the training and evaluation implementation.
- `rl_mimicgen/` only adapts MimicGen datasets and generates official-DPPO-compatible configs and artifacts.
- `configs/mimicgen_tasks/*.yaml` is the canonical source of truth for how each scenario should run.

This document is for the low-dim path only.

## Overview

The modern workflow has four pieces:

1. A task spec in `configs/mimicgen_tasks/<task>.yaml`
2. Generated DPPO-ready dataset artifacts and configs
3. Immutable run directories with frozen snapshots
4. The official `dppo/script/run.py` runtime

The primary interface is:

- [scripts/run_official_dppo_mimicgen.py](/home/nelly/projects/rl-mimicgen/scripts/run_official_dppo_mimicgen.py:1)

The canonical task specs live here:

- [configs/mimicgen_tasks](/home/nelly/projects/rl-mimicgen/configs/mimicgen_tasks:1)

## How It Works

When you run a stage through the launcher, it does this:

1. Loads the canonical task spec.
2. Materializes or refreshes the MimicGen low-dim bundle for that task.
3. Regenerates the derived official-DPPO configs under `dppo/cfg/mimicgen/generated/<task>/`.
4. Merges task-spec stage overrides plus any CLI overrides.
5. Writes a frozen run snapshot into the final run directory.
6. Launches official `dppo` against that snapshotted config.

This means the editable source of truth is the task spec, but each run is still reproducible because it contains its own snapshot.

## Run Model

New runs are append-only.

- every launcher-driven run gets a unique `run_id`
- launching a new run creates a new directory; it does not overwrite an older run
- old runs keep their own config snapshot and artifact references
- if you want to chain from an old artifact, pass `--checkpoint` or `--run-dir` explicitly

There is no separate mutable task-level control plane in the launcher. The run directory itself is the durable source of truth.

## Source Of Truth

Each task spec is the place to define persistent per-environment defaults.

Example shape:

```yaml
dataset_id: square_d0

dataset:
  path: runs/datasets/core/square_d0.hdf5

materialize:
  output_root: runs/official_dppo_mimicgen
  val_split: 0.0
  split_seed: 0

logging:
  root: logs/official_dppo/mimicgen

runtime:
  mujoco_gl: glx
  video_mujoco_gl: osmesa

wandb:
  entity: null

pretrain:
  config: {}

finetune:
  config:
    train:
      n_train_itr: 201
    env:
      n_envs: 50

eval_bc:
  config:
    n_episodes: 20
    env:
      n_envs: 20

eval_rl_init:
  config:
    n_episodes: 20
    env:
      n_envs: 20

sweep:
  eval_mode: bc
  device: cpu
  n_envs: 20
  n_episodes: 20
  n_steps: 400
  max_episode_steps: 400
  video_checkpoints: none
```

## Override Model

There are three override layers:

1. Generated base config
2. Task-spec stage override
3. CLI `--set` override

Precedence goes in that order, so `--set` wins over the task spec, and the task spec wins over the generated defaults.

Examples:

Persistent per-task override in the task spec:

```yaml
finetune:
  config:
    train:
      n_train_itr: 400
      save_model_freq: 25
    env:
      n_envs: 32
    ft_denoising_steps: 8
```

One-off launch override:

```bash
.venv/bin/python scripts/run_official_dppo_mimicgen.py pretrain \
  --task square_d0 \
  --set wandb=null \
  --set device=cpu \
  --set train.n_epochs=1
```

## Run Snapshots

Every launcher-driven run writes a frozen copy of what it actually used.

Run directories now include:

- `generated_config_snapshot.yaml`
- `run_config.yaml`
- `task_spec_snapshot.yaml`
- `run_manifest.json`

They also get a unique `run_id` in the directory name and a run-local `wandb/` directory when `wandb` is enabled.

That gives you:

- one editable canonical task definition
- immutable historical runs
- a clear audit trail when defaults change over time

If you want to launch a new stage from an old run, point at it directly:

```bash
cd /home/nelly/projects/rl-mimicgen
.venv/bin/python scripts/run_official_dppo_mimicgen.py finetune \
  --task coffee_d1 \
  --run-dir /ABS/PATH/TO/OLD/SWEEP/OR/PRETRAIN/RUN
```

`--run-dir` resolves `best_checkpoint.pt` first, then falls back to the latest `checkpoint/state_*.pt` in that run.

## Generated Artifacts

For a task like `coffee_d0`, materialization writes:

- `runs/official_dppo_mimicgen/data/coffee_d0/train.npz`
- `runs/official_dppo_mimicgen/data/coffee_d0/val.npz`
- `runs/official_dppo_mimicgen/data/coffee_d0/normalization.npz`
- `runs/official_dppo_mimicgen/data/coffee_d0/env_meta.json`
- `runs/official_dppo_mimicgen/data/coffee_d0/dataset_meta.json`

It also generates:

- `dppo/cfg/mimicgen/generated/coffee_d0/pre_diffusion_mlp.yaml`
- `dppo/cfg/mimicgen/generated/coffee_d0/ft_ppo_diffusion_mlp.yaml`
- `dppo/cfg/mimicgen/generated/coffee_d0/eval_bc_diffusion_mlp.yaml`
- `dppo/cfg/mimicgen/generated/coffee_d0/eval_rl_init_diffusion_mlp.yaml`
- `dppo/cfg/mimicgen/generated/coffee_d0/eval_diffusion_mlp.yaml`
- `dppo/cfg/mimicgen/generated/coffee_d0/task_manifest.json`

`eval_diffusion_mlp.yaml` is kept as a compatibility alias for BC-style eval.

## Evaluation Modes

Two eval surfaces are generated on purpose.

`eval_bc_diffusion_mlp`
- evaluates the checkpoint as a BC policy
- uses the BC-style `DiffusionEval` path
- targets a fixed number of completed episodes instead of relying only on a rollout-step budget
- use this to rank saved pretraining checkpoints as BC models

`eval_rl_init_diffusion_mlp`
- evaluates the checkpoint in a way that matches RL initialization more closely
- uses the PPO diffusion model family and finetune-style denoising settings
- also targets a fixed number of completed episodes
- use this when you want a number that should line up with RL iteration 0

The default generated low-dim eval config now uses `n_episodes: 20`.

That is the right default for fast comparison against robomimic-style evaluation because:

- many robomimic-style logs in this repo report `evaluated_episodes=20`
- `n_episodes` is the apples-to-apples quantity; `n_envs` is only parallelism
- `n_steps` remains in the config as a safety cap, but eval stops early once the target episode count is reached

## Example: `coffee_d0`

This is the end-to-end path for one task.

### 1. Prepare

```bash
cd /home/nelly/projects/rl-mimicgen
.venv/bin/python scripts/run_official_dppo_mimicgen.py prepare --task coffee_d0
```

This reads [coffee_d0.yaml](/home/nelly/projects/rl-mimicgen/configs/mimicgen_tasks/coffee_d0.yaml:1), materializes the dataset bundle, and refreshes the generated DPPO configs.

### 2. Pretrain

```bash
cd /home/nelly/projects/rl-mimicgen
.venv/bin/python scripts/run_official_dppo_mimicgen.py pretrain --task coffee_d0
```

This runs the official DPPO demo pretraining loop and writes checkpoints under:

- `logs/official_dppo/mimicgen/pretrain/coffee_d0/.../checkpoint/`

### 3. BC Eval

```bash
cd /home/nelly/projects/rl-mimicgen
.venv/bin/python scripts/run_official_dppo_mimicgen.py eval-bc \
  --task coffee_d0 \
  --checkpoint /ABS/PATH/TO/CHECKPOINT/state_1500.pt
```

### 4. RL-Init Eval

```bash
cd /home/nelly/projects/rl-mimicgen
.venv/bin/python scripts/run_official_dppo_mimicgen.py eval-rl-init \
  --task coffee_d0 \
  --checkpoint /ABS/PATH/TO/CHECKPOINT/state_1500.pt
```

### 5. Fine-Tune Online

```bash
cd /home/nelly/projects/rl-mimicgen
.venv/bin/python scripts/run_official_dppo_mimicgen.py finetune \
  --task coffee_d0 \
  --checkpoint /ABS/PATH/TO/CHECKPOINT/state_1500.pt
```

If `--checkpoint` is omitted for `finetune`, `eval-bc`, or `eval-rl-init`, the launcher uses the latest pretrain checkpoint for that task.

### 6. Sweep Saved Checkpoints

```bash
cd /home/nelly/projects/rl-mimicgen
.venv/bin/python scripts/run_official_dppo_mimicgen.py sweep --task coffee_d0
```

Sweep defaults come from the task spec under `sweep:`.

## Optional Direct Interfaces

The lower-level tools still exist:

- [scripts/prepare_mimicgen_dppo_lowdim.py](/home/nelly/projects/rl-mimicgen/scripts/prepare_mimicgen_dppo_lowdim.py:1)
- [dppo/script/run.py](/home/nelly/projects/rl-mimicgen/dppo/script/run.py:1)
- [dppo/script/eval_checkpoint_sweep.py](/home/nelly/projects/rl-mimicgen/dppo/script/eval_checkpoint_sweep.py:1)

Use those when you need direct debugging or want to bypass the task-spec layer. For normal operation, prefer the launcher.

## Runtime Notes

Environment setup:

```bash
cd /home/nelly/projects/rl-mimicgen
UV_CACHE_DIR=/home/nelly/projects/rl-mimicgen/.uv-cache uv sync --locked
```

WSL video rendering:

- for normal low-dim training and eval, `glx` is fine
- for checkpoint-video sweeps on WSL, `osmesa` is the reliable path
- task specs can define both with:
  - `runtime.mujoco_gl`
  - `runtime.video_mujoco_gl`

## What To Edit vs What Not To Edit

Edit:

- `configs/mimicgen_tasks/*.yaml`

Do not treat these as the primary source of truth:

- `dppo/cfg/mimicgen/generated/*`

Those generated configs are derived artifacts. They are regenerated from the adapter and task spec workflow.

## Related Files

- Launcher: [scripts/run_official_dppo_mimicgen.py](/home/nelly/projects/rl-mimicgen/scripts/run_official_dppo_mimicgen.py:1)
- Task specs: [configs/mimicgen_tasks](/home/nelly/projects/rl-mimicgen/configs/mimicgen_tasks:1)
- Adapter: [rl_mimicgen/adapters/dppo_lowdim.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/adapters/dppo_lowdim.py:1)
- Lower-level adapter CLI: [scripts/prepare_mimicgen_dppo_lowdim.py](/home/nelly/projects/rl-mimicgen/scripts/prepare_mimicgen_dppo_lowdim.py:1)
- Command scratchpad: [docs/cli_scratch.md](/home/nelly/projects/rl-mimicgen/docs/cli_scratch.md:1)
