# Official DPPO on MimicGen Low-Dim: General Workflow

This document describes the intended workflow for running the official `dppo/` training loop on MimicGen low-dimensional datasets in this repository.

The key design choice is simple:

- `dppo/` remains the authoritative implementation for diffusion pretraining, PPO fine-tuning, and evaluation.
- `rl_mimicgen/` only provides adapters that convert MimicGen HDF5 demos into the file formats and config structure that official `dppo/` already expects.

This is the low-dim path. Image-based MimicGen tasks are not covered here yet.

## What the Pipeline Looks Like

For a MimicGen task such as `coffee_d0`, the workflow has three main stages:

1. Materialize a DPPO-compatible dataset bundle from the MimicGen HDF5 file.
2. Pre-train a diffusion policy on the demo dataset.
3. Fine-tune that diffusion policy online with PPO in the robomimic / robosuite environment.

Evaluation is a separate fourth step that loads a checkpoint and runs rollout-only measurement.

In other words, yes: there is a separate demo pretraining stage and a separate online diffusion PPO stage.

## One-Time Environment Setup

Use the shared project environment:

```bash
cd /home/nelly/projects/rl-mimicgen
UV_CACHE_DIR=/home/nelly/projects/rl-mimicgen/.uv-cache uv sync --locked
```

The central environment now includes the official `dppo` runtime dependencies needed for this path, including `hydra-core`, `omegaconf`, `gym`, `gdown`, `wandb`, `einops`, `imageio`, and `tqdm`.

## Step 1: Materialize a MimicGen Task for Official DPPO

The adapter CLI:

- reads a MimicGen HDF5 file
- infers the low-dim observation keys directly from the file
- writes official-DPPO-ready artifacts
- generates task-specific Hydra configs under `dppo/cfg/mimicgen/generated/<dataset_id>/`

The command is:

```bash
cd /home/nelly/projects/rl-mimicgen
MUJOCO_GL=glx .venv/bin/python scripts/prepare_mimicgen_dppo_lowdim.py \
  --dataset runs/datasets/core/coffee_d0.hdf5 \
  --output-root runs/official_dppo_mimicgen \
  --verify-env-reset
```

For a different low-dim MimicGen task, replace only the `--dataset` path.

### Artifacts Produced

For dataset id `coffee_d0`, the adapter writes:

- `runs/official_dppo_mimicgen/data/coffee_d0/train.npz`
- `runs/official_dppo_mimicgen/data/coffee_d0/val.npz`
- `runs/official_dppo_mimicgen/data/coffee_d0/normalization.npz`
- `runs/official_dppo_mimicgen/data/coffee_d0/env_meta.json`
- `runs/official_dppo_mimicgen/data/coffee_d0/dataset_meta.json`

It also generates:

- `dppo/cfg/mimicgen/generated/coffee_d0/pre_diffusion_mlp.yaml`
- `dppo/cfg/mimicgen/generated/coffee_d0/ft_ppo_diffusion_mlp.yaml`
- `dppo/cfg/mimicgen/generated/coffee_d0/eval_diffusion_mlp.yaml`
- `dppo/cfg/mimicgen/generated/coffee_d0/eval_bc_diffusion_mlp.yaml`
- `dppo/cfg/mimicgen/generated/coffee_d0/eval_rl_init_diffusion_mlp.yaml`

### What the Adapter Decides Automatically

The low-dim adapter is metadata-driven.

It infers:

- `dataset_id`
- `env_name`
- low-dim observation key order
- observation dimension
- action dimension
- episode horizon

That is important because MimicGen low-dim observation schemas vary by task, so hard-coded observation lists do not scale.

## Step 2: Pre-Train the Diffusion Policy on Demos

This stage uses the official pretraining loop in `dppo/`. It is offline imitation-style training over the demo dataset and does not interact with the environment.

Run:

```bash
cd /home/nelly/projects/rl-mimicgen/dppo
MUJOCO_GL=glx ../.venv/bin/python script/run.py \
  --config-dir=cfg/mimicgen/generated/coffee_d0 \
  --config-name=pre_diffusion_mlp
```

This trains a diffusion policy on the normalized `train.npz` demo bundle and writes checkpoints under:

- `logs/official_dppo/mimicgen/pretrain/coffee_d0/.../checkpoint/`

### Optional: Sweep Saved BC Checkpoints During Pretraining

The generated pretrain config now includes an optional `train.checkpoint_eval` block.

If enabled, every checkpoint save triggers an external sweep script that:

- evaluates saved pretraining checkpoints with the official eval config
- writes per-checkpoint metrics
- selects the current best checkpoint
- copies the best checkpoint to a stable path
- optionally attempts to render a video for the best checkpoint

Example:

```bash
cd /home/nelly/projects/rl-mimicgen/dppo
MUJOCO_GL=glx ../.venv/bin/python script/run.py \
  --config-dir=cfg/mimicgen/generated/coffee_d0 \
  --config-name=pre_diffusion_mlp \
  wandb=null \
  train.checkpoint_eval.enabled=True \
  train.checkpoint_eval.n_envs=5 \
  train.checkpoint_eval.n_steps=400 \
  train.checkpoint_eval.max_episode_steps=400
```

The external sweep script is:

- [dppo/script/eval_checkpoint_sweep.py](/home/nelly/projects/rl-mimicgen/dppo/script/eval_checkpoint_sweep.py:1)

In headless environments, the metrics sweep and best-checkpoint copy work even if optional video rendering fails.

### WSL Video Rendering Note

For this repo on WSL, the most reliable video backend is currently `osmesa`.

The important bug was that the official robomimic env factory was forcing `MUJOCO_GL=egl` whenever offscreen rendering was enabled. That has been fixed so an explicitly provided backend is respected.

If you are sweeping checkpoints and want videos written to disk on WSL, use:

```bash
cd /home/nelly/projects/rl-mimicgen
env -u PYOPENGL_PLATFORM MUJOCO_GL=osmesa .venv/bin/python dppo/script/eval_checkpoint_sweep.py \
  --config-dir /home/nelly/projects/rl-mimicgen/dppo/cfg/mimicgen/generated/coffee_d0 \
  --config-name eval_diffusion_mlp \
  --checkpoint-dir /home/nelly/projects/rl-mimicgen/logs/official_dppo/mimicgen/pretrain/coffee_d0/coffee_d0_pre_diffusion_mlp_ta4_td20/2026-04-18_16-14-40_42/checkpoint \
  --output-dir /home/nelly/projects/rl-mimicgen/runs/official_dppo_mimicgen/eval_sweep_debug/coffee_d0_all_video_osmesa \
  --device cpu \
  --n-envs 1 \
  --n-steps 100 \
  --max-episode-steps 100 \
  --every-n 1 \
  --video-checkpoints all \
  --render-num 1
```

This command has been verified to write:

- `state_1/render/eval_trial-0.mp4`

Do not force `PYOPENGL_PLATFORM=glx` for this path. Under WSL that was conflicting with the EGL import path and is not needed for the working `osmesa` setup.

## Step 3: Fine-Tune Online with DPPO

This is the actual online RL stage.

It loads the pretrained diffusion actor, creates the robomimic environment from `env_meta.json`, runs rollouts, and applies PPO updates to the diffusion policy.

You need a pretrained checkpoint first. To find the newest one:

```bash
find /home/nelly/projects/rl-mimicgen/logs/official_dppo/mimicgen/pretrain/coffee_d0 \
  -path '*/checkpoint/state_*.pt' | sort | tail -1
```

Then launch online PPO fine-tuning:

```bash
cd /home/nelly/projects/rl-mimicgen/dppo
MUJOCO_GL=glx ../.venv/bin/python script/run.py \
  --config-dir=cfg/mimicgen/generated/coffee_d0 \
  --config-name=ft_ppo_diffusion_mlp \
  base_policy_path=/ABS/PATH/TO/PRETRAIN/CHECKPOINT/state_3000.pt
```

This stage writes logs and checkpoints under:

- `logs/official_dppo/mimicgen/finetune/coffee_d0/...`

### What Is Different From Pretraining

Pretraining:

- uses demos only
- reads `train.npz`
- optimizes diffusion reconstruction / denoising loss
- does not step the environment

Fine-tuning:

- loads a pretrained diffusion checkpoint
- steps the actual environment
- collects rewards and success signals online
- updates the diffusion actor and critic using PPO

### Suggested MimicGen Fine-Tune Configs

If you want to stay close to official DPPO defaults, the safest way to think about MimicGen low-dim tasks is:

- light robomimic-style tasks
  - examples: `lift`, `can`, and MimicGen tasks with relatively small observation spaces and shorter horizons
- heavy robomimic-style tasks
  - examples: `transport`, and MimicGen tasks with larger observation spaces, longer horizons, or more expensive environment dynamics

The generated MimicGen configs already follow the official DPPO PPO-diffusion shape fairly closely:

- keep `denoising_steps: 20`
- keep `ft_denoising_steps: 10`
- keep `gamma: 0.999`
- keep `gae_lambda: 0.95`
- keep `actor_lr: 1e-4`
- keep `critic_lr: 1e-3`
- keep `target_kl: 1`
- keep `gamma_denoising: 0.99`

Those are the highest-value defaults to preserve if your goal is paper faithfulness.

#### Light Robomimic-Style Tasks

Use this profile for tasks that look more like official `lift` / `can` than `transport`.

Suggested settings:

```yaml
train:
  n_train_itr: 81-151
  n_steps: 300
  batch_size: 7500
  update_epochs: 10
  actor_lr_scheduler:
    warmup_steps: 10
  critic_lr_scheduler:
    warmup_steps: 10
env:
  n_envs: 50
model:
  ft_denoising_steps: 10
```

This mirrors the official low-dim robomimic PPO diffusion configs under:

- [dppo/cfg/robomimic/finetune/lift/ft_ppo_diffusion_mlp.yaml](/home/nelly/projects/rl-mimicgen/dppo/cfg/robomimic/finetune/lift/ft_ppo_diffusion_mlp.yaml)
- [dppo/cfg/robomimic/finetune/can/ft_ppo_diffusion_mlp.yaml](/home/nelly/projects/rl-mimicgen/dppo/cfg/robomimic/finetune/can/ft_ppo_diffusion_mlp.yaml)

#### Heavy Robomimic-Style Tasks

Use this profile for tasks that look more like official `transport`, or for MimicGen tasks whose low-dim observation size and rollout cost are materially larger.

Suggested settings:

```yaml
train:
  n_train_itr: 201
  n_steps: 400
  batch_size: 10000
  update_epochs: 5
  actor_lr_scheduler:
    warmup_steps: 10
  critic_lr_scheduler:
    warmup_steps: 10
env:
  n_envs: 50
model:
  ft_denoising_steps: 10
```

This mirrors the official heavier robomimic PPO diffusion config:

- [dppo/cfg/robomimic/finetune/transport/ft_ppo_diffusion_mlp.yaml](/home/nelly/projects/rl-mimicgen/dppo/cfg/robomimic/finetune/transport/ft_ppo_diffusion_mlp.yaml)

#### How to Reduce Compute While Staying Close to DPPO

If you want high probability of success while reducing wall-clock time, change the following in this order:

1. `train.update_epochs`
2. `env.n_envs` together with `train.n_steps`
3. scheduler `warmup_steps`

What to change:

- Reduce `train.update_epochs` first.
  - This is the cleanest speed knob.
  - For heavy tasks, `5` is already an official DPPO setting.
  - For light tasks, `10` is still official, but dropping to `5` is usually the least invasive compute reduction.
- If CPU rollout is the bottleneck, reduce `env.n_envs` and increase `train.n_steps` so total rollout volume stays similar.
  - Official DPPO explicitly recommends this tradeoff when fewer CPU threads are available.
  - Try to keep `n_envs * n_steps * act_steps` in the same ballpark.
- Use scheduler `warmup_steps: 10` rather than `0`.
  - This is closer to the official robomimic configs and lowers optimization risk during early finetuning.

What not to change first if paper faithfulness matters:

- do not change `denoising_steps`
- do not change `ft_denoising_steps` first
- do not change `target_kl`
- do not change `gamma_denoising`
- do not change `actor_lr` / `critic_lr` first

In practice, the most paper-faithful compute-reduced variant is usually:

```yaml
env:
  n_envs: 24-32
train:
  n_steps: adjusted upward so total rollout steps stay similar
  update_epochs: 5
  actor_lr_scheduler:
    warmup_steps: 10
  critic_lr_scheduler:
    warmup_steps: 10
```

That keeps the diffusion setup and PPO objective intact while cutting a substantial amount of wall-clock time on a single workstation.

#### BC / RL Alignment Note

The generator intentionally keeps the heavier diffusion MLP architecture unchanged by default even when PPO-side defaults are made cheaper.

That is deliberate for full BC / RL alignment:

- pretraining BC config
- PPO fine-tuning config
- RL-init evaluation config

should all agree on the architecture-critical diffusion settings:

- `denoising_steps`
- `horizon_steps`
- `cond_steps`
- actor network shape such as `time_dim`, `mlp_dims`, and conditional projection layout

For the generated MimicGen low-dim path, this means the default actor architecture remains aligned across BC and RL, while the recommended compute reduction only touches PPO-side training knobs such as:

- `update_epochs`
- scheduler `warmup_steps`

This is the safest way to reduce wall-clock time without introducing an avoidable mismatch between the BC checkpoint family and the RL fine-tuning / evaluation stack.

## Step 4: Evaluate a Checkpoint

Evaluation is a separate run that loads a checkpoint and measures rollout performance without doing training updates.

The adapter now generates two evaluation surfaces on purpose:

- `eval_bc_diffusion_mlp`
  - pure BC checkpoint evaluation
  - uses the BC-style `DiffusionEval` path
  - `eval_diffusion_mlp` is kept as a compatibility alias to this config
- `eval_rl_init_diffusion_mlp`
  - matches the RL initialization settings as closely as possible
  - uses the same PPO diffusion model family and `ft_denoising_steps` default as fine-tuning

Use the BC config when ranking saved pretraining checkpoints as BC policies. Use the RL-init config when you want a number that should line up with RL iteration 0.

Run:

```bash
cd /home/nelly/projects/rl-mimicgen/dppo
MUJOCO_GL=glx ../.venv/bin/python script/run.py \
  --config-dir=cfg/mimicgen/generated/coffee_d0 \
  --config-name=eval_diffusion_mlp \
  base_policy_path=/ABS/PATH/TO/CHECKPOINT/state_3000.pt
```

This writes outputs under:

- `logs/official_dppo/mimicgen/eval/coffee_d0/...`

For RL-init-aligned evaluation, run:

```bash
cd /home/nelly/projects/rl-mimicgen/dppo
MUJOCO_GL=glx ../.venv/bin/python script/run.py \
  --config-dir=cfg/mimicgen/generated/coffee_d0 \
  --config-name=eval_rl_init_diffusion_mlp \
  base_policy_path=/ABS/PATH/TO/CHECKPOINT/state_3000.pt
```

## How to Use This for Another MimicGen Low-Dim Task

For another task such as `square_d0`, `coffee_d1`, or `threading_d0`, the pattern is the same:

1. Run `scripts/prepare_mimicgen_dppo_lowdim.py` on that task’s HDF5 file.
2. Use the generated config directory under `dppo/cfg/mimicgen/generated/<dataset_id>/`.
3. Run `pre_diffusion_mlp`.
4. Run `ft_ppo_diffusion_mlp` with the pretrained checkpoint path.
5. Optionally run `eval_bc_diffusion_mlp` or `eval_rl_init_diffusion_mlp` depending on which comparison you need.

The adapter is what makes this scale. You should not hand-author observation keys, dimensions, or environment metadata for each task.

## Important Notes

- The current documented path is low-dim only.
- The official `dppo` implementation remains the training authority.
- The MimicGen adapter layer should stay thin and restricted to dataset conversion plus config generation.
- The generated fine-tune and eval configs use `success_info_key: success`, which allows official DPPO evaluation logic to consume robomimic task success directly.

## Related Files

- Adapter implementation: [rl_mimicgen/adapters/dppo_lowdim.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/adapters/dppo_lowdim.py:1)
- Adapter CLI: [scripts/prepare_mimicgen_dppo_lowdim.py](/home/nelly/projects/rl-mimicgen/scripts/prepare_mimicgen_dppo_lowdim.py:1)
- Example command scratchpad: [docs/cli_scratch.md](/home/nelly/projects/rl-mimicgen/docs/cli_scratch.md:1)
- Generated `coffee_d0` pretrain config: [dppo/cfg/mimicgen/generated/coffee_d0/pre_diffusion_mlp.yaml](/home/nelly/projects/rl-mimicgen/dppo/cfg/mimicgen/generated/coffee_d0/pre_diffusion_mlp.yaml:1)
- Generated `coffee_d0` fine-tune config: [dppo/cfg/mimicgen/generated/coffee_d0/ft_ppo_diffusion_mlp.yaml](/home/nelly/projects/rl-mimicgen/dppo/cfg/mimicgen/generated/coffee_d0/ft_ppo_diffusion_mlp.yaml:1)
