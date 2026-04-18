# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A research project that fine-tunes robomimic BC checkpoints (trained on MimicGen demonstrations) with on-policy RL — either standard PPO or DPPO (Diffusion Policy PPO). The pipeline is: BC pretrain → load checkpoint → PPO online fine-tuning in a live robosuite simulation.

## Environment Setup

- **Python 3.11**, package manager: `uv`, local repo manager: `gitman`
- Local editable checkouts of `robosuite`, `robomimic`, `mimicgen`, `robosuite-task-zoo` are managed by `gitman`

```bash
pip install uv gitman
gitman install    # clones local dependency repos
uv sync           # installs everything including editable local repos
```

Or: `just bootstrap`

Verify: `MUJOCO_GL=glx uv run python scripts/verify_blackwell_env.py` (or `just verify`)

## Commands

**Run all tests:**
```bash
uv run pytest tests/
```

**Run a single test file:**
```bash
uv run pytest tests/test_online_rl_config.py
```

**Run a single test by name:**
```bash
uv run pytest tests/test_online_rl_config.py::test_online_rl_config_round_trip
```

**Lint / format:**
```bash
uv run ruff check rl_mimicgen/
uv run ruff format rl_mimicgen/
```

**BC pretraining (parallel jobs):**
```bash
bash scripts/mimicgen_train_bc.sh --task coffee --max-parallel 2
```

**Online RL fine-tuning:**
```bash
uv run python scripts/train_online_rl.py --config config/online_rl_coffee_smoke.json
```

**Evaluate a checkpoint:**
```bash
uv run python scripts/eval_online_rl.py --checkpoint <path.pth> --task coffee
```

**Analyze training logs:**
```bash
uv run python scripts/analyze_rl_logs.py --run-dir logs/online_rl/<run_name>
```

## Architecture

### Core RL Engine (`rl_mimicgen/rl/`)

The online RL system has two parallel tracks — standard BC policies and diffusion policies:

```
BC Checkpoint (.pth)
    ├──(standard)──► OnlinePolicyAdapter  ──► DemoAugmentedPPO  ──► RolloutStorage
    └──(diffusion)──► DiffusionOnlinePolicyAdapter ──► DiffusionPPO ──► DiffusionRolloutStorage
                                                   └── both ──►
                                                   OnlineRLTrainer
                                                   ParallelRobomimicVectorEnv / SerialRobomimicVectorEnv
```

**Key files:**
| File | Role |
|---|---|
| `config.py` | All training config as dataclasses (`OnlineRLConfig`, `PPOConfig`, `DiffusionConfig`, etc.); serialized to/from JSON |
| `policy.py` | `PolicyBundle` (loads checkpoint) + `OnlinePolicyAdapter` (wraps BC actor as PPO actor+critic; handles RNN state, GMM/Gaussian heads, residual mode) |
| `diffusion_policy.py` | `DiffusionOnlinePolicyAdapter` — manages DDPM/DDIM schedulers, EMA weights, action chain sampling, obs history queue |
| `ppo.py` | `DemoAugmentedPPO` — PPO with optional decaying demo BC regularization loss; RNN-aware minibatch slicing |
| `dppo.py` | `DiffusionPPO` — PPO over decision-level denoising chains (DPPO algorithm) |
| `storage.py` | `RolloutStorage` — `(T, N, ...)` tensors + GAE computation |
| `diffusion_storage.py` | `DiffusionRolloutStorage` — decision-level denoising chain batches for DPPO |
| `robomimic_env.py` | `ParallelRobomimicVectorEnv` / `SerialRobomimicVectorEnv` — gymnasium-compatible vector env wrappers around robosuite; uses `cloudpickle` for subprocess spawning |
| `trainer.py` | `OnlineRLTrainer` — top-level loop: collect rollouts → GAE → PPO update → periodic eval → checkpoint best |

**Training loop** (`OnlineRLTrainer.train()`):
1. Collect `rollout_steps` steps across `num_envs` parallel envs
2. Compute GAE returns and advantages
3. Run `update_epochs` PPO epochs with `num_minibatches` minibatches
4. Optionally mix in demo BC loss (decays by `demo.decay` each update)
5. Evaluate every `evaluation.every_n_updates` updates; save best checkpoint by success rate

### BC Pipeline (`rl_mimicgen/mimicgen/`)

`paper_bc_one_task.py` orchestrates: download dataset from HuggingFace → generate robomimic config → launch training via `train_robomimic.py` (which calls `robomimic/scripts/train.py` via `runpy`).

### Diffusion Runtime Profiles (`rl_mimicgen/diffusion_runtime.py`)

Named presets that configure DDPM/DDIM inference (scheduler, num_steps, EMA, act_steps, gamma). Applied as JSON mutations to BC training configs or as dataclass mutations to `OnlineRLConfig`. Key profiles:
- `robomimic_ddpm100_ema` — standard 100-step DDPM
- `dppo_ddim5_ft5_act8` — DPPO paper setting (5-step DDIM, 5 fine-tuning denoising steps, 8 action steps)

### Config Files (`config/`)

JSON files matching the `OnlineRLConfig` schema. Smoke configs (`*_smoke.json`) use minimal settings (2 envs, 8 rollout steps, 2 updates) for fast end-to-end iteration.

### Checkpoint Formats

- **Standard**: robomimic `.pth` (algo state dict embedded in `ckpt_dict`)
- **Residual**: `.pt` dict with `mode=residual`, frozen base ckpt, residual actor state dict, log_std, scale
- **Trainer state**: `trainer_*.pt` with optimizer state, demo_coef, PPO step counts

### Output Layout

```
logs/online_rl/<run_name>/
  ├── config.json
  ├── metrics.jsonl          # one JSON row per update
  ├── policy_latest.pth
  ├── policy_best_update_NNNN_success_X.XXXX.pth
  └── trainer_latest.pt
analysis/<run_name>/
  ├── summary.csv / summary.md
  └── <run>_dashboard.png
```

## Local Dependencies (managed by gitman)

All installed as editable packages into `.venv`:
- `robosuite` — MuJoCo-based manipulation simulator
- `robosuite-task-zoo` — additional task environments  
- `robomimic` — BC/diffusion policy training library + dataset loading
- `mimicgen` — demonstration generation + HuggingFace dataset registry
