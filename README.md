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

## Pipelines

Both pipelines log to Weights & Biases — authenticate once:

```bash
wandb login
```

### BC → DAPG (robomimic + RSL-RL)

```bash
uv run python scripts/bc_to_rl_pipeline.py --task coffee --variant D0
```

Trains BC-RNN, picks the best-SR checkpoint, then warm-starts DAPG fine-tuning.
Pass `--algo ppo` for PPO, or `--skip_bc --bc_output_dir <path>` to reuse an
existing BC run. Env counts and rollout sizes are baked in; override via
`--bc_extra` / `--rl_extra`.

### DPPO (diffusion policy)

```bash
bash scripts/run_dppo_bc_to_rl.sh --task stack_d0
```

Wraps prepare → pretrain → sweep → eval-bc → finetune. Task specs live in
`configs/mimicgen_tasks/`. Or run the stages manually:

```bash
uv run python scripts/run_official_dppo_mimicgen.py prepare  --task stack_d0
uv run python scripts/run_official_dppo_mimicgen.py pretrain --task stack_d0
uv run python scripts/run_official_dppo_mimicgen.py finetune --task stack_d0
```

`finetune` picks the best checkpoint from the latest pretrain run (async
checkpoint-eval writes `best_checkpoint.pt`).

### WSL video rendering

MuJoCo's default `glx`/`egl` backends need WSLg or an X server. On plain WSL2
the offscreen render context fails and videos come out corrupt. Install
OSMesa (software rasterizer, works everywhere) and point the task spec at it:

```bash
sudo apt-get update
sudo apt-get install -y libosmesa6 libosmesa6-dev
```

Then in `configs/mimicgen_tasks/<task>.yaml`:

```yaml
runtime:
  mujoco_gl: osmesa
  video_mujoco_gl: osmesa
```

Software rasterization is fine for env-0 video logging (~a few ms per 512²
frame); don't use it if you ever add visual-obs training. If you have WSLg +
a recent NVIDIA WSL driver, `egl` works too and is faster.

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

## DPPO

The official low-dim DPPO workflow now runs through the task-spec launcher:

- task specs live under `configs/mimicgen_tasks/`
- the launcher is `scripts/run_official_dppo_mimicgen.py`
- official `dppo/` remains the training runtime

### Fresh Machine Bootstrap

On a new machine, do this in order.

```bash
cd /path/to/rl-mimicgen
UV_CACHE_DIR=/path/to/rl-mimicgen/.uv-cache uv sync --locked
```

Then make sure the MimicGen HDF5 files exist at the paths referenced by the task specs.

Most low-dim tasks in this repo expect datasets under:

- `runs/datasets/core/`

If they are not already there, you will need to either:

- run the dataset download / generation flow for this repo, or
- move / copy the HDF5 files into `runs/datasets/core/` on the new machine

One way to populate those datasets is to use the existing MimicGen training bootstrap without `--no-download-datasets`, for example:

```bash
cd /path/to/rl-mimicgen
bash scripts/mimicgen_train_bc.sh \
  --task square \
  --variant D0 \
  --modality low_dim
```

That path will download / prepare the underlying MimicGen data if it is not already present. If you already have the `.hdf5` files from another machine, copying them into `runs/datasets/core/` is simpler.

Examples:

- `runs/datasets/core/coffee_d1.hdf5`
- `runs/datasets/core/square_d0.hdf5`
- `runs/datasets/core/stack_d0.hdf5`

Then prime any missing task specs with the current defaults:

```bash
cd /path/to/rl-mimicgen
.venv/bin/python scripts/prime_mimicgen_task_specs.py --dataset-type core
```

Then regenerate the derived DPPO artifacts/configs for every task spec whose dataset exists locally:

```bash
cd /path/to/rl-mimicgen
for spec in configs/mimicgen_tasks/*.yaml; do
  .venv/bin/python scripts/run_official_dppo_mimicgen.py prepare --task "$spec"
done
```

This `prepare` step is what re-generates the official DPPO-derived config files under `dppo/cfg/mimicgen/generated/`. On a fresh machine, do not skip it even if the task specs already exist.

This bootstrap does three things:

- install the shared Python environment
- create any missing central task specs with the current defaults
- regenerate the derived DPPO artifacts and generated config files for every task whose dataset file is present locally

### New Run Behavior

Run/state behavior:

- new launcher-driven runs are append-only and get unique `run_id`s
- each run keeps its own `run_config.yaml`, `task_spec_snapshot.yaml`, and `run_manifest.json`
- new runs never need to overwrite or mutate old runs
- if you want to chain from an old artifact, pass `--checkpoint` or `--run-dir` explicitly

Example of launching from an older run directory:

```bash
cd /home/nelly/projects/rl-mimicgen
.venv/bin/python scripts/run_official_dppo_mimicgen.py finetune \
  --task coffee_d1 \
  --run-dir /ABS/PATH/TO/OLD/SWEEP/OR/PRETRAIN/RUN
```

### Example Pipeline: `stack_d0`

Once `runs/datasets/core/stack_d0.hdf5` exists locally, the full pipeline is:

Prepare:

```bash
cd /home/nelly/projects/rl-mimicgen
.venv/bin/python scripts/run_official_dppo_mimicgen.py prepare --task stack_d0
```

Pretrain BC:

```bash
cd /home/nelly/projects/rl-mimicgen
.venv/bin/python scripts/run_official_dppo_mimicgen.py pretrain --task stack_d0
```

BC eval on the latest checkpoint:

```bash
cd /home/nelly/projects/rl-mimicgen
.venv/bin/python scripts/run_official_dppo_mimicgen.py eval-bc --task stack_d0
```

RL-init-aligned eval on the latest checkpoint:

```bash
cd /home/nelly/projects/rl-mimicgen
.venv/bin/python scripts/run_official_dppo_mimicgen.py eval-rl-init --task stack_d0
```

Fine-tune with DPPO from the latest BC checkpoint:

```bash
cd /home/nelly/projects/rl-mimicgen
.venv/bin/python scripts/run_official_dppo_mimicgen.py finetune --task stack_d0
```

Sweep saved BC checkpoints:

```bash
cd /home/nelly/projects/rl-mimicgen
.venv/bin/python scripts/run_official_dppo_mimicgen.py sweep --task stack_d0
```

If you want one thin wrapper that runs `prepare -> pretrain -> sweep -> eval-bc -> finetune` using the best swept BC checkpoint:

```bash
cd /home/nelly/projects/rl-mimicgen
bash scripts/run_dppo_bc_to_rl.sh --task stack_d0
```

If you want to pin a specific BC checkpoint for eval or finetune:

```bash
cd /home/nelly/projects/rl-mimicgen
.venv/bin/python scripts/run_official_dppo_mimicgen.py finetune \
  --task stack_d0 \
  --checkpoint /ABS/PATH/TO/state_1500.pt
```

Note:

- `stack_d0` is configured centrally in `configs/mimicgen_tasks/stack_d0.yaml`
- the launcher snapshots the resolved run config into each run directory
- `prepare --task stack_d0` requires `runs/datasets/core/stack_d0.hdf5` to exist locally
- the full workflow doc is in `docs/official_dppo_mimicgen_workflow.md`

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

Notes:

- `--checkpoint` can point at a saved `policy_latest.*` or `policy_best_update_*.*` artifact.
- Both plain robomimic-style `.pth` checkpoints and residual-policy `.pt` artifacts are supported.
- Use the original training config as the base `--config`, then override only eval-specific settings on the command line.
- `--eval-num-envs` speeds up metric-only evaluation, but video export uses a single environment.
