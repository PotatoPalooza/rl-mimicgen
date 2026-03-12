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

## Train

Run the MimicGen one-task BC runner:

```bash
uv run python -m rl_mimicgen.mimicgen.paper_bc_one_task \
  --task square \
  --variant D0 \
  --modality low_dim
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

By default this creates one run root per job under `runs/<task>_parallel/`, for example
`runs/square_parallel/D0_low_dim`. Each job writes its logs to `runs/<task>_parallel/logs/`
and its robomimic outputs under a shallow results root inside that job directory.
