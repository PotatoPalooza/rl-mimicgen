# rl-mimicgen

Local setup and a minimal DAPG training flow for MimicGen and robomimic experiments.

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