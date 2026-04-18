# Repository Guidelines

## Project Structure & Module Organization

- `rl_mimicgen/`: main Python package. The online RL stack lives in `rl_mimicgen/rl/` (`trainer.py`, `policy.py`, `ppo.py`, `awac.py`, `storage.py`, `config.py`).
- `scripts/`: entry points for training, evaluation, environment verification, and analysis.
- `config/`: JSON configs for PPO, DPPO, and AWAC experiments.
- `tests/`: unit tests, named `test_*.py`.
- `docs/`: notes, papers, and technical reports.
- `mimicgen/`, `robomimic/`, `robosuite/`, `robosuite-task-zoo/`: local editable dependencies managed by `gitman`.
- `logs/` and `runs/`: generated outputs; do not treat these as source.

## Build, Test, and Development Commands

- `just bootstrap` or `gitman install && uv sync`: install local deps and create the `.venv`.
- `MUJOCO_GL=glx uv run python scripts/verify_blackwell_env.py`: verify the simulation environment.
- `uv run pytest tests/`: run the full test suite.
- `uv run pytest tests/test_online_rl_config.py::test_online_rl_config_round_trip`: run one test.
- `uv run ruff check rl_mimicgen/ tests/`: lint.
- `uv run ruff format rl_mimicgen/ tests/`: format code.
- `uv run python scripts/train_online_rl.py --config config/online_rl_coffee_smoke.json`: smoke-test online RL.

## Coding Style & Naming Conventions

- Follow Python 3.11 style with 4-space indentation and type hints for new public functions.
- Prefer small, explicit dataclasses for config and batch types.
- Module and file names use `snake_case`; classes use `PascalCase`; functions, variables, and config keys use `snake_case`.
- Keep algorithm-specific paths separate when semantics differ materially, instead of overloading shared PPO abstractions.

## Testing Guidelines

- Use `pytest`.
- Add or update tests for every algorithm or config change, especially under `tests/test_*rl*`.
- Favor focused unit tests over large end-to-end runs.
- For research code, validate both correctness and instrumentation: shapes, metrics, and checkpoint round-trips.

## Commit & Pull Request Guidelines

- Recent history uses short, imperative commit subjects, e.g. `diffusion alignment`, `action steps step 8`.
- Keep commits scoped to one change.
- PRs should include:
  - a concise problem statement and change summary
  - affected configs or scripts
  - exact validation commands run
  - key metrics or screenshots for training/eval changes

## Configuration & Experiment Tips

- Prefer editing JSON under `config/` over hard-coding values in scripts.
- Keep paper PDFs and design notes in `docs/`.
- Use fresh `output_dir` values for experiments so logs are comparable and reproducible.
