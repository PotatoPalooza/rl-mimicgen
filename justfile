bootstrap:
    gitman install
    uv sync

bootstrap-task-zoo:
    uv pip install -e ./robosuite-task-zoo --no-deps
    -uv pip uninstall mujoco-py

lock:
    uv lock

sync:
    uv sync

sync-task-zoo:
    uv pip install -e ./robosuite-task-zoo --no-deps
    -uv pip uninstall mujoco-py

verify:
    MUJOCO_GL=glx uv run python scripts/verify_blackwell_env.py
