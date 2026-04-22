bootstrap:
    gitman install
    uv sync
    uv run python scripts/apply_vendor_patches.py

bootstrap-task-zoo:
    uv pip install -e ./robosuite-task-zoo --no-deps
    -uv pip uninstall mujoco-py

lock:
    uv lock

sync:
    uv sync
    uv run python scripts/apply_vendor_patches.py

sync-task-zoo:
    uv pip install -e ./robosuite-task-zoo --no-deps
    -uv pip uninstall mujoco-py

verify:
    MUJOCO_GL=glx uv run python scripts/verify_blackwell_env.py
