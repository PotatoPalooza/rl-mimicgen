bootstrap:
    gitman install
    uv sync

lock:
    uv lock

sync:
    uv sync

verify:
    MUJOCO_GL=glx uv run python scripts/verify_blackwell_env.py
