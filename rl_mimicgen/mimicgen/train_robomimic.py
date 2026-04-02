from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parents[2]
for repo_name in ("mimicgen", "robomimic", "robosuite", "robosuite-task-zoo"):
    repo_path = WORKSPACE_DIR / repo_name
    if repo_path.exists():
        sys.path.insert(0, str(repo_path))

import mimicgen  # noqa: F401


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "glx")
    train_script = WORKSPACE_DIR / "robomimic" / "robomimic" / "scripts" / "train.py"
    runpy.run_path(str(train_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
