from __future__ import annotations

import os
import re
import socket
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_SCRIPT = REPO_ROOT / "scripts" / "run_dppo_batch_worker.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run_worker(tmp_path: Path, task_text: str, pipeline_body: str) -> subprocess.CompletedProcess[str]:
    task_file = tmp_path / "tasks.txt"
    task_file.write_text(task_text, encoding="utf-8")
    pipeline_script = tmp_path / "pipeline.sh"
    _write_executable(pipeline_script, pipeline_body)

    env = os.environ.copy()
    env.update(
        {
            "PIPELINE_SCRIPT": str(pipeline_script),
            "TASK_FILE": str(task_file),
            "LOCK_FILE": str(tmp_path / "tasks.lock"),
            "LOG_ROOT": str(tmp_path / "logs"),
            "RUN_ID": "test_run",
            "WORKER_ID": f"{socket.gethostname()}-424242",
            "MAX_RETRIES": "3",
        }
    )
    return subprocess.run(
        ["bash", str(WORKER_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def test_batch_worker_retries_failed_tasks_up_to_three_times(tmp_path: Path) -> None:
    result = _run_worker(
        tmp_path,
        "TODO flaky_task\n",
        "#!/usr/bin/env bash\nexit 1\n",
    )

    assert result.returncode == 1
    queue_text = (tmp_path / "tasks.txt").read_text(encoding="utf-8")
    assert re.search(r"^FAILED flaky_task .+ 3$", queue_text, re.MULTILINE)
    assert "Task failed permanently after exhausting retries: flaky_task" in result.stdout


def test_batch_worker_recovers_stale_running_task_and_increments_retry(tmp_path: Path) -> None:
    stale_worker = f"{socket.gethostname()}-999999"
    result = _run_worker(
        tmp_path,
        f"RUNNING stale_task {stale_worker} 2026-04-23T09:53:27-07:00 0\n",
        "#!/usr/bin/env bash\nexit 0\n",
    )

    assert result.returncode == 0
    queue_text = (tmp_path / "tasks.txt").read_text(encoding="utf-8")
    assert re.search(r"^DONE stale_task .+ 1$", queue_text, re.MULTILINE)
    assert "Recovered 1 stale RUNNING task(s) back to TODO." in result.stdout
