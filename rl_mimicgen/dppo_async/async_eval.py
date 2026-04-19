"""Background-thread eval runner for upstream DPPO BC pretraining.

BC pretraining's rollout-eval step is normally the dominant tail-cost after
each ``save_model_freq`` checkpoint — the training loop blocks while the
eval subprocess finishes. ``AsyncCheckpointEvalManager`` moves that work
onto a worker thread so training keeps making progress while evaluation
runs in parallel on the same GPU (Python threads hold the GIL but CUDA
kernels are async, so compute overlaps cleanly).

Design summary:
  - The manager owns an ``EvalDiffusionAgent`` (persistent vectorized env
    + eval model), instantiated once at manager init from the same Hydra
    eval config the subprocess path would have used.
  - Each ``submit(epoch, state_dict)`` call eagerly clones the main
    model's EMA weights (``detach().clone()`` per tensor, stays on device)
    and pushes ``(epoch, state_dict_snapshot)`` onto a bounded queue. If
    the queue is full, ``submit`` blocks — intended backpressure so
    training can't get arbitrarily far ahead of evaluation.
  - The worker thread pops a request, loads the snapshot into the eval
    agent's model via ``load_pretrain_state_dict``, runs
    ``run_return_metrics``, and pushes an :class:`AsyncEvalResult` back.
  - The main thread calls ``drain()`` once per training iter to collect
    results and log them via wandb's ``local_step`` custom x-axis (past
    epoch numbers). At shutdown, ``drain_blocking()`` waits for any
    remaining rollouts and ``close()`` joins the worker.

Only pretraining (train-on-fixed-dataset) is the intended caller. The main
thread never touches the eval env or eval model after init, so no lock is
needed.
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)

# Sentinel placed on the request queue to tell the worker to exit.
_SHUTDOWN = object()


@dataclass
class AsyncEvalResult:
    """One completed async checkpoint-eval rollout.

    Consumed by the training loop's drain pass to log metrics and videos
    at ``local_step=epoch``. ``ema_state_dict`` carries the exact weights
    that produced the metrics, so a best-score checkpoint can be written
    with those weights rather than the current (drifted) live weights.
    """

    epoch: int
    metrics: dict
    video_paths: list[str]
    wall_time: float
    error: BaseException | None = None
    ema_state_dict: dict | None = None
    extra: dict = field(default_factory=dict)


class AsyncCheckpointEvalManager:
    """Background-thread wrapper around ``EvalDiffusionAgent.run_return_metrics``.

    The manager loads the upstream DPPO eval Hydra config from
    ``eval_config_dir + eval_config_name``, builds an
    ``EvalDiffusionAgent`` once (the expensive step: vectorized env build
    + eval model instantiation), and keeps the worker thread polling a
    bounded request queue.

    Notes:
      - The eval config's ``base_policy_path`` is overwritten with a
        temporary checkpoint file populated from the caller-provided
        ``initial_ema_state_dict``. The temp file is deleted right after
        the eval agent finishes construction; subsequent submits update
        weights in-memory via ``load_pretrain_state_dict``.
      - Weight snapshots are per-tensor ``detach().clone()`` on the
        submitter's thread, so training can keep mutating the live EMA
        model while the worker consumes the snapshot.
      - ``video_dir`` is the **root** directory for eval videos; each
        submitted epoch gets a ``state_{epoch}/`` subdirectory so async
        rollouts don't clobber each other.
    """

    def __init__(
        self,
        *,
        eval_config_dir: str | Path,
        eval_config_name: str,
        initial_ema_state_dict: dict,
        device: str | None = None,
        video_dir: str | Path | None = None,
        save_video: bool = False,
        render_num: int | None = None,
        n_envs: int | None = None,
        n_steps: int | None = None,
        n_episodes: int | None = None,
        max_episode_steps: int | None = None,
        queue_size: int = 2,
        overrides: dict | None = None,
    ) -> None:
        self._device = device
        self._video_dir = Path(video_dir).expanduser().resolve() if video_dir is not None else None

        self._request_q: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._result_q: queue.Queue = queue.Queue()
        self._inflight = 0
        self._inflight_lock = threading.Lock()
        self._stopped = False

        self._eval_agent = self._build_eval_agent(
            eval_config_dir=Path(eval_config_dir).expanduser().resolve(),
            eval_config_name=eval_config_name,
            initial_ema_state_dict=initial_ema_state_dict,
            device=device,
            save_video=save_video,
            render_num=render_num,
            n_envs=n_envs,
            n_steps=n_steps,
            n_episodes=n_episodes,
            max_episode_steps=max_episode_steps,
            overrides=overrides,
        )

        self._worker = threading.Thread(
            target=self._run, name="async-dppo-eval", daemon=True
        )
        self._worker.start()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_dppo_sys_path() -> None:
        dppo_root = Path(__file__).resolve().parents[2] / "resources" / "dppo"
        if str(dppo_root) not in sys.path:
            sys.path.insert(0, str(dppo_root))

    def _build_eval_agent(
        self,
        *,
        eval_config_dir: Path,
        eval_config_name: str,
        initial_ema_state_dict: dict,
        device: str | None,
        save_video: bool,
        render_num: int | None,
        n_envs: int | None,
        n_steps: int | None,
        n_episodes: int | None,
        max_episode_steps: int | None,
        overrides: dict | None,
    ):
        """Compose the Hydra eval cfg and instantiate ``EvalDiffusionAgent``."""
        self._ensure_dppo_sys_path()

        import hydra
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf

        # Upstream's script/run.py registers these resolvers; do it here
        # too so ${eval:…}, ${round_up:…}, ${round_down:…} all resolve.
        import math as _math

        for name, fn in (
            ("eval", eval),
            ("round_up", _math.ceil),
            ("round_down", _math.floor),
        ):
            try:
                OmegaConf.register_new_resolver(name, fn, replace=True)
            except Exception:  # resolver already registered — benign
                pass

        # Dump the initial EMA state_dict to a temp file and point the
        # eval cfg's base_policy_path at it. The eval agent's DiffusionEval
        # constructor expects an on-disk checkpoint — we feed it a real
        # one built from the current training EMA weights, then delete.
        with tempfile.NamedTemporaryFile(
            prefix="dppo_async_eval_init_", suffix=".pt", delete=False
        ) as tmp_ckpt:
            torch.save(
                {"epoch": 0, "ema": initial_ema_state_dict},
                tmp_ckpt.name,
            )
            tmp_ckpt_path = tmp_ckpt.name

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        cfg_overrides: list[str] = [f"base_policy_path={tmp_ckpt_path}"]
        if device is not None:
            cfg_overrides.append(f"device={device}")
        if save_video:
            cfg_overrides.append("env.save_video=True")
        if render_num is not None:
            cfg_overrides.append(f"render_num={int(render_num)}")
        if n_envs is not None:
            cfg_overrides.append(f"env.n_envs={int(n_envs)}")
        if n_steps is not None:
            cfg_overrides.append(f"n_steps={int(n_steps)}")
        if n_episodes is not None:
            cfg_overrides.append(f"n_episodes={int(n_episodes)}")
        if max_episode_steps is not None:
            cfg_overrides.append(
                f"env.max_episode_steps={int(max_episode_steps)}"
            )
        if overrides:
            for key, value in overrides.items():
                cfg_overrides.append(f"{key}={value}")

        # ``initialize_config_dir`` requires an absolute path.
        with initialize_config_dir(
            config_dir=str(eval_config_dir),
            version_base=None,
        ):
            cfg = compose(config_name=eval_config_name, overrides=cfg_overrides)
            OmegaConf.resolve(cfg)

        GlobalHydra.instance().clear()

        try:
            cls = hydra.utils.get_class(cfg._target_)
            log.info(
                "Async eval manager: building %s with %d envs (video=%s, n_steps=%s, render_num=%s)",
                cfg._target_,
                int(cfg.env.n_envs),
                bool(cfg.env.get("save_video", False)),
                str(cfg.get("n_steps", "?")),
                str(cfg.get("render_num", "?")),
            )
            eval_agent = cls(cfg)
        finally:
            # Temp checkpoint is no longer needed — EvalDiffusionAgent
            # has already loaded its contents into the eval model.
            try:
                os.unlink(tmp_ckpt_path)
            except OSError:
                pass

        return eval_agent

    # ------------------------------------------------------------------
    # Public API (main thread)
    # ------------------------------------------------------------------

    def submit(self, epoch: int, ema_state_dict: dict) -> None:
        """Clone EMA weights and queue a rollout for ``epoch``.

        Blocks if the request queue is at capacity (intended backpressure).
        """
        if self._stopped:
            raise RuntimeError("AsyncCheckpointEvalManager is shut down")

        # Per-tensor detach+clone: keeps tensors on their original device
        # (cheap GPU→GPU copy) but breaks storage-sharing with the live
        # model, so training can continue mutating weights while the
        # worker consumes the snapshot.
        snapshot = {
            k: v.detach().clone() for k, v in ema_state_dict.items()
        }

        with self._inflight_lock:
            self._inflight += 1
        self._request_q.put((int(epoch), snapshot))

    def drain(self) -> list[AsyncEvalResult]:
        """Return every result currently available (non-blocking)."""
        results: list[AsyncEvalResult] = []
        while True:
            try:
                results.append(self._result_q.get_nowait())
            except queue.Empty:
                break
        return results

    def drain_blocking(
        self, timeout: float | None = None
    ) -> list[AsyncEvalResult]:
        """Wait for all in-flight rollouts to finish, then drain.

        Polls the in-flight counter with a small sleep rather than
        blocking on the result queue, so multiple completions can finish
        before we return. ``timeout=None`` waits indefinitely.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._inflight_lock:
                if self._inflight == 0 and self._request_q.empty():
                    break
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        return self.drain()

    def pending_count(self) -> int:
        """Number of submitted rollouts that haven't been drained yet."""
        with self._inflight_lock:
            return self._inflight

    def close(self, timeout: float = 30.0) -> None:
        """Stop the worker. Safe to call multiple times."""
        if self._stopped:
            return
        self._stopped = True
        try:
            self._request_q.put_nowait(_SHUTDOWN)
        except queue.Full:
            self._request_q.put(_SHUTDOWN)
        self._worker.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _video_dir_for_epoch(self, epoch: int) -> str | None:
        if self._video_dir is None:
            return None
        subdir = self._video_dir / f"state_{int(epoch)}"
        return str(subdir)

    def _run(self) -> None:
        while True:
            item = self._request_q.get()
            if item is _SHUTDOWN:
                return
            epoch, snapshot = item
            start = time.time()
            error: BaseException | None = None
            metrics: dict = {}
            video_paths: list[str] = []
            try:
                with torch.no_grad():
                    self._eval_agent.load_pretrain_state_dict(snapshot)
                metrics, video_paths = self._eval_agent.run_return_metrics(
                    video_dir=self._video_dir_for_epoch(epoch),
                    video_prefix=f"state_{int(epoch)}",
                )
            except BaseException as exc:  # surface the error to the main thread
                error = exc
                log.exception("Async eval worker failed for epoch %d", epoch)
            finally:
                with self._inflight_lock:
                    self._inflight -= 1
                self._result_q.put(
                    AsyncEvalResult(
                        epoch=int(epoch),
                        metrics=metrics,
                        video_paths=list(video_paths),
                        wall_time=time.time() - start,
                        error=error,
                        ema_state_dict=snapshot if error is None else None,
                    )
                )


__all__ = ["AsyncCheckpointEvalManager", "AsyncEvalResult"]
