"""Async checkpoint-eval manager for upstream DPPO pretraining.

Exports :class:`AsyncCheckpointEvalManager`, a background-thread runner that
instantiates an ``EvalDiffusionAgent`` once (persistent vectorized env +
eval model), receives EMA state_dicts from the BC training loop, and runs
rollouts on a worker thread so training keeps making progress. Results are
drained by the training loop and logged to wandb using a ``local_step``
metric so past-epoch rollouts plot correctly.
"""

from __future__ import annotations

from rl_mimicgen.dppo_async.async_eval import (
    AsyncCheckpointEvalManager,
    AsyncEvalResult,
)

__all__ = ["AsyncCheckpointEvalManager", "AsyncEvalResult"]
