"""Per-task MjSimWarp buffer sizes for RL training.

These override the class-level defaults on ``MjSimWarp`` (3500 / 60) via
``env_meta.env_kwargs`` so that ``robosuite.make`` pops them and stashes them
on ``MjSimWarp._ACTIVE_*`` (see
``resources/robosuite/robosuite/environments/base.py``).

**Currently empty.** An earlier audit on Coffee_D0 (model_2000, 2048 envs)
observed peak ``nefc=150`` and suggested ``(500, 15)`` with ~3x headroom.
``bench_caps_sweep`` (256 envs, model_2000, 2026-04-18) then showed that
shrunk caps cause silent overflow during the rollout: at 500/15 the sim
clipped nefc on 10 steps and insertion rate collapsed 0.883 -> 0.246. The
earlier audit only measured peak ``nefc`` under the 3500-cap kernel variant
- once caps shrink, physics diverges and contacts spawn past the audit's
observed peak. Positive-feedback failure. Leaving class defaults in place
until a more robust audit methodology exists.
"""

from __future__ import annotations

import re
from typing import Optional


# Prefix match after underscore-strip; trailing chars must be a variant suffix
# (``d<N>``/``o<N>``) or empty so ``coffee`` does not match ``CoffeePreparation_D0``.
PER_TASK_WARP_BUFFER_SIZES: dict[str, dict[str, int]] = {
    # ThreePieceAssembly_D2 hit "narrowphase overflow - please increase nconmax to
    # 174 or naconmax to 44422" under DPPO finetune (n_envs=1024). Bump naconmax
    # per-env to absorb the spike with >2x headroom regardless of whether the
    # effective nworld is 1024 or a smaller internal split. Covers D0/D1/D2 via
    # the prefix match.
    "three_piece_assembly": {"naconmax_per_env": 200},
}


_VARIANT_SUFFIX_RE = re.compile(r"^(d\d+|o\d+)?$")


def resolve_warp_buffer_sizes(env_name: Optional[str]) -> Optional[dict[str, int]]:
    """Return the buffer-size overrides for a given robosuite env_name, or
    ``None`` if the task is unknown (callers should leave the class defaults
    alone in that case).

    Keys are tried longest-first so ``coffee_preparation`` beats ``coffee``
    for ``CoffeePreparation_D0``.
    """
    if not env_name:
        return None
    needle = env_name.lower().replace("_", "")
    for task_key in sorted(PER_TASK_WARP_BUFFER_SIZES.keys(), key=len, reverse=True):
        stripped_key = task_key.lower().replace("_", "")
        if needle.startswith(stripped_key):
            rest = needle[len(stripped_key):]
            if _VARIANT_SUFFIX_RE.match(rest):
                return dict(PER_TASK_WARP_BUFFER_SIZES[task_key])
    return None
