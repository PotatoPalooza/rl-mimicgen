"""Per-task MjSimWarp buffer sizes for RL training.

These override the class-level defaults on ``MjSimWarp`` (3500 / 60) via
``env_meta.env_kwargs`` so that ``robosuite.make`` pops them and stashes them
on ``MjSimWarp._ACTIVE_*`` (see
``resources/robosuite/robosuite/environments/base.py``).

**RL-only, not BC**. An A/B test showed that shrinking these caps — even when
no overflow ever occurs — selects different JIT-specialised mujoco-warp
kernels with slightly different f32 numerical behaviour. A trained policy
doesn't care, but early-training BC rollouts can destabilise (machine jitters
off the table, SR drops). RL fine-tuning starts from a trained BC policy, so
it tolerates the kernel-variant drift. BC-time rollouts keep the class
defaults.

Values here come from ``rl_mimicgen.rsl_rl.bench_jmax_audit``; set the cap to
the observed peak x ~3 (headroom for RL exploration).
"""

from __future__ import annotations

import re
from typing import Optional


# Keyed by the MimicGen short task name ("coffee", "three_piece_assembly",
# ...). Matched against the robosuite env_name by stripping underscores and
# comparing as a prefix, with the constraint that the trailing chars must be
# a variant suffix (``d<N>`` / ``o<N>``) or end-of-string — so ``coffee`` does
# **not** match ``CoffeePreparation_D0``.
PER_TASK_WARP_BUFFER_SIZES: dict[str, dict[str, int]] = {
    # Coffee_D0 audit (bench_jmax_audit on BC model_2000, 2048 envs, 400 steps,
    # 2026-04-17): peak per-world nefc=150, peak per-env nacon=5.1.
    # ~3x headroom for RL exploration.
    "coffee": {"njmax_per_env": 500, "naconmax_per_env": 15},
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
