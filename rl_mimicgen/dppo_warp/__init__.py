"""Warp-backed VecEnv adapter for upstream DPPO's fine-tuning stack.

Exports :class:`WarpRobomimicVectorEnv`, a drop-in replacement for
``AsyncVectorEnv(RobomimicLowdimWrapper + MultiStep)`` built on top of a
single MuJoCo-Warp batched robosuite env. Wired into upstream DPPO via the
``env_type: warp`` branch in ``resources/dppo/env/gym_utils/__init__.py``.
"""

from rl_mimicgen.dppo_warp.warp_vector_env import WarpRobomimicVectorEnv

__all__ = ["WarpRobomimicVectorEnv"]
