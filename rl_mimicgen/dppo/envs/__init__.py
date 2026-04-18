"""Environment registry and wrappers for DPPO."""

from rl_mimicgen.dppo.envs.mimicgen_lowdim_env import MimicGenLowDimEnv, make_mimicgen_lowdim_env
from rl_mimicgen.dppo.envs.task_registry import DPPOTaskSpec, get_task_spec

__all__ = [
    "DPPOTaskSpec",
    "MimicGenLowDimEnv",
    "get_task_spec",
    "make_mimicgen_lowdim_env",
]
