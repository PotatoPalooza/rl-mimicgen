from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils import configclass


class ManagerBasedILEnv(ManagerBasedRLEnv):
    pass


@configclass
class ManagerBasedILEnvCfg(ManagerBasedRLEnvCfg):
    pass
