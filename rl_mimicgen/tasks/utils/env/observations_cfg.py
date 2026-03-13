from dataclasses import MISSING

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

import rl_mimicgen.tasks.mdp as mdp


@configclass
class LowDimCfg(ObsGroup):
    robot0_eef_pos = ObsTerm(func=mdp.frame_transform_pos)
    robot0_eef_quat = ObsTerm(func=mdp.frame_transform_quat)
    robot0_gripper_qpos = ObsTerm(func=mdp.joint_pos_rel, params=MISSING)
    object: ObsTerm = MISSING

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = False
