from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_tasks.manager_based.manipulation.lift import mdp

from rl_mimicgen.tasks.coffee.coffee_cfg import CoffeeEnvCfg

# from rl_mimicgen.tasks.utils.robots import FRANKA_PANDA_CFG


@configclass
class FrankaCoffeeEnvCfg(CoffeeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "panda_joint[1,3,5]": 0.0,
                    "panda_joint2": 0.196,
                    "panda_joint4": -2.62,
                    "panda_joint6": 2.94,
                    "panda_joint7": 0.785,
                    "panda_finger_joint.*": 0.02,
                },
                pos=(-0.56, 0.0, 0.912),
            ),
        )

        # ee frame transformer
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_hand", name="end_effector"),
            ],
        )
        self.scene.ee_frame.visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        # object frame transformer
        self.scene.eef_to_objects.prim_path = "{ENV_REGEX_NS}/Robot/panda_hand"
        self.scene.eef_to_objects.source_frame_offset = OffsetCfg(pos=(0.0, 0.0, 0.1034))

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_joint.*": 0.04},
            close_command_expr={"panda_finger_joint.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"


@configclass
class FrankaCoffeeEnvCfg_PLAY(FrankaCoffeeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
