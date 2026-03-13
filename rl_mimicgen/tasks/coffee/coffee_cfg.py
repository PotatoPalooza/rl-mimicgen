import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

import rl_mimicgen.tasks.coffee.mdp as mdp
from rl_mimicgen.tasks.utils.env import LowDimCfg, SingleArmTableSceneCfg

EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
"""Path to the extension source directory."""

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(SingleArmTableSceneCfg):
    """
    Configuration for the coffee scene. Adds coffee machine and pod to the single arm table scene.
    """

    coffee_machine = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/coffee_machine",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{EXT_DIR}/assets/coffee_machine/coffee_machine_usd/coffee_machine.usd"),
        actuators={},
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, -0.1, 0.9105), rot=(0.96592583, 0.0, 0.0, -0.25881905)),
    )
    coffee_pod = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/coffee_pod",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{EXT_DIR}/assets/coffee_pod/coffee_pod_usd/coffee_pod.usd"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.1, 0.2, 0.823103), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    eef_to_objects = FrameTransformerCfg(
        prim_path=MISSING,
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/coffee_pod", name="pod"),
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/coffee_machine/coffee_machine_root", name="machine"),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/coffee_machine/coffee_machine_pod_holder_root",
                name="pod_holder",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/coffee_machine/coffee_machine_lid_main",
                name="lid",
            ),
        ],
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.25, 0.25),
            pos_z=(0.25, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    class CoffeeLowDimCfg(LowDimCfg):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.robot0_gripper_qpos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint.*")}
            self.object = ObsTerm(func=mdp.object_state)

    @configclass
    class PolicyCfg(CoffeeLowDimCfg):
        """Observations for policy group."""

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    low_dim = CoffeeLowDimCfg()
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_coffee_pod = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("coffee_pod"),
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    task_reward = RewTerm(
        func=mdp.coffee_task_reward,
        weight=1.0,
        params={
            "lid_cfg": SceneEntityCfg(
                "coffee_machine",
                body_names="coffee_machine_lid_main",
                joint_names="coffee_machine_lid_main_joint0",
            ),
            "pod_cfg": SceneEntityCfg("coffee_pod"),
            "holder_cfg": SceneEntityCfg("coffee_machine", body_names="coffee_machine_pod_holder_root"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class CoffeeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 8.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
