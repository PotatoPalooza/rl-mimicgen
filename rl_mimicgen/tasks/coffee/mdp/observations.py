from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("eef_to_objects"),
    machine_cfg: SceneEntityCfg = SceneEntityCfg("coffee_machine", joint_names="coffee_machine_lid_main_joint0"),
) -> torch.Tensor:
    """The absolute position and rotation of a frame target, plus its position relative to source."""
    # object states
    sensor = env.scene.sensors[sensor_cfg.name]
    assert isinstance(sensor, FrameTransformer), (
        f"Expected sensor {sensor_cfg.name} to be a FrameTransformer, instead got: {type(sensor)}"
    )
    target_pos_w = sensor.data.target_pos_w[:, sensor_cfg.body_ids, :] - env.scene.env_origins.unsqueeze(1)
    target_quat_w = sensor.data.target_quat_w[:, sensor_cfg.body_ids, :].clone()
    target_pos_source = sensor.data.target_pos_source[:, sensor_cfg.body_ids, :].clone()
    target_quat_source = sensor.data.target_quat_source[:, sensor_cfg.body_ids, :].clone()

    # coffee machine state
    machine: Articulation = env.scene[machine_cfg.name]
    hinge_qpos = machine.data.joint_pos[:, machine_cfg.joint_ids].clone()

    full_state = torch.cat(
        [
            target_pos_w.view(env.num_envs, -1),
            target_quat_w.view(env.num_envs, -1),
            target_pos_source.view(env.num_envs, -1),
            target_quat_source.view(env.num_envs, -1),
            hinge_qpos.view(env.num_envs, -1),
        ],
        dim=-1,
    )
    return full_state
