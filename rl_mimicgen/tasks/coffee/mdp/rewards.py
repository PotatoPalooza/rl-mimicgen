import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def coffee_task_reward(
    env: ManagerBasedRLEnv,
    lid_cfg: SceneEntityCfg,  # = SceneEntityCfg("coffee_machine", body_names="coffee_machine_lid_main"),
    pod_cfg: SceneEntityCfg,  # = SceneEntityCfg("coffee_pod"),
    holder_cfg: SceneEntityCfg,  # = SceneEntityCfg("coffee_machine", body_names="coffee_machine_pod_holder_root"),
    hinge_tolerance_deg: float = 15.0,
    pod_holder_size: tuple[float, float, float] = (0.0295, 0.0295, 0.028),
    pod_size: tuple[float, float, float] = (0.0243, 0.0243, 0.023103),
    lid_size: tuple[float, float, float] = (0.0295, 0.044, 0.0095),
) -> torch.Tensor:
    # lid should be closed (angle < hinge_tolerance)
    machine: Articulation = env.scene[lid_cfg.name]
    hinge_tolerance = torch.tensor(hinge_tolerance_deg, device=env.device).deg2rad_()
    assert (isinstance(lid_cfg.joint_ids, slice) and machine.num_joints == 1) or (
        not isinstance(lid_cfg.joint_ids, slice) and len(lid_cfg.joint_ids) == 1
    )
    lid_check = machine.data.joint_pos[:, lid_cfg.joint_ids].squeeze(1) < hinge_tolerance

    # pod should be in pod holder
    pod: RigidObject = env.scene[pod_cfg.name]
    assert holder_cfg.name == lid_cfg.name
    assert not isinstance(holder_cfg.body_ids, slice) and len(holder_cfg.body_ids) == 1
    assert not isinstance(lid_cfg.body_ids, slice) and len(lid_cfg.body_ids) == 1
    pod_holder_pos = machine.data.body_pos_w[:, holder_cfg.body_ids, :].squeeze(1)
    pod_pos = pod.data.root_pos_w
    xy_tolerance = pod_holder_size[0] - pod_size[0]
    assert xy_tolerance > 0.0
    xy_pod_check = torch.norm(pod_pos[:, :2] - pod_holder_pos[:, :2], dim=-1) > xy_tolerance

    lid_z = machine.data.body_pos_w[:, lid_cfg.body_ids, 2].squeeze(1)
    z_pod_check = torch.logical_or(
        pod_pos[:, 2] - pod_size[2] < pod_holder_pos[:, 2] - pod_holder_size[2],
        pod_pos[:, 2] + pod_size[2] > lid_z - lid_size[2],
    )
    pod_check = torch.logical_and(xy_pod_check, z_pod_check)

    # success is when both lid and pod pass checks
    return torch.logical_and(lid_check, pod_check).float()
