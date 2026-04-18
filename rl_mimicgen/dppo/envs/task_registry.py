from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


WORKSPACE_DIR = Path(__file__).resolve().parents[3]

DEFAULT_LOW_DIM_OBS_KEYS: tuple[str, ...] = (
    "object",
    "robot0_eef_pos",
    "robot0_eef_pos_rel_pod",
    "robot0_eef_pos_rel_pod_holder",
    "robot0_eef_quat",
    "robot0_eef_quat_rel_pod",
    "robot0_eef_quat_rel_pod_holder",
    "robot0_eef_vel_ang",
    "robot0_eef_vel_lin",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "robot0_joint_pos",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
)

SQUARE_LOW_DIM_OBS_KEYS: tuple[str, ...] = (
    "object",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_eef_vel_ang",
    "robot0_eef_vel_lin",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "robot0_joint_pos",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
)


@dataclass(frozen=True, slots=True)
class DPPOTaskSpec:
    task: str
    variant: str
    env_name: str
    dataset_path: str
    horizon: int
    obs_keys: tuple[str, ...] = DEFAULT_LOW_DIM_OBS_KEYS


TASK_REGISTRY: dict[tuple[str, str], DPPOTaskSpec] = {
    ("coffee", "d0"): DPPOTaskSpec(
        task="coffee",
        variant="D0",
        env_name="Coffee_D0",
        dataset_path=str(WORKSPACE_DIR / "runs/datasets/core/coffee_d0.hdf5"),
        horizon=400,
    ),
    ("coffee", "d1"): DPPOTaskSpec(
        task="coffee",
        variant="D1",
        env_name="Coffee_D1",
        dataset_path=str(WORKSPACE_DIR / "runs/datasets/core/coffee_d1.hdf5"),
        horizon=400,
    ),
    ("coffee", "d2"): DPPOTaskSpec(
        task="coffee",
        variant="D2",
        env_name="Coffee_D2",
        dataset_path=str(WORKSPACE_DIR / "runs/datasets/core/coffee_d2.hdf5"),
        horizon=400,
    ),
    ("square", "d0"): DPPOTaskSpec(
        task="square",
        variant="D0",
        env_name="Square_D0",
        dataset_path=str(WORKSPACE_DIR / "runs/datasets/core/square_d0.hdf5"),
        horizon=400,
        obs_keys=SQUARE_LOW_DIM_OBS_KEYS,
    ),
    ("square", "d1"): DPPOTaskSpec(
        task="square",
        variant="D1",
        env_name="Square_D1",
        dataset_path=str(WORKSPACE_DIR / "runs/datasets/core/square_d1.hdf5"),
        horizon=400,
        obs_keys=SQUARE_LOW_DIM_OBS_KEYS,
    ),
    ("square", "d2"): DPPOTaskSpec(
        task="square",
        variant="D2",
        env_name="Square_D2",
        dataset_path=str(WORKSPACE_DIR / "runs/datasets/core/square_d2.hdf5"),
        horizon=400,
        obs_keys=SQUARE_LOW_DIM_OBS_KEYS,
    ),
}


def get_task_spec(task: str, variant: str) -> DPPOTaskSpec:
    key = (task.lower(), variant.lower())
    if key not in TASK_REGISTRY:
        raise KeyError(f"Unsupported DPPO task spec: task={task} variant={variant}")
    return TASK_REGISTRY[key]
