import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def joint_pos_cos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.cos(asset.data.joint_pos[:, asset_cfg.joint_ids])


def joint_pos_sin(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sin(asset.data.joint_pos[:, asset_cfg.joint_ids])


def frame_transform_pos(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    assert isinstance(sensor, FrameTransformer), (
        f"Expected sensor {sensor_cfg.name} to be a FrameTransformer, instead got: {type(sensor)}"
    )
    pos = sensor.data.target_pos_source[:, sensor_cfg.body_ids, :].clone()
    return pos.view(env.num_envs, -1)


def frame_transform_quat(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    assert isinstance(sensor, FrameTransformer), (
        f"Expected sensor {sensor_cfg.name} to be a FrameTransformer, instead got: {type(sensor)}"
    )
    quat = sensor.data.target_quat_source[:, sensor_cfg.body_ids, :].clone()
    return quat.view(env.num_envs, -1)


def frame_inverse_pos(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    assert isinstance(sensor, FrameTransformer), (
        f"Expected sensor {sensor_cfg.name} to be a FrameTransformer, instead got: {type(sensor)}"
    )
    source_pos_target, _ = math_utils.subtract_frame_transforms(
        sensor.data.target_pos_w[:, sensor_cfg.body_ids, :],
        sensor.data.target_quat_w[:, sensor_cfg.body_ids, :],
        sensor.data.source_pos_w.unsqueeze(1),
    )
    return source_pos_target.view(env.num_envs, -1)


def frame_inverse_quat(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    assert isinstance(sensor, FrameTransformer), (
        f"Expected sensor {sensor_cfg.name} to be a FrameTransformer, instead got: {type(sensor)}"
    )
    _, source_quat_target = math_utils.subtract_frame_transforms(
        sensor.data.target_pos_w[:, sensor_cfg.body_ids, :],
        sensor.data.target_quat_w[:, sensor_cfg.body_ids, :],
        q02=sensor.data.source_quat_w.unsqueeze(1),
    )
    return source_quat_target.view(env.num_envs, -1)


def frame_to_frame_pos(env: ManagerBasedRLEnv, source_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg) -> torch.Tensor:
    source = env.scene.sensors[source_cfg.name]
    assert isinstance(source, FrameTransformer), (
        f"Expected sensor {source_cfg.name} to be a FrameTransformer, instead got: {type(source)}"
    )
    assert not isinstance(source_cfg.body_ids, slice) and len(source_cfg.body_ids) == 1, (
        "Only a single source frame is supported."
    )
    target = env.scene.sensors[target_cfg.name]
    assert isinstance(target, FrameTransformer), (
        f"Expected sensor {target_cfg.name} to be a FrameTransformer, instead got: {type(target)}"
    )
    target_pos_source, _ = math_utils.subtract_frame_transforms(
        source.data.target_pos_w[:, source_cfg.body_ids, :],
        source.data.target_quat_w[:, source_cfg.body_ids, :],
        target.data.target_pos_w[:, target_cfg.body_ids, :],
    )
    return target_pos_source.view(env.num_envs, -1)


def frame_to_frame_quat(env: ManagerBasedRLEnv, source_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg) -> torch.Tensor:
    source = env.scene.sensors[source_cfg.name]
    assert isinstance(source, FrameTransformer), (
        f"Expected sensor {source_cfg.name} to be a FrameTransformer, instead got: {type(source)}"
    )
    assert not isinstance(source_cfg.body_ids, slice) and len(source_cfg.body_ids) == 1, (
        "Only a single source frame is supported."
    )
    target = env.scene.sensors[target_cfg.name]
    assert isinstance(target, FrameTransformer), (
        f"Expected sensor {target_cfg.name} to be a FrameTransformer, instead got: {type(target)}"
    )
    _, target_quat_source = math_utils.subtract_frame_transforms(
        source.data.target_pos_w[:, source_cfg.body_ids, :],
        source.data.target_quat_w[:, source_cfg.body_ids, :],
        q02=target.data.target_quat_w[:, target_cfg.body_ids, :],
    )
    return target_quat_source.view(env.num_envs, -1)


def body_pos_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    body_pos_b, _ = math_utils.subtract_frame_transforms(
        asset.data.root_pos_w.unsqueeze(1),
        asset.data.root_quat_w.unsqueeze(1),
        body_pos_w,
    )
    return body_pos_b.view(env.num_envs, -1)


def body_quat_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    _, body_quat_b = math_utils.subtract_frame_transforms(
        asset.data.root_pos_w.unsqueeze(1),
        asset.data.root_quat_w.unsqueeze(1),
        q02=body_quat_w,
    )
    # robomimic uses xyzw quats
    body_quat_b = math_utils.convert_quat(body_quat_b, to="xyzw")
    # necessary bc isaac lab typing is bad
    assert isinstance(body_quat_b, torch.Tensor)
    return body_quat_b.view(env.num_envs, -1)


def object_to_frame_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    frame_cfg: SceneEntityCfg = SceneEntityCfg("eef_frame"),
) -> torch.Tensor:
    """The relative position of the object in a provided FrameTransformer frame. If a body is specified in the object
    config, the position of that body is reported. Otherwise, the object's root frame is used."""
    object: RigidObject | Articulation = env.scene[asset_cfg.name]
    frame = env.scene.sensors[frame_cfg.name]
    assert isinstance(frame, FrameTransformer), (
        f"Expected sensor {frame_cfg.name} to be a FrameTransformer, instead got: {type(frame)}"
    )
    if not isinstance(asset_cfg.body_ids, slice):
        if len(asset_cfg.body_ids) != 1:
            raise ValueError("At most one object body may be specified.")
        object_pos_w = object.data.body_pos_w[:, asset_cfg.body_ids, :].squeeze(1)
    else:
        object_pos_w = object.data.root_pos_w
    object_pos_frame, _ = math_utils.subtract_frame_transforms(
        frame.data.target_pos_w,
        frame.data.target_quat_w,
        object_pos_w,
    )
    return object_pos_frame.view(env.num_envs, -1)


def object_to_frame_quat(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    frame_cfg: SceneEntityCfg = SceneEntityCfg("eef_frame"),
) -> torch.Tensor:
    """The relative rotation (quat) of the object in a provided FrameTransformer frame. If a body is specified in the
    object config, the rotation of that body is reported. Otherwise, the object's root frame is used."""
    object: RigidObject | Articulation = env.scene[asset_cfg.name]
    frame = env.scene.sensors[frame_cfg.name]
    assert isinstance(frame, FrameTransformer), (
        f"Expected sensor {frame_cfg.name} to be a FrameTransformer, instead got: {type(frame)}"
    )
    if not isinstance(asset_cfg.body_ids, slice):
        if len(asset_cfg.body_ids) != 1:
            raise ValueError("At most one object body may be specified.")
        object_quat_w = object.data.body_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    else:
        object_quat_w = object.data.root_quat_w
    _, object_quat_frame = math_utils.subtract_frame_transforms(
        frame.data.target_pos_w,
        frame.data.target_quat_w,
        q02=object_quat_w,
    )
    return object_quat_frame.view(env.num_envs, -1)


def frame_pos_w(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """The world position of the specified bodies (relative to environment origin)."""
    frame = env.scene.sensors[sensor_cfg.name]
    assert isinstance(frame, FrameTransformer), (
        f"Expected sensor {sensor_cfg.name} to be a FrameTransformer, instead got: {type(frame)}"
    )
    target_pos_w = frame.data.target_pos_w[:, sensor_cfg.body_ids, :] - env.scene.env_origins.unsqueeze(1)
    return target_pos_w.view(env.num_envs, -1)


def frame_quat_w(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """The absolute quaternion of the specified bodies."""
    frame = env.scene.sensors[sensor_cfg.name]
    assert isinstance(frame, FrameTransformer), (
        f"Expected sensor {sensor_cfg.name} to be a FrameTransformer, instead got: {type(frame)}"
    )
    target_quat_w = frame.data.target_quat_w[:, sensor_cfg.body_ids, :].clone()
    return target_quat_w.view(env.num_envs, -1)


def frame_full_state(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """The absolute position and rotation of a frame target, plus its position relative to source."""
    sensor = env.scene.sensors[sensor_cfg.name]
    assert isinstance(sensor, FrameTransformer), (
        f"Expected sensor {sensor_cfg.name} to be a FrameTransformer, instead got: {type(sensor)}"
    )
    target_pos_w = sensor.data.target_pos_w[:, sensor_cfg.body_ids, :] - env.scene.env_origins.unsqueeze(1)
    target_quat_w = sensor.data.target_quat_w[:, sensor_cfg.body_ids, :].clone()
    target_pos_source = sensor.data.target_pos_source[:, sensor_cfg.body_ids, :].clone()
    full_state = torch.cat([target_pos_w, target_quat_w, target_pos_source], dim=-1)
    return full_state.view(env.num_envs, -1)
