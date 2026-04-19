"""Parity smoke test: warp contact-group checks vs mujoco-python equivalents.

Builds a ``Coffee_D0`` env under both backends (``num_envs=1`` warp, single
non-warp), synthesises a few qpos configurations directly (default reset
state, pod co-located with pod-holder, pod co-located with fingerpads), and
asserts the three contact-dependent pod checks agree between backends:

    _check_pod_is_grasped
    _check_pod_and_pod_holder_contact
    _check_pod_on_rim

Bypasses the controller / action path entirely — we only exercise
``sim.forward()`` so collision detection runs against posed qpos. Exits
nonzero on any disagreement.

Usage::

    python -m rl_mimicgen.rsl_rl.test_warp_contact_parity
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("robomimic", "robosuite", "mimicgen"):
    sys.path.insert(0, str(_REPO_ROOT / "resources" / _sub))

os.environ.setdefault("ROBOSUITE_WARP_GRAPH", "0")

import robosuite  # noqa: E402
import mimicgen  # noqa: F401,E402  (registers Coffee_D0 etc. with robosuite)
import mimicgen.envs.robosuite.coffee  # noqa: F401,E402
import warp as wp  # noqa: E402

from rl_mimicgen.rsl_rl.warp_buffer_sizes import resolve_warp_buffer_sizes  # noqa: E402


CHECKS: tuple[str, ...] = (
    "_check_pod_is_grasped",
    "_check_pod_and_pod_holder_contact",
    "_check_pod_on_rim",
)


def _to_bool_tensor(v) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v.to(torch.bool).flatten()
    if isinstance(v, np.ndarray):
        return torch.as_tensor(v.astype(bool)).flatten()
    return torch.tensor([bool(v)])


def build_env(use_warp: bool, env_name: str) -> object:
    kwargs: dict = dict(
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        horizon=1000,
        control_freq=20,
        reward_shaping=False,
        ignore_done=True,
    )
    if use_warp:
        kwargs["use_warp"] = True
        kwargs["num_envs"] = 1
        caps = resolve_warp_buffer_sizes(env_name) or {}
        kwargs.update(caps)
    return robosuite.make(env_name, **kwargs)


def pose_both(warp_env, cpu_env, qpos: np.ndarray, qvel: np.ndarray) -> None:
    """Set both envs to the given qpos/qvel and run ``sim.forward()``."""
    nq = qpos.shape[0]
    nv = qvel.shape[0]
    cpu_env.sim.data.qpos[:] = qpos
    cpu_env.sim.data.qvel[:] = qvel
    cpu_env.sim.forward()

    warp_qpos = wp.from_numpy(
        np.tile(qpos.astype(np.float32), (1, 1)),
        dtype=wp.float32,
        device=warp_env.sim._warp_data.qpos.device,
    )
    warp_qvel = wp.from_numpy(
        np.tile(qvel.astype(np.float32), (1, 1)),
        dtype=wp.float32,
        device=warp_env.sim._warp_data.qvel.device,
    )
    wp.copy(warp_env.sim._warp_data.qpos, warp_qpos)
    wp.copy(warp_env.sim._warp_data.qvel, warp_qvel)
    warp_env.sim.forward()


def state_default_reset(warp_env, cpu_env) -> tuple[np.ndarray, np.ndarray]:
    """Return the warp env's post-reset qpos/qvel for env 0."""
    return (
        warp_env.sim._warp_data.qpos.numpy()[0].copy(),
        warp_env.sim._warp_data.qvel.numpy()[0].copy(),
    )


def state_pod_on_holder(warp_env, cpu_env, qpos0: np.ndarray, qvel0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Move the pod's xyz to coincide with the pod-holder body position."""
    qpos = qpos0.copy()
    qvel = qvel0.copy()
    holder_pos = cpu_env.sim.data.body_xpos[cpu_env.obj_body_id["coffee_pod_holder"]].copy()
    pod_joint = cpu_env.coffee_pod.joints[0]
    addr = cpu_env.sim.model.get_joint_qpos_addr(pod_joint)
    start = addr[0] if isinstance(addr, tuple) else addr
    qpos[start:start + 3] = holder_pos + np.array([0.0, 0.0, 0.005])
    qpos[start + 3:start + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    return qpos, qvel


def state_pod_between_fingers(warp_env, cpu_env, qpos0: np.ndarray, qvel0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Move the pod between the two fingerpad geoms."""
    qpos = qpos0.copy()
    qvel = qvel0.copy()
    gripper = cpu_env.robots[0].gripper
    left_ids = [cpu_env.sim.model.geom_name2id(g) for g in gripper.important_geoms["left_fingerpad"]]
    right_ids = [cpu_env.sim.model.geom_name2id(g) for g in gripper.important_geoms["right_fingerpad"]]
    left_pos = np.mean([cpu_env.sim.data.geom_xpos[g] for g in left_ids], axis=0)
    right_pos = np.mean([cpu_env.sim.data.geom_xpos[g] for g in right_ids], axis=0)
    mid = 0.5 * (left_pos + right_pos)

    pod_joint = cpu_env.coffee_pod.joints[0]
    addr = cpu_env.sim.model.get_joint_qpos_addr(pod_joint)
    start = addr[0] if isinstance(addr, tuple) else addr
    qpos[start:start + 3] = mid
    qpos[start + 3:start + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    return qpos, qvel


def compare(warp_env, cpu_env, label: str, mismatches: dict[str, int], true_seen: dict[str, int]) -> None:
    print(f"[state] {label}", file=sys.stderr)
    for name in CHECKS:
        warp_out = _to_bool_tensor(getattr(warp_env, name)()).cpu()
        cpu_out = _to_bool_tensor(getattr(cpu_env, name)()).cpu()
        assert warp_out.numel() == 1, f"warp {name} returned shape {warp_out.shape}"
        assert cpu_out.numel() == 1, f"cpu  {name} returned shape {cpu_out.shape}"
        w = bool(warp_out.item())
        c = bool(cpu_out.item())
        print(f"    {name:40s}  warp={w!s:5s}  cpu={c!s:5s}", file=sys.stderr)
        if w != c:
            mismatches[name] += 1
        if w or c:
            true_seen[name] += 1


def run(env_name: str, seed: int) -> int:
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"[info] building warp env {env_name}", file=sys.stderr)
    warp_env = build_env(use_warp=True, env_name=env_name)
    warp_env.reset()

    print(f"[info] building non-warp env {env_name}", file=sys.stderr)
    cpu_env = build_env(use_warp=False, env_name=env_name)
    cpu_env.reset()

    qpos0, qvel0 = state_default_reset(warp_env, cpu_env)
    pose_both(warp_env, cpu_env, qpos0, qvel0)

    mismatches: dict[str, int] = {name: 0 for name in CHECKS}
    true_seen: dict[str, int] = {name: 0 for name in CHECKS}
    compare(warp_env, cpu_env, "default_reset", mismatches, true_seen)

    qpos_h, qvel_h = state_pod_on_holder(warp_env, cpu_env, qpos0, qvel0)
    pose_both(warp_env, cpu_env, qpos_h, qvel_h)
    compare(warp_env, cpu_env, "pod_on_holder", mismatches, true_seen)

    qpos_f, qvel_f = state_pod_between_fingers(warp_env, cpu_env, qpos0, qvel0)
    pose_both(warp_env, cpu_env, qpos_f, qvel_f)
    compare(warp_env, cpu_env, "pod_between_fingers", mismatches, true_seen)

    print("\n=== results ===")
    any_fail = 0
    for name in CHECKS:
        status = "OK" if mismatches[name] == 0 else "FAIL"
        if mismatches[name] != 0:
            any_fail = 1
        print(f"  {name:40s}  {status}  mismatches={mismatches[name]}  true_in_either={true_seen[name]}")
    return any_fail


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="Coffee_D0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rc = run(args.env, args.seed)
    sys.exit(rc)


if __name__ == "__main__":
    main()
