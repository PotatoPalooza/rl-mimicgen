"""Verify slim per-env reset doesn't clobber kept envs' object placements.

Reproduces the bug where `_reset_internal`'s warp branch wrote a
`(num_envs, 7)` tensor with row-0's sampled placement broadcast into all
rows, overwriting every kept env's object qpos on each per-env reset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("robomimic", "robosuite", "mimicgen"):
    sys.path.insert(0, str(_REPO_ROOT / "resources" / _sub))


def main() -> None:
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    import warp as wp

    from rl_mimicgen.rsl_rl import RobomimicVecEnv, load_bc_checkpoint
    from rl_mimicgen.rsl_rl.warp_buffer_sizes import resolve_warp_buffer_sizes

    bc_path = _REPO_ROOT / "runs" / "robomimic_rl" / "2026-04-18_20-28-43" / "bc_ckpt" / "model_2000.pth"
    bc_info = load_bc_checkpoint(str(bc_path))
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=[bc_info.obs_modalities])

    env_meta = bc_info.env_meta
    env_meta.setdefault("env_kwargs", {}).update(
        {"extra_randomization": True, "tipover_prob": 0.0}
    )
    caps = resolve_warp_buffer_sizes(env_meta.get("env_name")) or {}
    if caps:
        env_meta["env_kwargs"].update(caps)

    num_envs = 8
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, render_offscreen=False, use_image_obs=False,
        use_warp=True, num_envs=num_envs,
    )
    vec_env = RobomimicVecEnv(env=env, horizon=400, device="cuda:0", obs_keys=bc_info.obs_keys)
    vec_env.reset()

    rsuite = env.env
    sim = rsuite.sim

    pod_addr = sim.model.get_joint_qpos_addr("coffee_pod_joint0")
    holder_addr = sim.model.get_joint_qpos_addr("coffee_machine_joint0")
    qpos = wp.to_torch(sim._warp_data.qpos)  # (num_envs, nq)

    print("=== before reset of envs [2,5] ===")
    pod_before = qpos[:, pod_addr[0]:pod_addr[1]].detach().clone().cpu()
    holder_before = qpos[:, holder_addr[0]:holder_addr[1]].detach().clone().cpu()
    for i in range(num_envs):
        print(f"  env {i}: pod_xy={pod_before[i, :2].tolist()}  "
              f"holder_xy={holder_before[i, :2].tolist()}")

    reset_mask = torch.zeros(num_envs, dtype=torch.bool, device="cuda:0")
    reset_mask[2] = True
    reset_mask[5] = True
    vec_env._reset_envs(reset_mask)

    print("\n=== after reset of envs [2,5] ===")
    pod_after = qpos[:, pod_addr[0]:pod_addr[1]].detach().clone().cpu()
    holder_after = qpos[:, holder_addr[0]:holder_addr[1]].detach().clone().cpu()
    for i in range(num_envs):
        pod_moved = not torch.allclose(pod_before[i], pod_after[i], atol=1e-5)
        holder_moved = not torch.allclose(holder_before[i], holder_after[i], atol=1e-5)
        tag = " [RESET]" if i in (2, 5) else ""
        print(f"  env {i}: pod_moved={pod_moved}  holder_moved={holder_moved}{tag}")

    # Kept envs (all except 2,5) must NOT have their objects moved.
    fail = []
    for i in range(num_envs):
        if i in (2, 5):
            continue
        if not torch.allclose(pod_before[i], pod_after[i], atol=1e-5):
            fail.append(f"env {i} pod moved: {pod_before[i].tolist()} -> {pod_after[i].tolist()}")
        if not torch.allclose(holder_before[i], holder_after[i], atol=1e-5):
            fail.append(f"env {i} holder moved: {holder_before[i].tolist()} -> {holder_after[i].tolist()}")

    if fail:
        print("\n=== FAIL: kept envs had objects moved by masked reset ===")
        for line in fail:
            print(f"  {line}")
        sys.exit(1)

    # Reset envs MUST have their objects moved.
    for i in (2, 5):
        if torch.allclose(pod_before[i], pod_after[i], atol=1e-5):
            print(f"\n=== FAIL: reset env {i} pod did not move ===")
            sys.exit(1)

    print("\n=== PASS: kept envs' objects untouched; reset envs got fresh placements ===")


if __name__ == "__main__":
    main()
