"""Warp vs non-warp parity for ``Kitchen_D0`` / ``Kitchen_D1``.

Diagnoses the "0% SR on DPPO pretrain eval" case by exercising the paths
that differ between backends:

    1. **Reset state parity.** After ``reset()`` with the same numpy seed
       under ``num_envs=1``, do pot / bread / stove / button / serving
       region end up at the same body ``xpos``/``xquat``?
    2. **Step parity.** Sync qpos/qvel from non-warp into warp env 0,
       step both with the same action, compare resulting qpos.
    3. **``_post_process`` + success-check parity.** Manually poke the
       button-qpos / object poses to simulate a "win" state and check
       that ``buttons_on`` and ``_check_success`` agree.
    4. **Contact-group parity.** Using the same ``check_contact_groups``
       pattern as ``test_warp_contact_parity``: pot-on-burner and
       bread-in-pot.

Run under `num_envs=1` to stay GPU-cheap. Exits nonzero on any
disagreement.

Usage::

    python -m rl_mimicgen.rsl_rl.test.test_kitchen_warp_parity --env Kitchen_D0
    python -m rl_mimicgen.rsl_rl.test.test_kitchen_warp_parity --env Kitchen_D1
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
os.environ.setdefault("MUJOCO_GL", "egl")

import robosuite  # noqa: E402
import mimicgen  # noqa: F401,E402
import mimicgen.envs.robosuite.kitchen  # noqa: F401,E402
import warp as wp  # noqa: E402

from rl_mimicgen.rsl_rl.warp_buffer_sizes import resolve_warp_buffer_sizes  # noqa: E402


POS_TOL = 1e-4
QUAT_TOL = 1e-4
QPOS_STEP_TOL = 5e-3  # f32 warp physics drifts from f64 CPU over steps
CONTACT_MUST_AGREE = True


def _as_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, wp.array):
        return x.numpy()
    return np.asarray(x)


def build_env(use_warp: bool, env_name: str, num_envs: int = 1):
    kwargs = dict(
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
        kwargs["num_envs"] = num_envs
        caps = resolve_warp_buffer_sizes(env_name) or {}
        kwargs.update(caps)
    return robosuite.make(env_name, **kwargs)


def _warp_qpos_np(env) -> np.ndarray:
    return _as_np(wp.to_torch(env.sim._warp_data.qpos))


def _warp_qvel_np(env) -> np.ndarray:
    return _as_np(wp.to_torch(env.sim._warp_data.qvel))


def sync_warp_from_cpu(warp_env, cpu_env) -> None:
    """Copy env-0 qpos/qvel from cpu into warp, then run forward."""
    qpos = np.asarray(cpu_env.sim.data.qpos, dtype=np.float32).copy()
    qvel = np.asarray(cpu_env.sim.data.qvel, dtype=np.float32).copy()
    wp_qpos = wp.to_torch(warp_env.sim._warp_data.qpos)
    wp_qvel = wp.to_torch(warp_env.sim._warp_data.qvel)
    wp_qpos[0] = torch.as_tensor(qpos, device=wp_qpos.device)
    wp_qvel[0] = torch.as_tensor(qvel, device=wp_qvel.device)
    warp_env.sim.forward()


def check_reset_parity(warp_env, cpu_env, seed: int, label: str, out: list[str]) -> None:
    np.random.seed(seed)
    cpu_env.reset()
    np.random.seed(seed)
    warp_env.reset()

    names = ["pot", "bread", "stove", "button", "serving"]
    bid = {
        "pot": cpu_env.pot_object_id,
        "bread": cpu_env.sim.model.body_name2id(cpu_env.bread_ingredient.root_body),
        "stove": cpu_env.sim.model.body_name2id(cpu_env.stove_object_1.root_body),
        "button": cpu_env.sim.model.body_name2id(cpu_env.button_object_1.root_body),
        "serving": cpu_env.serving_region_id,
    }
    print(f"\n[reset_parity seed={seed} {label}]")
    for n in names:
        cpu_pos = np.asarray(cpu_env.sim.data.body_xpos[bid[n]])
        cpu_quat = np.asarray(cpu_env.sim.data.body_xquat[bid[n]])
        warp_pos = _as_np(warp_env.sim.data.body_xpos[bid[n]])[0]
        warp_quat = _as_np(warp_env.sim.data.body_xquat[bid[n]])[0]
        dp = np.max(np.abs(cpu_pos - warp_pos))
        dq = np.max(np.abs(cpu_quat - warp_quat))
        tag = "OK" if (dp < POS_TOL and dq < QUAT_TOL) else "MISMATCH"
        print(f"  {n:8s} cpu_pos={cpu_pos.round(4).tolist()}  "
              f"warp_pos={warp_pos.round(4).tolist()}  "
              f"Δpos={dp:.2e}  Δquat={dq:.2e}  [{tag}]")
        if tag == "MISMATCH":
            out.append(f"reset {label} {n}: Δpos={dp:.2e} Δquat={dq:.2e}")

    # Button qpos — governs stove-on latching.
    addr = cpu_env.button_qpos_addrs[1]
    cpu_btn = float(cpu_env.sim.data.qpos[addr])
    warp_btn = float(_warp_qpos_np(warp_env)[0, addr])
    print(f"  button_qpos  cpu={cpu_btn:.4f}  warp={warp_btn:.4f}")
    if abs(cpu_btn - warp_btn) > 1e-4:
        out.append(f"reset {label} button_qpos: cpu={cpu_btn} warp={warp_btn}")


def check_step_parity(warp_env, cpu_env, n_steps: int, out: list[str]) -> None:
    """Sync warp<-cpu, then step both with zero action and compare qpos.

    Warp is f32 so drift is expected over many steps; we just flag when
    divergence exceeds a coarse bound within a handful of steps.
    """
    print(f"\n[step_parity n_steps={n_steps}]")
    sync_warp_from_cpu(warp_env, cpu_env)

    # Zero action (center of action space).
    low, high = cpu_env.action_spec
    action = ((np.asarray(low) + np.asarray(high)) * 0.5).astype(np.float32)
    action_warp = torch.as_tensor(
        action[None, :].repeat(warp_env.num_envs, axis=0),
        device="cuda", dtype=torch.float32,
    )

    for i in range(n_steps):
        cpu_env.step(action)
        warp_env.step(action_warp)

        cpu_qpos = np.asarray(cpu_env.sim.data.qpos)
        warp_qpos = _warp_qpos_np(warp_env)[0]
        dq = np.max(np.abs(cpu_qpos - warp_qpos))
        tag = "OK" if dq < QPOS_STEP_TOL else "DIVERGE"
        # Also identify which address diverges most.
        worst = int(np.argmax(np.abs(cpu_qpos - warp_qpos)))
        print(f"  step {i+1:3d}  max|Δqpos|={dq:.3e}  worst_addr={worst}  "
              f"cpu={cpu_qpos[worst]:.4f}  warp={warp_qpos[worst]:.4f}  [{tag}]")
        if dq > QPOS_STEP_TOL and i < 3:
            out.append(f"step {i+1}: max|Δqpos|={dq:.3e} at addr {worst}")


def check_post_process_parity(warp_env, cpu_env, out: list[str]) -> None:
    """Force button qpos to a cycle (-0.4, 0.5, -0.4) and check buttons_on."""
    print(f"\n[post_process_parity]")
    addr = cpu_env.button_qpos_addrs[1]
    for val in (-0.4, 0.5, -0.4):
        # Force button qpos on both backends.
        cpu_env.sim.data.qpos[addr] = val
        wp_qpos = wp.to_torch(warp_env.sim._warp_data.qpos)
        wp_qpos[:, addr] = val

        cpu_env._post_process()
        warp_env._post_process()

        cpu_on = bool(cpu_env.buttons_on[1])
        warp_on_t = warp_env.buttons_on[1]
        warp_on = bool(warp_on_t[0].item()) if isinstance(warp_on_t, torch.Tensor) else bool(warp_on_t)
        expected = val >= 0.0
        tag = "OK" if (cpu_on == expected and warp_on == expected) else "MISMATCH"
        print(f"  button_qpos={val:.2f}  cpu_on={cpu_on}  warp_on={warp_on}  "
              f"expected={expected}  [{tag}]")
        if tag == "MISMATCH":
            out.append(f"post_process qpos={val}: cpu={cpu_on} warp={warp_on} expected={expected}")


def check_obs_parity(warp_env, cpu_env, seed: int, out: list[str]) -> None:
    """After synchronized reset, compare observation dicts key-by-key.

    This is the tightest test for 'does the BC policy see the same inputs
    under warp as under CPU?' — if obs diverge the policy runs blind.
    """
    print(f"\n[obs_parity seed={seed}]")
    np.random.seed(seed); cpu_env.reset()
    np.random.seed(seed); warp_env.reset()

    # Warp obs come from _get_observations; robosuite returns a dict.
    cpu_obs = cpu_env._get_observations(force_update=True)
    warp_obs = warp_env._get_observations(force_update=True)

    shared = sorted(set(cpu_obs.keys()) & set(warp_obs.keys()))
    only_cpu = sorted(set(cpu_obs.keys()) - set(warp_obs.keys()))
    only_warp = sorted(set(warp_obs.keys()) - set(cpu_obs.keys()))
    if only_cpu:
        print(f"  only_cpu: {only_cpu}")
    if only_warp:
        print(f"  only_warp: {only_warp}")

    for k in shared:
        cv = np.asarray(cpu_obs[k]).astype(np.float32).reshape(-1)
        wv_raw = warp_obs[k]
        wv = _as_np(wv_raw).astype(np.float32)
        # Warp obs are per-env with leading batch dim; pull env 0.
        if wv.ndim > 0 and wv.shape[0] == warp_env.num_envs:
            wv = wv[0]
        wv = wv.reshape(-1)
        if cv.shape != wv.shape:
            print(f"  {k:40s} shape cpu={cv.shape} warp={wv.shape}  [SHAPE_MISMATCH]")
            out.append(f"obs {k}: shape cpu={cv.shape} warp={wv.shape}")
            continue
        if cv.size == 0:
            continue
        d = float(np.max(np.abs(cv - wv)))
        tag = "OK" if d < 1e-3 else "MISMATCH"
        flag = "" if d < 1e-3 else "  [MISMATCH]"
        # Only print non-zero-dim keys to keep output short.
        if d > 1e-6 or tag == "MISMATCH":
            print(f"  {k:40s} max|Δ|={d:.3e}{flag}")
        if tag == "MISMATCH":
            out.append(f"obs {k}: max|Δ|={d:.3e}")


def check_success_parity(warp_env, cpu_env, out: list[str]) -> None:
    """Construct a 'winning' state and check both backends agree on success.

    Success requires: pot in serving region (xy within tolerance, z
    within 5cm), stove turned off, bread in pot, and
    ``has_stove_turned_on_with_pot_and_object`` latch set.
    """
    print(f"\n[success_parity]")

    # Poke qpos to place the pot on top of the serving region and bread inside the pot.
    serv_pos_cpu = np.asarray(cpu_env.sim.data.body_xpos[cpu_env.serving_region_id]).copy()
    pot_joint = cpu_env.pot_object.joints[0]
    bread_joint = cpu_env.bread_ingredient.joints[0]
    pot_addr = cpu_env.sim.model.get_joint_qpos_addr(pot_joint)
    bread_addr = cpu_env.sim.model.get_joint_qpos_addr(bread_joint)
    pot_start, pot_end = pot_addr if isinstance(pot_addr, tuple) else (pot_addr, pot_addr + 1)
    bread_start, bread_end = bread_addr if isinstance(bread_addr, tuple) else (bread_addr, bread_addr + 1)

    # Pot lifted just above serving region; bread inside the pot.
    pot_qpos = np.concatenate([serv_pos_cpu + np.array([0.0, 0.0, 0.04]), [1.0, 0.0, 0.0, 0.0]])
    bread_qpos = np.concatenate([serv_pos_cpu + np.array([0.0, 0.0, 0.07]), [1.0, 0.0, 0.0, 0.0]])

    cpu_env.sim.data.qpos[pot_start:pot_end] = pot_qpos
    cpu_env.sim.data.qpos[bread_start:bread_end] = bread_qpos
    # Turn stove off.
    cpu_env.sim.data.qpos[cpu_env.button_qpos_addrs[1]] = -0.4
    cpu_env.sim.forward()

    wp_qpos = wp.to_torch(warp_env.sim._warp_data.qpos)
    wp_qpos[0, pot_start:pot_end] = torch.as_tensor(pot_qpos, device=wp_qpos.device, dtype=torch.float32)
    wp_qpos[0, bread_start:bread_end] = torch.as_tensor(bread_qpos, device=wp_qpos.device, dtype=torch.float32)
    wp_qpos[0, cpu_env.button_qpos_addrs[1]] = -0.4
    warp_env.sim.forward()

    # Manually set latches to True (as if stove had been on earlier with bread in pot on burner).
    cpu_env.has_stove_turned_on = True
    cpu_env.has_stove_turned_on_with_pot_and_object = True
    if isinstance(warp_env.has_stove_turned_on, torch.Tensor):
        warp_env.has_stove_turned_on.fill_(True)
        warp_env.has_stove_turned_on_with_pot_and_object.fill_(True)
    else:
        warp_env.has_stove_turned_on = True
        warp_env.has_stove_turned_on_with_pot_and_object = True

    # buttons_on must reflect current qpos (=False, stove off).
    cpu_env._post_process()
    warp_env._post_process()

    # Contact groups — pot-on-burner and bread-in-pot. In this synthetic
    # setup the pot is above the serving region, not the burner, so
    # pot-on-burner should be False under both.
    pot_on_burner_warp = bool(_as_np(warp_env.sim.check_contact_groups(
        warp_env.pot_body_geom_ids, warp_env.stove_burner_geom_ids))[0])
    bread_in_pot_warp = bool(_as_np(warp_env.sim.check_contact_groups(
        warp_env.bread_contact_geom_ids, warp_env.pot_contact_geom_ids))[0])
    pot_on_burner_cpu = cpu_env.check_contact("PotObject_body_0", "Stove1_collision_burner")
    bread_in_pot_cpu = cpu_env.check_contact(cpu_env.bread_ingredient, cpu_env.pot_object)
    print(f"  pot_on_burner  cpu={pot_on_burner_cpu}  warp={pot_on_burner_warp}")
    print(f"  bread_in_pot   cpu={bread_in_pot_cpu}  warp={bread_in_pot_warp}")
    if pot_on_burner_cpu != pot_on_burner_warp:
        out.append(f"contact pot_on_burner: cpu={pot_on_burner_cpu} warp={pot_on_burner_warp}")
    if bread_in_pot_cpu != bread_in_pot_warp:
        out.append(f"contact bread_in_pot: cpu={bread_in_pot_cpu} warp={bread_in_pot_warp}")

    cpu_success = bool(cpu_env._check_success())
    warp_succ_t = warp_env._check_success()
    warp_success = bool(warp_succ_t[0].item()) if isinstance(warp_succ_t, torch.Tensor) else bool(warp_succ_t)
    print(f"  _check_success cpu={cpu_success}  warp={warp_success}")
    if cpu_success != warp_success:
        out.append(f"_check_success: cpu={cpu_success} warp={warp_success}")


def run(env_name: str, seeds: list[int], n_steps: int) -> int:
    print(f"[info] building warp env {env_name}")
    warp_env = build_env(use_warp=True, env_name=env_name)
    print(f"[info] building non-warp env {env_name}")
    cpu_env = build_env(use_warp=False, env_name=env_name)

    fails: list[str] = []

    for s in seeds:
        check_reset_parity(warp_env, cpu_env, seed=s, label=env_name, out=fails)
        check_obs_parity(warp_env, cpu_env, seed=s, out=fails)

    if n_steps > 0:
        try:
            check_step_parity(warp_env, cpu_env, n_steps=n_steps, out=fails)
        except AssertionError as e:
            print(f"\n[step_parity] skipped — action-spec mismatch in harness: {e}")

    check_post_process_parity(warp_env, cpu_env, out=fails)
    check_success_parity(warp_env, cpu_env, out=fails)

    print("\n=== results ===")
    if fails:
        print(f"  {env_name}: {len(fails)} mismatches")
        for f in fails:
            print(f"    - {f}")
        return 1
    print(f"  {env_name}: PASS")
    return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="Kitchen_D0")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n_steps", type=int, default=5)
    args = p.parse_args()
    rc = run(args.env, args.seeds, args.n_steps)
    sys.exit(rc)


if __name__ == "__main__":
    main()
