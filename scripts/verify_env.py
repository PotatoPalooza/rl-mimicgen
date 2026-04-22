"""Verify a warp-backed robosuite/mimicgen env is fit for training.

Runs a battery of checks, each reported as one line (``[ok|warn|fail]``).
The script exits non-zero if any check fails; warnings are surfaced but
do not fail the run.

Checks
------
1.  CPU XML load (`use_warp=False`). Catches MuJoCo XML validation
    regressions like the Sawyer `mesh volume is too small` bug that
    shipped to hcrl2 because the old smoke hardcoded `robots="Panda"`.
2.  Warp env construction.
3.  Dense qM (`opt.jacobian == mjJAC_DENSE`). OSC's `qM_full` slicing
    breaks silently under sparse `qM` at `nv > 32`.
4.  Obs invariants after reset: leading dim `== num_envs`, rank `>= 2`,
    any `*_quat` obs has `w >= 0` (mat2quat_torch sign convention).
5.  No NaN/Inf in reset obs or sim `qpos/qvel/xpos/xquat`.
6.  Per-env uniqueness for mocap-converted fixtures (Kitchen_D1 stove,
    MugCleanup drawer, HammerCleanup cabinet). Caught the "disappearing
    stove" regression.
7.  Random-action rollout (scale configurable): NaN scan per step, peak
    `nacon / naconmax` and `nefc / njmax` headroom, peak `|qvel|`.
    Silent contact-buffer overflow picks a different f32 kernel.
8.  Settle check: zero action for K steps, then `|qvel|` should be small.
9.  Masked `_reset_internal`: unmasked rows' qpos must be bit-unchanged.
10. Optional `--parity`: build a single-env CPU sim with the same
    robot, reset both, compare `robot0_eef_pos` within eps.

Usage
-----
    python scripts/verify_env.py --env Square_D1 --num-envs 256
    python scripts/verify_env.py --env PickPlace_D0 --robot Sawyer
    python scripts/verify_env.py --env Coffee_D1 --env-meta path/to/env_meta.json
    python scripts/verify_env.py --env Square_D0 --parity
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Callable

import numpy as np
import torch


def _set_gl_defaults() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")


def _load_env_meta(path: str) -> dict[str, Any]:
    with open(path) as f:
        meta = json.load(f)
    kw = meta.get("env_kwargs") or meta.get("config", {}).get("env_kwargs") or {}
    robots = kw.get("robots") or ["Panda"]
    robot = robots[0] if isinstance(robots, (list, tuple)) else robots
    env_name = meta.get("env_name") or kw.get("env_name")
    return {"env_name": env_name, "robot": robot, "kwargs": kw}


def _make_cpu_env(env_name: str, robot: str, control_freq: int):
    import mimicgen  # noqa: F401
    import robosuite as suite
    from robosuite.controllers import load_controller_config

    controller = load_controller_config(default_controller="OSC_POSE")
    return suite.make(
        env_name=env_name,
        robots=robot,
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=control_freq,
        use_warp=False,
    )


def _make_warp_env(env_name: str, robot: str, num_envs: int, control_freq: int):
    import mimicgen  # noqa: F401
    import robosuite as suite
    from robosuite.controllers import load_controller_config

    controller = load_controller_config(default_controller="OSC_POSE")
    return suite.make(
        env_name=env_name,
        robots=robot,
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=control_freq,
        use_warp=True,
        num_envs=num_envs,
    )


# Reporting

_STATUS_TAGS = {"ok": "  [ok]  ", "warn": " [warn] ", "fail": " [FAIL] "}


class _Report:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str, str]] = []

    def record(self, name: str, status: str, msg: str = "") -> None:
        assert status in _STATUS_TAGS
        self.entries.append((name, status, msg))
        print(f"{_STATUS_TAGS[status]} {name}" + (f" — {msg}" if msg else ""))

    def summarize(self) -> int:
        n_ok = sum(1 for _, s, _ in self.entries if s == "ok")
        n_warn = sum(1 for _, s, _ in self.entries if s == "warn")
        n_fail = sum(1 for _, s, _ in self.entries if s == "fail")
        print()
        print(f"[verify_env] {n_ok} ok, {n_warn} warn, {n_fail} fail")
        return 1 if n_fail else 0


# Helpers — warp-data reads

def _wp_to_torch(arr):
    import warp as wp
    return wp.to_torch(arr)


def _read_buffer_counts(sim) -> tuple[int, int, int, int]:
    """Return (nacon, naconmax, peak_nefc_per_world, njmax_per_world)."""
    nacon_arr = sim._warp_data.nacon
    nacon = int(_wp_to_torch(nacon_arr).sum().item()) if hasattr(nacon_arr, "numpy") else int(nacon_arr)
    naconmax = int(sim._warp_data.naconmax)
    nefc = _wp_to_torch(sim._warp_data.nefc)  # (nworld,)
    njmax = int(sim._warp_data.njmax)
    return nacon, naconmax, int(nefc.max().item()), njmax


def _check_no_nan_sim(sim, ctx: str) -> None:
    for name in ("qpos", "qvel", "xpos", "xquat"):
        t = _wp_to_torch(getattr(sim._warp_data, name))
        if not torch.isfinite(t).all():
            bad = (~torch.isfinite(t).reshape(t.shape[0], -1).all(dim=1)).nonzero(as_tuple=True)[0]
            raise AssertionError(f"{ctx}: {name} has NaN/Inf in envs {bad[:5].tolist()}... ({bad.numel()} total)")


def _check_no_nan_obs(obs: dict, ctx: str) -> None:
    for k, v in obs.items():
        t = v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))
        if not torch.isfinite(t).all():
            raise AssertionError(f"{ctx}: obs[{k}] has NaN/Inf")


# Check implementations

def _check_obs_shapes_and_rank(obs: dict, num_envs: int) -> None:
    for k, v in obs.items():
        shape = tuple(v.shape) if hasattr(v, "shape") else (len(v),)
        if len(shape) < 2:
            raise AssertionError(f"obs[{k}] rank={len(shape)} < 2 (need (N, ...); CLAUDE.md warp-stub invariant)")
        if shape[0] != num_envs:
            raise AssertionError(f"obs[{k}] leading dim={shape[0]} != num_envs={num_envs}")


def _check_quat_w_sign(obs: dict) -> list[str]:
    """Report obs keys with any `w < 0` (xyzw convention).

    The positive-w convention is enforced at ``mat2quat_torch`` output, but
    downstream quat multiplies in obs pipelines can legitimately reintroduce
    negative w on both CPU and warp. This surfaces as a *warning* only —
    the real CPU↔warp divergence bug is caught by ``--parity``.
    """
    bad: list[str] = []
    for k, v in obs.items():
        if not k.endswith("_quat"):
            continue
        if not hasattr(v, "shape") or len(v.shape) < 1 or v.shape[-1] != 4:
            continue
        w = v[..., -1]
        w_t = w if torch.is_tensor(w) else torch.as_tensor(np.asarray(w))
        if (w_t < 0).any().item():
            bad.append(k)
    return bad


def _check_dense_jacobian(sim) -> None:
    # Checked on the underlying CPU MjModel — MjSimWarp.__init__ sets
    # jacobian to DENSE here before put_model, and the warp Option struct
    # does not mirror `jacobian` as a field.
    import mujoco as _mj
    jac_v = int(sim.model._model.opt.jacobian)
    expected = int(_mj.mjtJacobian.mjJAC_DENSE)
    if jac_v != expected:
        raise AssertionError(f"jacobian={jac_v}, expected mjJAC_DENSE={expected}")


def _check_mocap_uniqueness(env, num_envs: int, report: _Report) -> None:
    fixtures: list[tuple[str, int]] = []
    if getattr(env, "_fixture_mocap_ids", None):
        for name, mid in env._fixture_mocap_ids.items():
            fixtures.append((name, int(mid)))
    if getattr(env, "_drawer_mocap_id", None) is not None:
        fixtures.append(("drawer", int(env._drawer_mocap_id)))
    if getattr(env, "_cabinet_mocap_id", None) is not None:
        fixtures.append(("cabinet", int(env._cabinet_mocap_id)))

    if not fixtures:
        report.record("mocap_uniqueness", "ok", "no mocap fixtures on this env")
        return
    if num_envs < 2:
        report.record("mocap_uniqueness", "warn", "num_envs<2; cannot check per-env variation")
        return

    mocap_pos = _wp_to_torch(env.sim._warp_data.mocap_pos)  # (nworld, nmocap, 3)
    bits = []
    any_degenerate = False
    for name, mid in fixtures:
        std = mocap_pos[:, mid, :].std(dim=0).max().item()
        bits.append(f"{name}:σ={std:.3g}")
        if std < 1e-6:
            any_degenerate = True
    status = "warn" if any_degenerate else "ok"
    note = "; some σ≈0 — expected for degenerate bounds (e.g. MugCleanup_D0 drawer), else a per-env fixture-rebake regression" if any_degenerate else ""
    report.record("mocap_uniqueness", status, " ".join(bits) + note)


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _action_bounds(env) -> tuple[np.ndarray, np.ndarray]:
    """Bypass ``env.action_spec`` — it np.concatenates CUDA tensors under warp and dies."""
    lo_parts: list[np.ndarray] = []
    hi_parts: list[np.ndarray] = []
    for robot in env.robots:
        lo, hi = robot.action_limits
        lo_parts.append(_to_numpy(lo).reshape(-1).astype(np.float32))
        hi_parts.append(_to_numpy(hi).reshape(-1).astype(np.float32))
    return np.concatenate(lo_parts), np.concatenate(hi_parts)


def _random_actions(env, low: np.ndarray, high: np.ndarray, scale: float, rng: np.random.Generator) -> torch.Tensor:
    a = rng.uniform(low, high, size=(env.num_envs, low.shape[0])).astype(np.float32)
    return torch.as_tensor(a * scale, device="cuda")


def _rollout_checks(env, n_steps: int, action_scale: float, report: _Report) -> None:
    sim = env.sim
    low, high = _action_bounds(env)
    rng = np.random.default_rng(0)
    peak_nacon = 0
    peak_nefc = 0
    peak_qvel = 0.0
    naconmax = njmax = 0
    for step in range(n_steps):
        action = _random_actions(env, low, high, action_scale, rng)
        obs, reward, done, info = env.step(action)
        # NaN scan on obs and sim state
        try:
            _check_no_nan_obs(obs, f"step {step}")
            _check_no_nan_sim(sim, f"step {step}")
        except AssertionError as e:
            report.record("rollout_nan", "fail", str(e).splitlines()[0])
            return
        # Buffer headroom
        nacon, naconmax_cur, nefc_peak, njmax_cur = _read_buffer_counts(sim)
        peak_nacon = max(peak_nacon, nacon)
        peak_nefc = max(peak_nefc, nefc_peak)
        naconmax = naconmax_cur
        njmax = njmax_cur
        # qvel peak
        qv = _wp_to_torch(sim._warp_data.qvel).abs().max().item()
        peak_qvel = max(peak_qvel, qv)

    report.record(
        "rollout_nan",
        "ok",
        f"no NaN/Inf over {n_steps} random steps (scale={action_scale})",
    )

    def _headroom(name: str, peak: int, cap: int) -> None:
        frac = peak / max(cap, 1)
        if cap <= 0:
            report.record(name, "warn", "cap reports as 0")
            return
        status = "fail" if frac >= 0.9 else ("warn" if frac >= 0.5 else "ok")
        report.record(name, status, f"peak {peak}/{cap} ({frac*100:.1f}%)")

    _headroom("nacon_headroom", peak_nacon, naconmax)
    _headroom("nefc_headroom", peak_nefc, njmax)

    status = "fail" if peak_qvel > 500.0 else ("warn" if peak_qvel > 50.0 else "ok")
    report.record("qvel_peak_random", status, f"|qvel|.max={peak_qvel:.3g}")


def _settle_check(env, n_steps: int, report: _Report, threshold: float = 2.0) -> None:
    # env.reset() rebuilds env.sim (base.py:290 -> _load_model -> new MjSimWarp);
    # always re-fetch sim after reset().
    env.reset()
    sim = env.sim
    action = torch.zeros(env.num_envs, env.action_dim, device="cuda", dtype=torch.float32)
    for _ in range(n_steps):
        env.step(action)
    peak = _wp_to_torch(sim._warp_data.qvel).abs().max().item()
    status = "ok" if peak < threshold else "warn"
    report.record("settle", status, f"after {n_steps} zero-action steps, |qvel|.max={peak:.3g} (threshold {threshold})")


def _masked_reset_check(env, report: _Report) -> None:
    """Exercise the slim-reset invariant: masked _reset_internal + skip_robot_reset
    leaves unmasked rows' qpos bit-unchanged (CLAUDE.md 'Masked qpos writes')."""
    num_envs = env.num_envs
    if num_envs < 2:
        report.record("masked_reset_invariance", "warn", "num_envs<2")
        return
    env.reset()
    sim = env.sim
    qpos_before = _wp_to_torch(sim._warp_data.qpos).clone()
    mask = torch.zeros(num_envs, dtype=torch.bool, device="cuda")
    mask[::2] = True
    env._reset_env_mask = mask
    env._skip_robot_reset = True
    try:
        env._reset_internal()
    finally:
        env._reset_env_mask = None
        env._skip_robot_reset = False
    qpos_after = _wp_to_torch(sim._warp_data.qpos)
    unmasked = (~mask).nonzero(as_tuple=True)[0]
    diff = (qpos_after[unmasked] - qpos_before[unmasked]).abs().max().item()
    if diff > 1e-6:
        report.record("masked_reset_invariance", "fail", f"unmasked qpos changed by max|Δ|={diff:.3g}")
    else:
        report.record(
            "masked_reset_invariance",
            "ok",
            f"{int(mask.sum())} envs reset, {int((~mask).sum())} untouched",
        )


def _parity_check(env_name: str, robot: str, control_freq: int, report: _Report) -> None:
    try:
        cpu_env = _make_cpu_env(env_name, robot, control_freq)
    except Exception as e:
        report.record("parity[build]", "warn", f"CPU env construct failed: {e}")
        return
    try:
        warp_env = _make_warp_env(env_name, robot, 1, control_freq)
    except Exception as e:
        cpu_env.close()
        report.record("parity[build]", "warn", f"warp env (n=1) construct failed: {e}")
        return
    try:
        np.random.seed(0)
        torch.manual_seed(0)
        cpu_obs = cpu_env.reset()
        np.random.seed(0)
        torch.manual_seed(0)
        warp_obs = warp_env.reset()

        # Compare keys and values within tolerance. Seeds line up on the
        # shared numpy RNG path used by robosuite placement samplers, so
        # initial robot pose should match within f32 precision.
        for key, eps in [("robot0_eef_pos", 5e-3), ("robot0_eef_quat", 2e-2)]:
            if key not in cpu_obs or key not in warp_obs:
                report.record(f"parity[{key}]", "warn", "missing on one side")
                continue
            c = np.asarray(cpu_obs[key]).reshape(-1).astype(np.float64)
            w_raw = warp_obs[key]
            w = w_raw.detach().cpu().numpy() if torch.is_tensor(w_raw) else np.asarray(w_raw)
            w = w.reshape(-1).astype(np.float64)
            # quat sign ambiguity: compare min(|c-w|, |c+w|)
            diff = np.abs(c - w).max()
            if key.endswith("_quat"):
                diff = min(diff, np.abs(c + w).max())
            status = "ok" if diff < eps else "warn"
            report.record(f"parity[{key}]", status, f"max|Δ|={diff:.3g} (eps={eps})")
    finally:
        cpu_env.close()
        warp_env.close()


# Main

def verify(args: argparse.Namespace) -> int:
    env_name = args.env
    robot = args.robot
    if args.env_meta:
        meta = _load_env_meta(args.env_meta)
        env_name = meta["env_name"] or env_name
        robot = meta["robot"]
        print(f"[verify_env] env_meta: env={env_name} robot={robot}")
    print(
        f"[verify_env] env={env_name} robot={robot} num_envs={args.num_envs} "
        f"control_freq={args.control_freq} random_steps={args.random_steps} settle_steps={args.settle_steps}"
    )
    report = _Report()

    # 1. CPU XML load
    try:
        cpu_env = _make_cpu_env(env_name, robot, args.control_freq)
        cpu_env.close()
        report.record("cpu_xml_load", "ok")
    except Exception as e:
        report.record("cpu_xml_load", "fail", str(e).splitlines()[0])
        return report.summarize()

    # 2. Warp construction
    try:
        env = _make_warp_env(env_name, robot, args.num_envs, args.control_freq)
    except Exception as e:
        report.record("warp_construct", "fail", str(e).splitlines()[0])
        return report.summarize()
    report.record("warp_construct", "ok")

    try:
        # 3. Dense qM
        try:
            _check_dense_jacobian(env.sim)
            report.record("dense_qM", "ok")
        except AssertionError as e:
            report.record("dense_qM", "fail", str(e))

        # Verify env.sim is the expected type
        from robosuite.utils.binding_utils import MjSimWarp
        if not isinstance(env.sim, MjSimWarp):
            report.record("sim_type", "fail", f"env.sim is {type(env.sim).__name__}, not MjSimWarp")
            return report.summarize()
        report.record("sim_type", "ok")

        # 4–5. Reset + obs invariants + NaN
        obs = env.reset()
        try:
            _check_obs_shapes_and_rank(obs, args.num_envs)
            report.record("obs_shape_rank", "ok", f"{len(obs)} keys, all rank>=2")
        except AssertionError as e:
            report.record("obs_shape_rank", "fail", str(e))

        bad_quat = _check_quat_w_sign(obs)
        n_quat = sum(1 for k in obs if k.endswith("_quat"))
        if bad_quat:
            report.record(
                "quat_w_positive",
                "warn",
                f"{len(bad_quat)}/{n_quat} quat keys have w<0 (diagnostic; use --parity for CPU/warp divergence check): {bad_quat}",
            )
        else:
            report.record("quat_w_positive", "ok", f"{n_quat} quat keys all w>=0")

        try:
            _check_no_nan_obs(obs, "reset")
            _check_no_nan_sim(env.sim, "reset")
            report.record("no_nan_reset", "ok")
        except AssertionError as e:
            report.record("no_nan_reset", "fail", str(e).splitlines()[0])

        # 6. Mocap fixture per-env uniqueness
        _check_mocap_uniqueness(env, args.num_envs, report)

        # 7. Random-action rollout
        _rollout_checks(env, args.random_steps, args.action_scale, report)

        # 8. Settle
        _settle_check(env, args.settle_steps, report, threshold=args.settle_threshold)

        # 9. Masked reset invariance
        _masked_reset_check(env, report)

        # Success mask shape (kept from old smoke — cheap and useful)
        succ = env._check_success()
        try:
            t = succ if torch.is_tensor(succ) else torch.as_tensor(np.asarray(succ))
            if t.shape[0] != args.num_envs:
                raise AssertionError(f"shape[0]={t.shape[0]} != {args.num_envs}")
            report.record("check_success_shape", "ok", f"{int(t.sum())} / {args.num_envs} succeeded")
        except Exception as e:
            report.record("check_success_shape", "fail", str(e))
    finally:
        env.close()

    # 10. Optional parity
    if args.parity:
        _parity_check(env_name, robot, args.control_freq, report)

    return report.summarize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default="Square_D1", help="robosuite env id (e.g. Square_D0, Coffee_D1)")
    parser.add_argument("--robot", default="Panda", help="robot name (Panda, Sawyer, IIWA, ...)")
    parser.add_argument("--env-meta", default=None, help="optional env_meta.json (overrides --env/--robot)")
    parser.add_argument("--num-envs", type=int, default=256, help="parallel env count (default: 256)")
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--random-steps", type=int, default=20, help="steps of random-action rollout")
    parser.add_argument("--action-scale", type=float, default=0.5, help="random action scale (0..1 of action bounds)")
    parser.add_argument("--settle-steps", type=int, default=30, help="zero-action steps before settle qvel check")
    parser.add_argument("--settle-threshold", type=float, default=2.0, help="max |qvel| after settle")
    parser.add_argument("--parity", action="store_true", help="also run CPU↔warp parity check (single-env CPU sim)")
    args = parser.parse_args()

    _set_gl_defaults()
    try:
        return verify(args)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
