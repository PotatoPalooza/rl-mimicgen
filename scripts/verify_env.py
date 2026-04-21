"""Smoke-test a mimicgen/robosuite environment under MuJoCo-Warp.

Constructs the env with ``use_warp=True, num_envs=N`` (same path used by
DPPO's ``WarpRobomimicVectorEnv`` and our RSL-RL ``RobomimicVecEnv``),
then exercises batched reset + step + per-env masked reset + success
mask shapes. Defaults to ``N=256`` because the historical Square_D1
warp rejection-sampling bug only surfaces at ``N >> 1``.

Usage
-----
    python scripts/verify_env.py --env Square_D1 --num-envs 256
    python scripts/verify_env.py --env Coffee_D0 --num-envs 1024 --steps 20
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np
import torch


def _set_gl_defaults() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")


def _make_env(env_name: str, num_envs: int, control_freq: int = 20):
    import mimicgen  # noqa: F401
    import robosuite as suite
    from robosuite.controllers import load_controller_config

    controller = load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name=env_name,
        robots="Panda",
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=control_freq,
        use_warp=True,
        num_envs=num_envs,
    )
    return env


def _zero_action(env, num_envs: int) -> torch.Tensor:
    low, high = env.action_spec
    act = ((np.asarray(low) + np.asarray(high)) * 0.5).astype(np.float32)
    return torch.as_tensor(np.broadcast_to(act, (num_envs, act.shape[0])).copy(), device="cuda")


def _assert_per_env_shape(name: str, value, num_envs: int) -> None:
    if isinstance(value, torch.Tensor):
        assert value.shape[0] == num_envs, f"{name}: expected leading dim {num_envs}, got {tuple(value.shape)}"
    elif isinstance(value, np.ndarray):
        assert value.shape[0] == num_envs, f"{name}: expected leading dim {num_envs}, got {value.shape}"
    elif isinstance(value, (list, tuple)):
        assert len(value) == num_envs, f"{name}: expected len {num_envs}, got {len(value)}"
    else:
        raise AssertionError(f"{name}: unsupported type {type(value)}")


def verify(env_name: str, num_envs: int, n_steps: int) -> None:
    print(f"[verify_env] env={env_name} num_envs={num_envs} steps={n_steps}")

    env = _make_env(env_name, num_envs)

    # 1. Batched reset
    obs = env.reset()
    assert isinstance(obs, dict), f"expected dict obs, got {type(obs)}"
    for k, v in obs.items():
        _assert_per_env_shape(f"obs[{k}]", v, num_envs)
    print(f"  reset ok — {len(obs)} obs keys, sample={next(iter(obs))}")

    # 2. Batched step
    action = _zero_action(env, num_envs)
    for i in range(n_steps):
        obs, reward, done, info = env.step(action)
        _assert_per_env_shape("obs[*]", next(iter(obs.values())), num_envs)
        _assert_per_env_shape("reward", reward, num_envs)
        # Raw robosuite `done` is a global scalar under ignore_done=True; a
        # few warp-ported envs (HammerCleanup, Kitchen) override `done` with
        # `_check_success()` which is a per-env tensor.
        if isinstance(done, torch.Tensor):
            _assert_per_env_shape("done", done, num_envs)
        else:
            assert isinstance(done, (bool, np.bool_)), f"done: expected scalar bool or per-env tensor, got {type(done)}"
    print(f"  step x{n_steps} ok — reward.mean={float(torch.as_tensor(reward).float().mean()):.4f}")

    # 3. Success mask — must be per-env in warp mode
    succ = env._check_success()
    _assert_per_env_shape("_check_success", succ, num_envs)
    print(f"  _check_success ok — {int(torch.as_tensor(succ).sum())} / {num_envs} succeeded")

    # 4. Per-env masked reset (hits the same _reset_internal + _reset_env_mask
    # path that broke under Square_D1 batched rejection sampling).
    from robosuite.utils.binding_utils import MjSimWarp
    sim = env.sim
    assert isinstance(sim, MjSimWarp), "env.sim is not MjSimWarp — warp branch not active"
    mask = torch.zeros(num_envs, dtype=torch.bool, device="cuda")
    mask[::2] = True
    env._reset_env_mask = mask
    try:
        env._reset_internal()
    finally:
        env._reset_env_mask = None
    print(f"  masked _reset_internal ok — {int(mask.sum())} envs reset")

    env.close()
    print("[verify_env] OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="Square_D1", help="robosuite env id (e.g. Square_D0, Coffee_D1)")
    parser.add_argument("--num-envs", type=int, default=256, help="parallel env count (default: 256)")
    parser.add_argument("--steps", type=int, default=5, help="step iterations after reset")
    args = parser.parse_args()

    _set_gl_defaults()
    try:
        verify(args.env, args.num_envs, args.steps)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
