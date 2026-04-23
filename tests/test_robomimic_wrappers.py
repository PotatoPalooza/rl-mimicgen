from __future__ import annotations

import numpy as np

from dppo.env.gym_utils.wrapper.robomimic_image import RobomimicImageWrapper
from dppo.env.gym_utils.wrapper.robomimic_lowdim import RobomimicLowdimWrapper


class _FakeLowdimEnv:
    def __init__(self) -> None:
        self.action_dimension = 2
        self.last_action = None

    def get_observation(self):
        return {"state0": np.array([0.25, -0.5], dtype=np.float32)}

    def reset(self):
        return self.get_observation()

    def step(self, action):
        self.last_action = np.asarray(action)
        return self.get_observation(), 1.0, False, {}

    def render(self, **kwargs):
        return np.zeros((4, 4, 3), dtype=np.uint8)


class _FakeImageEnv:
    def __init__(self) -> None:
        self.action_dimension = 2
        self.last_action = None

    def _obs(self):
        return {
            "state0": np.array([0.1, -0.2], dtype=np.float32),
            "agentview_image": np.zeros((3, 4, 4), dtype=np.float32),
        }

    def reset(self):
        return self._obs()

    def step(self, action):
        self.last_action = np.asarray(action)
        return self._obs(), 1.0, False, {}

    def render(self, **kwargs):
        return np.zeros((4, 4, 3), dtype=np.uint8)


def _write_normalization(path) -> None:
    np.savez(
        path,
        obs_min=np.array([-1.0, -1.0], dtype=np.float32),
        obs_max=np.array([1.0, 1.0], dtype=np.float32),
        action_min=np.array([-2.0, -4.0], dtype=np.float32),
        action_max=np.array([2.0, 4.0], dtype=np.float32),
    )


def test_lowdim_wrapper_clips_and_unnormalizes_actions(tmp_path) -> None:
    normalization_path = tmp_path / "normalization.npz"
    _write_normalization(normalization_path)
    env = _FakeLowdimEnv()
    wrapper = RobomimicLowdimWrapper(
        env,
        normalization_path=normalization_path.as_posix(),
        low_dim_keys=["state0"],
    )

    assert wrapper.action_space.dtype == np.float32

    wrapper.step(np.array([1.5, -2.0], dtype=np.float32))

    np.testing.assert_allclose(env.last_action, np.array([2.0, -4.0], dtype=np.float32))


def test_lowdim_wrapper_rejects_nonfinite_actions(tmp_path) -> None:
    normalization_path = tmp_path / "normalization.npz"
    _write_normalization(normalization_path)
    wrapper = RobomimicLowdimWrapper(
        _FakeLowdimEnv(),
        normalization_path=normalization_path.as_posix(),
        low_dim_keys=["state0"],
    )

    try:
        wrapper.step(np.array([np.nan, 0.0], dtype=np.float32))
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("Expected non-finite action values to be rejected")


def test_image_wrapper_clips_and_unnormalizes_actions(tmp_path) -> None:
    normalization_path = tmp_path / "normalization.npz"
    _write_normalization(normalization_path)
    env = _FakeImageEnv()
    wrapper = RobomimicImageWrapper(
        env,
        shape_meta={
            "obs": {
                "rgb": {"shape": (3, 4, 4)},
                "state": {"shape": (2,)},
            }
        },
        normalization_path=normalization_path.as_posix(),
        low_dim_keys=["state0"],
        image_keys=["agentview_image"],
    )

    assert wrapper.action_space.dtype == np.float32

    wrapper.step(np.array([-3.0, 0.5], dtype=np.float32))

    np.testing.assert_allclose(env.last_action, np.array([-2.0, 2.0], dtype=np.float32))
