"""Smoke-test the Blackwell CUDA + MuJoCo-Warp + robosuite + video stack.

Checks: torch/CUDA, mimicgen env registration, a single-env reset/step,
mujoco-warp ``put_model``/``put_data`` round-trip, and an end-to-end
offscreen render → mp4 writer pass with the same codec/pixfmt settings
``WarpRobomimicVectorEnv._open_video_writer`` pins for wandb.
"""

import os
import tempfile

import imageio
import imageio_ffmpeg
import mimicgen  # noqa: F401
import mujoco_warp as mjw
import numpy as np
import robosuite as suite
import torch
from robosuite.controllers import load_controller_config

os.environ.setdefault("MUJOCO_GL", "egl")

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("compiled_arches", sorted(torch.cuda.get_arch_list()))
print("cuda_available", torch.cuda.is_available())
print("device0", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

assert "Coffee_D0" in suite.ALL_ENVIRONMENTS
print("registered_envs", len(suite.ALL_ENVIRONMENTS))

cfg = load_controller_config(default_controller="OSC_POSE")
env = suite.make(
    env_name="Coffee_D0",
    robots="Panda",
    controller_configs=cfg,
    has_renderer=False,
    has_offscreen_renderer=True,
    ignore_done=True,
    use_camera_obs=False,
    control_freq=20,
)

obs = env.reset()
low, high = env.action_spec
obs, reward, done, info = env.step((low + high) / 2.0)

assert isinstance(obs, dict)
assert isinstance(reward, float)
assert isinstance(info, dict)

m = mjw.put_model(env.sim.model._model)
d = mjw.put_data(env.sim.model._model, env.sim.data._data)

assert m is not None
assert d is not None

# Offscreen render + mp4 writer end-to-end. Mirrors
# WarpRobomimicVectorEnv._open_video_writer so if imageio-ffmpeg / the
# container's ffmpeg stops handling libx264 + yuv420p + faststart, this
# catches it before a run silently produces unplayable wandb clips.
width, height, n_frames = 128, 128, 8
frames = []
for _ in range(n_frames):
    obs, _, _, _ = env.step((low + high) / 2.0)
    frame = env.sim.render(width=width, height=height, camera_name="frontview")
    assert frame.shape == (height, width, 3), f"unexpected frame shape {frame.shape}"
    frames.append(np.ascontiguousarray(frame[::-1]))

video_path = os.path.join(tempfile.gettempdir(), "verify_blackwell_env.mp4")
if os.path.exists(video_path):
    os.remove(video_path)

writer = imageio.get_writer(
    video_path,
    fps=30,
    codec="libx264",
    pixelformat="yuv420p",
    macro_block_size=16,
    ffmpeg_params=["-movflags", "+faststart"],
)
for fr in frames:
    writer.append_data(fr)
writer.close()  # flushes moov atom

size = os.path.getsize(video_path)
assert size > 0, f"empty mp4 at {video_path}"

# Read it back — a corrupt / unfinalized mp4 would fail to open.
with imageio.get_reader(video_path, "ffmpeg") as reader:
    decoded = [reader.get_data(i) for i in range(n_frames)]
assert len(decoded) == n_frames, f"decoded {len(decoded)}/{n_frames} frames"
assert decoded[0].shape[:2] == (height, width)

print(
    "video_ok",
    "path", video_path,
    "bytes", size,
    "ffmpeg", imageio_ffmpeg.get_ffmpeg_version(),
)
print("verification", "ok", "action_dim", low.shape[0], "done", done)
env.close()
