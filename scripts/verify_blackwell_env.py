import torch
import robosuite as suite
import mimicgen
from robosuite.controllers import load_controller_config

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
    has_offscreen_renderer=False,
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

print("verification", "ok", "action_dim", low.shape[0], "done", done)
env.close()