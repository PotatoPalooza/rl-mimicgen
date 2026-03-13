# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

gym.register(
    id="Franka-Coffee-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_cfg:FrankaCoffeeEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{__name__}:franka_bc.json",
        "rsl_rl_cfg_entry_point": f"{__name__}.franka_ppo_cfg:FrankaCoffeePPORunnerCfg",
    },
    disable_env_checker=True,
)
