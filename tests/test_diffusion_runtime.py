import json

from rl_mimicgen.diffusion_runtime import (
    apply_runtime_profile_to_diffusion_payload,
    apply_runtime_profile_to_online_rl_config,
)
from rl_mimicgen.rl.config import DiffusionConfig


def test_apply_runtime_profile_to_diffusion_payload_updates_diffusion_policy() -> None:
    payload = {
        "algo_name": "diffusion_policy",
        "algo": {
            "ddpm": {"enabled": True, "num_inference_timesteps": 100},
            "ddim": {"enabled": False, "num_inference_timesteps": 10},
            "ema": {"enabled": True},
            "horizon": {"action_horizon": 8},
        },
        "experiment": {"name": "coffee_d0_low_dim"},
    }

    updated = apply_runtime_profile_to_diffusion_payload(payload, "dppo_ddim5_ft5_act8")

    assert updated["algo"]["ddpm"]["enabled"] is False
    assert updated["algo"]["ddim"]["enabled"] is True
    assert updated["algo"]["ddim"]["num_inference_timesteps"] == 5
    assert updated["algo"]["ema"]["enabled"] is False
    assert updated["algo"]["horizon"]["action_horizon"] == 8
    assert updated["experiment"]["name"].endswith("_dppo_ddim5_ft5_act8")


def test_apply_runtime_profile_to_diffusion_payload_ignores_non_diffusion_algo() -> None:
    payload = {
        "algo_name": "bc",
        "algo": {"ema": {"enabled": True}},
        "experiment": {"name": "coffee_d0_low_dim"},
    }
    original = json.loads(json.dumps(payload))

    updated = apply_runtime_profile_to_diffusion_payload(payload, "dppo_ddim5_ft5_act8")

    assert updated == original


def test_apply_runtime_profile_to_online_rl_config_overrides_runtime_fields() -> None:
    config = DiffusionConfig(
        enabled=True,
        runtime_profile="bridge_ddpm100_ema_act8",
        num_inference_timesteps=None,
        use_ema=False,
        ft_denoising_steps=5,
        use_ddim=True,
        ddim_steps=5,
        gamma_denoising=0.9,
        min_sampling_denoising_std=0.01,
        min_logprob_denoising_std=0.01,
        act_steps=4,
    )

    apply_runtime_profile_to_online_rl_config(config)

    assert config.use_ddim is False
    assert config.num_inference_timesteps == 100
    assert config.ddim_steps is None
    assert config.use_ema is True
    assert config.ft_denoising_steps is None
    assert config.gamma_denoising == 1.0
    assert config.min_sampling_denoising_std is None
    assert config.min_logprob_denoising_std is None
    assert config.act_steps == 8
