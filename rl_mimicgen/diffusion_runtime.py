from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DiffusionRuntimeProfile:
    name: str
    use_ddim: bool
    num_inference_timesteps: int | None
    ddim_steps: int | None
    use_ema: bool
    ft_denoising_steps: int | None
    gamma_denoising: float
    min_sampling_denoising_std: float | None
    min_logprob_denoising_std: float | None
    act_steps: int | None


RUNTIME_PROFILES: dict[str, DiffusionRuntimeProfile] = {
    "robomimic_ddpm100_ema": DiffusionRuntimeProfile(
        name="robomimic_ddpm100_ema",
        use_ddim=False,
        num_inference_timesteps=100,
        ddim_steps=None,
        use_ema=True,
        ft_denoising_steps=None,
        gamma_denoising=1.0,
        min_sampling_denoising_std=None,
        min_logprob_denoising_std=None,
        act_steps=8,
    ),
    "bridge_ddpm100_ema_act8": DiffusionRuntimeProfile(
        name="bridge_ddpm100_ema_act8",
        use_ddim=False,
        num_inference_timesteps=100,
        ddim_steps=None,
        use_ema=True,
        ft_denoising_steps=None,
        gamma_denoising=1.0,
        min_sampling_denoising_std=None,
        min_logprob_denoising_std=None,
        act_steps=8,
    ),
    "dppo_ddim5_ft5_act8": DiffusionRuntimeProfile(
        name="dppo_ddim5_ft5_act8",
        use_ddim=True,
        num_inference_timesteps=None,
        ddim_steps=5,
        use_ema=False,
        ft_denoising_steps=5,
        gamma_denoising=0.99,
        min_sampling_denoising_std=0.01,
        min_logprob_denoising_std=0.01,
        act_steps=8,
    ),
}


def available_runtime_profiles() -> tuple[str, ...]:
    return tuple(sorted(RUNTIME_PROFILES))


def get_runtime_profile(name: str) -> DiffusionRuntimeProfile:
    try:
        return RUNTIME_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(RUNTIME_PROFILES))
        raise ValueError(f"Unknown diffusion runtime profile {name!r}. Choose from: {choices}.") from exc


def apply_runtime_profile_to_diffusion_payload(payload: dict[str, Any], profile_name: str) -> dict[str, Any]:
    if payload.get("algo_name") != "diffusion_policy":
        return payload

    profile = get_runtime_profile(profile_name)
    algo = payload.setdefault("algo", {})
    horizon = algo.setdefault("horizon", {})
    ddpm = algo.setdefault("ddpm", {})
    ddim = algo.setdefault("ddim", {})
    ema = algo.setdefault("ema", {})

    if profile.use_ddim:
        ddpm["enabled"] = False
        ddim["enabled"] = True
        if profile.ddim_steps is not None:
            ddim["num_inference_timesteps"] = int(profile.ddim_steps)
    else:
        ddpm["enabled"] = True
        ddim["enabled"] = False
        if profile.num_inference_timesteps is not None:
            ddpm["num_inference_timesteps"] = int(profile.num_inference_timesteps)

    ema["enabled"] = bool(profile.use_ema)
    if profile.act_steps is not None:
        horizon["action_horizon"] = int(profile.act_steps)

    experiment = payload.setdefault("experiment", {})
    experiment_name = experiment.get("name")
    suffix = profile.name.replace("-", "_")
    if isinstance(experiment_name, str) and suffix not in experiment_name:
        experiment["name"] = f"{experiment_name}_{suffix}"

    return payload


def apply_runtime_profile_to_online_rl_config(diffusion_cfg: Any) -> Any:
    profile_name = getattr(diffusion_cfg, "runtime_profile", None)
    if not profile_name:
        return diffusion_cfg
    profile = get_runtime_profile(profile_name)
    diffusion_cfg.use_ddim = bool(profile.use_ddim)
    diffusion_cfg.num_inference_timesteps = profile.num_inference_timesteps
    diffusion_cfg.ddim_steps = profile.ddim_steps
    diffusion_cfg.use_ema = bool(profile.use_ema)
    diffusion_cfg.ft_denoising_steps = profile.ft_denoising_steps
    diffusion_cfg.gamma_denoising = float(profile.gamma_denoising)
    diffusion_cfg.min_sampling_denoising_std = profile.min_sampling_denoising_std
    diffusion_cfg.min_logprob_denoising_std = profile.min_logprob_denoising_std
    diffusion_cfg.act_steps = profile.act_steps
    return diffusion_cfg
