from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from rl_mimicgen.dppo.config.schema import DPPORunConfig
from rl_mimicgen.dppo.data import DPPODatasetBundle
from rl_mimicgen.dppo.model import DiffusionModel


@dataclass(slots=True)
class DPPORolloutSample:
    normalized_actions: np.ndarray
    actions: np.ndarray
    history: np.ndarray


class DiffusionPolicyAdapter:
    def __init__(
        self,
        config: DPPORunConfig,
        bundle: DPPODatasetBundle,
        checkpoint_path: str,
        deterministic: bool = True,
    ) -> None:
        self.config = config
        self.bundle = bundle
        self.device = torch.device(config.device)
        self.deterministic = deterministic
        self.model = DiffusionModel(
            obs_dim=bundle.obs_dim,
            action_dim=bundle.action_dim,
            horizon_steps=config.diffusion.horizon_steps,
            cond_steps=config.diffusion.cond_steps,
            denoising_steps=config.diffusion.denoising_steps,
            predict_epsilon=config.diffusion.predict_epsilon,
            denoised_clip_value=config.diffusion.denoised_clip_value,
            time_dim=config.diffusion.time_dim,
            mlp_dims=config.diffusion.mlp_dims,
            residual_style=config.diffusion.residual_style,
            device=config.device,
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"]
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        allowed_missing = {
            "ddpm_logvar_clipped",
            "ddpm_mu_coef1",
            "ddpm_mu_coef2",
        }
        unexpected = set(unexpected_keys)
        missing = set(missing_keys)
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys: {sorted(unexpected)}")
        disallowed_missing = missing - allowed_missing
        if disallowed_missing:
            raise RuntimeError(f"Missing required checkpoint keys: {sorted(disallowed_missing)}")
        self.model.eval()
        self.obs_history: deque[np.ndarray] = deque(maxlen=config.diffusion.cond_steps)

    def reset(self, obs: dict[str, np.ndarray]) -> None:
        flat_obs = self.bundle.flatten_obs(obs)
        normalized_obs = self.bundle.normalization.normalize_obs(flat_obs).astype(np.float32, copy=False)
        self.obs_history.clear()
        for _ in range(self.config.diffusion.cond_steps):
            self.obs_history.append(normalized_obs.copy())

    def sample(self, obs: dict[str, np.ndarray]) -> DPPORolloutSample:
        flat_obs = self.bundle.flatten_obs(obs)
        normalized_obs = self.bundle.normalization.normalize_obs(flat_obs).astype(np.float32, copy=False)
        if not self.obs_history:
            self.reset(obs)
        else:
            self.obs_history.append(normalized_obs.copy())
        history = np.stack(list(self.obs_history), axis=0)
        state_tensor = torch.as_tensor(history[None], dtype=torch.float32, device=self.device)
        sample = self.model({"state": state_tensor}, deterministic=self.deterministic)
        normalized_actions = sample.trajectories[0].detach().cpu().numpy()
        actions = self.bundle.normalization.unnormalize_action(normalized_actions)
        return DPPORolloutSample(
            normalized_actions=normalized_actions,
            actions=actions.astype(np.float32, copy=False),
            history=history,
        )
