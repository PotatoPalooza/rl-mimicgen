"""Distribution modules that match robomimic BC policy heads for seamless warm-start.

Two classes, one per BC head variant:

* :class:`GMMDistribution` — mirrors ``RNNGMMActorNetwork``'s Mixture-of-Gaussians
  head (see ``robomimic.models.policy_nets.RNNGMMActorNetwork.forward_train``).
  The MLP outputs a flat vector of size ``2 * num_modes * ac_dim + num_modes``
  which is split into means, scales, and mixing logits. Means are tanh-squashed
  (unless ``use_tanh``), scales go through softplus+min_std. This layout is
  chosen so BC's three separate decoder heads (``decoder.nets.{mean,scale,
  logits}``) can be stacked into the single final Linear layer RSL-RL's MLP
  produces.

* :class:`TanhGaussianDistribution` — a :class:`GaussianDistribution` whose mean
  is tanh-squashed. Matches BC-RNN (non-GMM), which hardcodes
  ``actions = torch.tanh(decoder_output)`` in ``RNNActorNetwork.forward``. No
  Jacobian correction in ``log_prob`` — matches BC's own (slightly loose)
  treatment, which is what we want for warm-start equivalence.
"""

from __future__ import annotations

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules.distribution import Distribution, GaussianDistribution


class GMMDistribution(Distribution):
    """Mixture-of-Gaussians distribution matching robomimic BC-RNN-GMM."""

    _STD_ACTIVATIONS = {
        "softplus": F.softplus,
        "exp": torch.exp,
    }

    def __init__(
        self,
        output_dim: int,
        num_modes: int = 5,
        min_std: float = 1e-4,
        std_activation: str = "softplus",
        low_noise_eval: bool = True,
        use_tanh: bool = False,
    ) -> None:
        """Initialize the GMM distribution.

        Args:
            output_dim: Action dimension.
            num_modes: Number of Gaussian mixture components.
            min_std: Minimum std added to the softplus/exp output.
            std_activation: ``"softplus"`` or ``"exp"``.
            low_noise_eval: If True, clamp component std to 1e-4 in eval mode
                (matches BC's low-noise-eval behavior; no effect during training).
            use_tanh: If True, do not tanh-squash the means here (BC's
                ``use_tanh=True`` path wraps the mixture in a tanh-transformed
                distribution instead — unsupported; falls back to no squash).
        """
        super().__init__(output_dim)
        if std_activation not in self._STD_ACTIVATIONS:
            raise ValueError(
                f"std_activation must be one of {list(self._STD_ACTIVATIONS)}, got {std_activation!r}"
            )
        self.num_modes = int(num_modes)
        self.min_std = float(min_std)
        self.std_activation = std_activation
        self.low_noise_eval = bool(low_noise_eval)
        self.use_tanh = bool(use_tanh)
        if self.use_tanh:
            # BC wraps the mixture in TanhWrappedDistribution; we don't replicate
            # that here because it'd require a reparameterizable mixture sampler.
            # Leaving means unsquashed is closer to "raw linear output" than
            # applying a mismatched tanh, so we warn loudly.
            import warnings
            warnings.warn(
                "GMMDistribution(use_tanh=True) is approximated: mean is not "
                "tanh-squashed and no Jacobian correction is applied in log_prob. "
                "Warm-starting from a tanh-wrapped BC policy is lossy.",
                RuntimeWarning,
            )

        # Populated by update(); stored for params / entropy / KL access.
        self._distribution: D.MixtureSameFamily | None = None
        self._component_means: torch.Tensor | None = None  # (..., num_modes, ac_dim)
        self._component_scales: torch.Tensor | None = None
        self._logits: torch.Tensor | None = None  # (..., num_modes)
        self._entropy_cache: torch.Tensor | None = None

    @property
    def input_dim(self) -> int:
        return 2 * self.num_modes * self.output_dim + self.num_modes

    def _split(self, mlp_output: torch.Tensor):
        """Slice the flat MLP output into (mean, scale, logits) with BC's post-processing."""
        n, d = self.num_modes, self.output_dim
        batch_shape = mlp_output.shape[:-1]
        means = mlp_output[..., : n * d].reshape(*batch_shape, n, d)
        scales_raw = mlp_output[..., n * d : 2 * n * d].reshape(*batch_shape, n, d)
        logits = mlp_output[..., 2 * n * d :]

        if not self.use_tanh:
            means = torch.tanh(means)

        if self.low_noise_eval and not self.training:
            scales = torch.full_like(scales_raw, 1e-4)
        else:
            scales = self._STD_ACTIVATIONS[self.std_activation](scales_raw) + self.min_std
        return means, scales, logits

    def update(self, mlp_output: torch.Tensor) -> None:
        means, scales, logits = self._split(mlp_output)
        self._component_means = means
        self._component_scales = scales
        self._logits = logits
        component = D.Independent(D.Normal(loc=means, scale=scales), 1)
        mixture = D.Categorical(logits=logits)
        self._distribution = D.MixtureSameFamily(
            mixture_distribution=mixture, component_distribution=component
        )
        self._entropy_cache = None  # invalidate MC estimate

    def sample(self) -> torch.Tensor:
        return self._distribution.sample()  # type: ignore

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Argmax-mode's mean — matches BC's ``low_noise_eval`` behavior."""
        means, _, logits = self._split(mlp_output)
        idx = logits.argmax(dim=-1)  # (*batch,)
        idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(*idx.shape, 1, self.output_dim)
        return torch.gather(means, -2, idx_exp).squeeze(-2)

    def as_deterministic_output_module(self) -> nn.Module:
        return _GMMArgmaxModeOutput(self.num_modes, self.output_dim, self.use_tanh)

    @property
    def mean(self) -> torch.Tensor:
        probs = F.softmax(self._logits, dim=-1)  # type: ignore
        return (probs.unsqueeze(-1) * self._component_means).sum(dim=-2)  # type: ignore

    @property
    def std(self) -> torch.Tensor:
        """Marginal std of the mixture (law of total variance)."""
        probs = F.softmax(self._logits, dim=-1).unsqueeze(-1)  # type: ignore
        mix_mean = (probs * self._component_means).sum(dim=-2, keepdim=True)  # type: ignore
        diff = self._component_means - mix_mean  # type: ignore
        var = (probs * (self._component_scales**2 + diff**2)).sum(dim=-2)  # type: ignore
        return torch.sqrt(var.clamp_min(1e-12))

    @property
    def entropy(self) -> torch.Tensor:
        """MC entropy estimate (single sample). Cached per ``update()``."""
        if self._entropy_cache is None:
            with torch.no_grad():
                samples = self._distribution.sample()  # type: ignore
            self._entropy_cache = -self._distribution.log_prob(samples)  # type: ignore
        return self._entropy_cache

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return (self._component_means, self._component_scales, self._logits)  # type: ignore

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """MixtureSameFamily reduces event dim internally (via Independent)."""
        return self._distribution.log_prob(outputs)  # type: ignore

    def kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """MC KL(old || new) estimate using a single sample from the old distribution."""
        old_means, old_scales, old_logits = old_params
        new_means, new_scales, new_logits = new_params
        old_dist = D.MixtureSameFamily(
            mixture_distribution=D.Categorical(logits=old_logits),
            component_distribution=D.Independent(D.Normal(old_means, old_scales), 1),
        )
        new_dist = D.MixtureSameFamily(
            mixture_distribution=D.Categorical(logits=new_logits),
            component_distribution=D.Independent(D.Normal(new_means, new_scales), 1),
        )
        samples = old_dist.sample()
        return old_dist.log_prob(samples) - new_dist.log_prob(samples)


class _GMMArgmaxModeOutput(nn.Module):
    """Exportable: pick the argmax-mode mean from a flat GMM param vector."""

    def __init__(self, num_modes: int, output_dim: int, use_tanh: bool) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.output_dim = output_dim
        self.use_tanh = use_tanh

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        n, d = self.num_modes, self.output_dim
        means = mlp_output[..., : n * d].reshape(*mlp_output.shape[:-1], n, d)
        logits = mlp_output[..., 2 * n * d :]
        if not self.use_tanh:
            means = torch.tanh(means)
        idx = logits.argmax(dim=-1)
        idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(*idx.shape, 1, d)
        return torch.gather(means, -2, idx_exp).squeeze(-2)


class TanhGaussianDistribution(GaussianDistribution):
    """Gaussian with a tanh-squashed mean.

    Matches robomimic's ``RNNActorNetwork`` (non-GMM BC-RNN), which applies
    ``torch.tanh`` to the decoder output before returning actions. No Jacobian
    correction is applied in ``log_prob`` — this matches BC's own treatment,
    keeping warm-start behavior consistent.
    """

    def update(self, mlp_output: torch.Tensor) -> None:
        super().update(torch.tanh(mlp_output))

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return torch.tanh(mlp_output)

    def as_deterministic_output_module(self) -> nn.Module:
        return _TanhOutput()


class _TanhOutput(nn.Module):
    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return torch.tanh(mlp_output)
