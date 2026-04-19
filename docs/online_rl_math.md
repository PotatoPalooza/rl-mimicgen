# Online RL Fine-Tuning on Top of a MimicGen BC Checkpoint

This note summarizes the math and design of the online RL method implemented in this repo. The goal is to start from a robomimic behavioral cloning checkpoint trained on MimicGen data and continue improving it with on-policy reinforcement learning in robosuite environments.

The relevant implementation lives in:

- [trainer.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/trainer.py)
- [policy.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/policy.py)
- [ppo.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/ppo.py)
- [storage.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/storage.py)

## 1. Starting Point

We load a robomimic policy checkpoint and treat its policy network as the initial actor:

\[
\pi_{\theta_0}(a_t \mid o_t, g_t, h_t)
\]

where:

- \(o_t\) is the current observation
- \(g_t\) is an optional goal observation
- \(h_t\) is recurrent hidden state for RNN policies

The checkpoint may be:

- low-dimensional or image-based
- feedforward or recurrent
- deterministic, Gaussian, or GMM

The adapter in [policy.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/policy.py) preserves the original robomimic observation preprocessing and checkpoint normalization.

## 2. Environment Interaction

Training is on-policy. At each update, we collect fresh rollouts from the current actor in a vectorized robosuite environment:

\[
\tau = \{(o_t, g_t, a_t, r_t, d_t, \log \pi_{\theta_{\text{old}}}(a_t), V_\phi(o_t, g_t))\}_{t=0}^{T-1}
\]

with:

- \(T =\) `rollout_steps`
- multiple parallel environments
- `terminated` and `truncated` combined into done flags

For recurrent policies, the rollout also stores the initial hidden state and step counter so that PPO replay uses the same sequence semantics as data collection.

## 3. Critic and Advantage Estimation

The value function is a separate network:

\[
V_\phi(o_t, g_t)
\]

It is not copied from the BC checkpoint. It is initialized fresh and trained online.

Advantages are computed with generalized advantage estimation (GAE):

\[
\delta_t = r_t + \gamma (1 - d_t) V_\phi(o_{t+1}, g_{t+1}) - V_\phi(o_t, g_t)
\]

\[
\hat{A}_t = \delta_t + \gamma \lambda (1 - d_t) \hat{A}_{t+1}
\]

Returns are then:

\[
\hat{R}_t = \hat{A}_t + V_\phi(o_t, g_t)
\]

After rollout collection, advantages are normalized over the batch.

## 4. PPO Objective

The base RL update is PPO. Let:

\[
r_t(\theta) = \frac{\pi_\theta(a_t \mid o_t, g_t, h_t)}{\pi_{\theta_{\text{old}}}(a_t \mid o_t, g_t, h_t)}
= \exp\left(\log \pi_\theta(a_t) - \log \pi_{\theta_{\text{old}}}(a_t)\right)
\]

The clipped surrogate objective is:

\[
L_{\text{PPO}}(\theta) =
\mathbb{E}_t \left[
\min \left(
r_t(\theta)\hat{A}_t,
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t
\right)
\right]
\]

The implementation minimizes the negative form:

\[
L_{\text{surrogate}} =
\mathbb{E}_t \left[
\max \left(
-r_t(\theta)\hat{A}_t,
-\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t
\right)
\right]
\]

The critic loss is:

\[
L_V(\phi) = \mathbb{E}_t \left[(\hat{R}_t - V_\phi(o_t, g_t))^2\right]
\]

An entropy bonus is subtracted to encourage exploration:

\[
L_H(\theta) = - \mathbb{E}_t \left[\mathcal{H}(\pi_\theta(\cdot \mid o_t, g_t, h_t))\right]
\]

So the pure PPO objective is:

\[
L_{\text{RL}} = L_{\text{surrogate}} + c_V L_V + c_H L_H
\]

where:

- \(c_V =\) `value_coef`
- \(c_H =\) `entropy_coef`

## 5. Demo-Augmented Policy Gradient

To keep the policy anchored near the demonstration manifold, the PPO update also includes a demonstration loss sampled from the MimicGen / robomimic training dataset.

The full objective is:

\[
L = L_{\text{surrogate}} + c_V L_V - c_H \mathcal{H} + \alpha L_{\text{demo}}
\]

where:

- \(\alpha =\) `demo.coef`
- `demo.coef` decays over training by `demo.decay`

This is DAPG-like in spirit: use online policy gradient for improvement, but regularize updates toward demonstrated behavior.

### Standard policy demo loss

For deterministic actors:

\[
L_{\text{demo}} = \mathbb{E}_{(o, g, a^*) \sim \mathcal{D}}
\left[\|\mu_\theta(o, g) - a^*\|_2^2\right]
\]

For stochastic Gaussian / GMM actors:

\[
L_{\text{demo}} = -\mathbb{E}_{(o, g, a^*) \sim \mathcal{D}}
\left[\log \pi_\theta(a^* \mid o, g)\right]
\]

This is computed directly from the current actor in [policy.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/policy.py), not by delegating the full update to robomimic.

## 6. Recurrent Policies

For RNN policies, correctness depends on replaying the same sequence structure seen during rollout collection.

This implementation handles that by:

- storing the rollout-start recurrent state
- storing per-step episode start flags
- respecting the robomimic checkpoint's RNN horizon
- recomputing log-probs in time order during PPO replay

Formally, for a recurrent actor:

\[
h_{t+1} = f_\theta(h_t, o_t, g_t)
\]

and the policy is:

\[
\pi_\theta(a_t \mid o_t, g_t, h_t)
\]

The PPO replay pass reconstructs \(h_t\) sequentially rather than flattening time arbitrarily. This is required for image-RNN and low-dim RNN policies to behave like the original checkpoint.

## 7. Actor Freezing and Critic Warmup

Two stabilization mechanisms are built into the optimizer:

### Critic warmup

For the first `critic_warmup_updates` PPO updates, the actor learning rate is set to zero:

\[
\eta_{\text{actor}} = 0
\]

This lets the value function begin fitting returns before the BC actor moves.

### Actor freeze by environment steps

For the first `actor_freeze_env_steps` collected steps, actor updates are also disabled:

\[
\eta_{\text{actor}} = 0 \quad \text{if} \quad N_{\text{env}} \leq N_{\text{freeze}}
\]

This is useful when starting from a strong BC checkpoint and wanting to avoid immediate policy drift.

## 8. KL-Based Early Stop

After each PPO minibatch, the implementation computes:

\[
\widehat{\mathrm{KL}} \approx \mathbb{E}_t \left[
\log \pi_{\theta_{\text{old}}}(a_t) - \log \pi_\theta(a_t)
\right]
\]

If:

\[
\widehat{\mathrm{KL}} > 1.5 \cdot \text{target\_kl}
\]

the PPO epoch stops early. This is not adaptive PPO in the trust-region sense, but it is a simple guardrail against overly large policy updates.

## 9. Evaluation Protocol

Evaluation runs the current policy in eval mode in a separate env pool. Metrics include:

- `eval/success_rate`
- `eval/return_mean`
- `eval/length_mean`
- `eval/evaluated_episodes`

The implementation distinguishes:

- latest eval result
- best-so-far eval result
- whether an eval ran on this update

This avoids conflating stale best metrics with fresh evaluation.

## 10. Optional Residual Variant

The repo also supports an optional residual policy mode.

Let the frozen BC policy be:

\[
a_{\text{base}} = \mu_{\text{BC}}(o_t, g_t, h_t)
\]

Let the trainable residual policy output:

\[
a_{\text{res}} \sim \pi_\theta(\cdot \mid o_t, g_t, h_t)
\]

Then the executed action is:

\[
a_t = \text{clip}\left(a_{\text{base}} + s \cdot a_{\text{res}}, -1, 1\right)
\]

where \(s =\) `residual.scale`.

In this mode:

- the base BC actor is frozen
- PPO trains only the residual actor
- the demo loss is applied to the combined action

Residual mode is intended as a conservative adaptation strategy, though in current experiments the plain policy update has performed better.

## 11. What This Method Is and Is Not

This approach is:

- on-policy
- policy-gradient
- demonstration-regularized
- initialized from a strong offline BC prior

This approach is not:

- off-policy Q-learning
- pure offline RL
- a trust-region method with exact KL constraints
- a generic RL library wrapper

The main research intuition is simple:

1. Start from a policy that already solves the MimicGen task often.
2. Use PPO to adapt it using real environment feedback.
3. Use demonstration regularization to reduce catastrophic drift away from the BC manifold.
4. Use conservative scheduling and evaluation because the initial policy is already strong.

## 12. Practical Interpretation

When this method works well, online RL should:

- preserve the BC checkpoint's initial competence
- improve success rate or robustness under the target reset distribution
- avoid rapid degradation from overly aggressive on-policy updates

In practice, the most important diagnostics are:

- eval success rate
- approximate KL
- demo loss magnitude
- whether actor updates are frozen
- train versus eval success mismatch

For strong MimicGen checkpoints, the optimization problem is usually not "learn from scratch." It is "make small, correct updates without destroying a good prior."
