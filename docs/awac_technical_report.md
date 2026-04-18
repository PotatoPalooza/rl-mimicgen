# AWAC Technical Report

This note documents how to replicate the algorithm from the AWAC paper and what would need to change in this repository to implement it accurately.

Primary sources:

- Paper PDF: [awac_2006.09359.pdf](/home/nelly/projects/rl-mimicgen/docs/awac_2006.09359.pdf)
- OpenReview page: https://openreview.net/forum?id=OJiM1R3jAtZ
- arXiv DOI: https://doi.org/10.48550/arXiv.2006.09359

Paper:

- Ashvin Nair, Murtaza Dalal, Abhishek Gupta, Sergey Levine
- "AWAC: Accelerating Online Reinforcement Learning with Offline Datasets"

## Goal

The paper studies offline-to-online reinforcement learning:

1. Pretrain from an offline dataset.
2. Continue improving online without throwing away the offline data.
3. Do so with an off-policy actor-critic method that remains sample efficient once online interaction begins.

The key point is that AWAC is not PPO with imitation regularization. It is an off-policy actor-critic algorithm with an actor update that reweights behavior actions by an exponentiated advantage.

## Paper-Level Algorithm

At a high level, AWAC does the following:

1. Maintain an actor `pi(a | s)` and an action-conditioned critic `Q(s, a)`.
2. Train the critic off-policy using Bellman targets from replay data.
3. Compute the actor objective on replayed actions, weighted by the estimated advantage:

   `A(s, a) = Q(s, a) - V(s)`

   with

   `V(s) = E_{a' ~ pi(. | s)}[Q(s, a')]`

4. Update the actor by weighted maximum likelihood:

   `L_actor = - E_{(s,a) ~ D}[ log pi(a | s) * exp(A(s,a) / lambda) ]`

5. Continue sampling both offline and online transitions from replay during fine-tuning.

In practice, the important characteristics are:

- off-policy
- replay-based
- Q-learning based
- actor trains on dataset actions
- actor weighting uses `Q - V`, not GAE

## What This Repository Currently Implements

Current AWAC-related files:

- [rl_mimicgen/rl/awac.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/awac.py)
- [rl_mimicgen/rl/policy.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/policy.py)
- [rl_mimicgen/rl/storage.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/storage.py)
- [rl_mimicgen/rl/trainer.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/trainer.py)
- [rl_mimicgen/rl/config.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/config.py)

The current implementation is best described as:

- a non-diffusion online RL branch
- using a BC / BC-RNN actor wrapper
- collecting rollout batches online
- estimating scalar state values with `ValueNetwork`
- computing GAE-style raw advantages from rollout rewards
- fitting the actor with an exponentiated-advantage weighted log-prob objective

This is now a reasonable "AWAC-style" algorithm after the raw-advantage fix, but it is not paper-faithful AWAC.

## Main Mismatches vs. the Paper

### 1. The critic is a value network, not a Q-function

Current code constructs only a scalar value network in [policy.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/policy.py:80).

Paper-faithful AWAC needs:

- `Q(s, a)` at minimum
- preferably twin Q critics for stability
- target Q networks for TD targets

Without an action-conditioned critic, the implementation cannot compute the actual AWAC advantage `Q(s,a) - V(s)`.

### 2. The critic target is Monte Carlo / GAE-style return, not a Bellman target

Current storage computes advantages and returns in [storage.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/storage.py:99), and AWAC fits the critic to those returns in [awac.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/awac.py:148).

Paper-faithful AWAC should instead do TD learning:

- sample `(s, a, r, s', done)` from replay
- compute a target value for `s'`
- regress `Q(s, a)` to `r + gamma * (1 - done) * V_target(s')`

### 3. The actor weights use rollout advantage, not `Q - V`

Current actor weights in [awac.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/awac.py:141) are based on rollout-derived raw advantages.

Paper-faithful AWAC weights should be based on:

- `Q(s, a_dataset)` for the replayed action
- minus `V(s)` computed from the current policy

This is the central algorithmic mismatch.

### 4. The data path is rollout-storage based, not replay-buffer based

Current trainer flow in [trainer.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/trainer.py:188) collects a fixed-horizon rollout and immediately updates from that batch.

Paper-faithful AWAC needs:

- persistent replay storage
- a mixed offline + online transition pool
- repeated off-policy sampling from replay

### 5. Next-state transitions are not first-class data in the AWAC path

The current rollout storage is built for PPO-like updates and does not model the AWAC critic target path directly.

True AWAC needs explicit transition tuples:

- `obs`
- `action`
- `reward`
- `next_obs`
- `done`
- `goal` and `next_goal` if goals are present

### 6. The update schedule is PPO-shaped

Current config fields in [config.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/config.py:34) are rollout-epoch-minibatch oriented.

Paper-faithful AWAC should expose replay and TD-specific controls instead:

- replay capacity
- replay warmup
- batch size
- target network update rate
- updates per environment step
- critic-to-actor update ratio
- number of policy action samples for `V(s)` estimation

## What To Keep Shared vs. Split

The right architecture is:

- keep the high-level trainer shell shared
- split the AWAC algorithm core from PPO/DPPO

Shared parts that should stay shared:

- checkpoint loading and robomimic actor wrapping in [policy.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/policy.py)
- environment construction and evaluation in [trainer.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/trainer.py)
- logging and top-level config plumbing
- demo loader plumbing

Parts that should split for accurate AWAC:

- storage format
- critic modules
- target computation
- update loop
- checkpoint payload for critic / target critic state

The current shared `RolloutBatch` abstraction is acceptable for the interim AWAC-style implementation, but it is not the right long-term abstraction for paper-faithful AWAC.

## Recommended Implementation Plan

### Phase 1: Introduce AWAC-specific transition storage

Add a new module:

- `rl_mimicgen/rl/replay_buffer.py`

This should store:

- `obs`
- `goal`
- `action`
- `reward`
- `next_obs`
- `next_goal`
- `done`
- `source` optional: `offline`, `online`, `demo`

Design notes:

- Keep it separate from [storage.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/storage.py).
- Do not overload rollout storage with replay semantics.
- Support vector-env insertion efficiently.

### Phase 2: Add Q critics and target critics

Extend the policy adapter or add a dedicated AWAC policy wrapper, for example:

- `rl_mimicgen/rl/awac_policy.py`

The class should provide:

- `actor`
- `q1(obs, goal, action)`
- `q2(obs, goal, action)` optional but recommended
- `target_q1`
- `target_q2`
- `sample_actions(obs, goal, n_samples)` or equivalent helper for value estimation

Do not replace the existing PPO policy path. Keep PPO / AWAC critic definitions separate.

### Phase 3: Build an AWAC transition batch type

Add a batch dataclass for replay samples, separate from `RolloutBatch`.

Suggested fields:

- `observations`
- `goals`
- `actions`
- `rewards`
- `next_observations`
- `next_goals`
- `dones`
- `source_mask` optional

This avoids continuing to encode two different learning semantics inside one batch type.

### Phase 4: Rewrite the AWAC critic update

Replace return regression in [awac.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/awac.py) with TD regression.

Suggested target:

1. Sample next-policy actions at `s'`.
2. Estimate `V_target(s')` with the target critics.
3. Compute:

   `target_q = r + gamma * (1 - done) * V_target(s')`

4. Fit Q critics to `target_q`.

If using twin critics:

- evaluate `min(target_q1, target_q2)` when constructing target values
- fit both critics independently to the same target

### Phase 5: Rewrite the AWAC actor update

Actor update should use replayed actions and Q-derived advantages:

1. For each replay sample, evaluate `Q(s, a_dataset)`.
2. Estimate `V(s)` by sampling actions from the current actor and evaluating the critic.
3. Compute:

   `adv = Q(s, a_dataset) - V(s)`

4. Weight:

   `w = exp(adv / beta)`

5. Clip the weight if configured.
6. Optimize:

   `L_actor = -(w * log pi(a_dataset | s)).mean()`

This is where the current implementation differs most from the paper.

### Phase 6: Move trainer orchestration to replay-driven updates

Modify the AWAC branch in [trainer.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/trainer.py) so that it:

1. Loads offline data into replay before training starts.
2. Collects online transitions every environment step.
3. Inserts transitions into replay continuously.
4. Performs replay-sampled critic and actor updates on a configurable schedule.

The PPO path should remain rollout-based and untouched.

### Phase 7: Load offline data into replay directly

Today the repository already has a demo loader intended for supervised BC-style demo losses.

For accurate AWAC, the offline dataset must also be loaded into replay as RL transitions.

That requires:

- reading datasets with next-state information
- converting dataset observations / actions / rewards / dones into replay format
- possibly reconstructing `next_obs` if the dataset stores sequence data rather than explicit transitions

This step is essential. Without it, the algorithm cannot faithfully do offline-to-online RL the way the paper does.

### Phase 8: Add target-network updates

Implement Polyak averaging:

- `target_param = tau * online_param + (1 - tau) * target_param`

Run this every update or every few updates.

This should live inside the AWAC algorithm implementation, not the shared trainer.

### Phase 9: Update checkpointing

Current trainer checkpoint payloads save actor and value-net state. Accurate AWAC checkpoints should additionally save:

- Q critics
- target critics
- replay metadata if resume fidelity matters
- optimizer state for all critic params

### Phase 10: Expand config

Add AWAC-specific config entries such as:

- `discount`
- `replay_capacity`
- `batch_size`
- `warmup_steps`
- `updates_per_step`
- `target_tau`
- `critic_updates_per_actor_update`
- `num_value_action_samples`
- `use_twin_q`
- `offline_replay_ratio` or equivalent if mixing offline and online samples explicitly

## Concrete File-by-File Change List

### New files

- `rl_mimicgen/rl/replay_buffer.py`
- `rl_mimicgen/rl/awac_policy.py` or `rl_mimicgen/rl/q_critic.py`
- `tests/test_replay_buffer.py`
- `tests/test_awac_policy.py`
- `tests/test_awac_targets.py`

### Existing files to modify

- [rl_mimicgen/rl/awac.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/awac.py)
  - replace return-regression critic
  - add TD target logic
  - add target critic updates
  - switch actor weights from replay `Q - V`

- [rl_mimicgen/rl/policy.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/policy.py)
  - either keep PPO-only pieces here and add an AWAC-specific policy wrapper elsewhere
  - or minimally expose actor sampling helpers that AWAC can reuse

- [rl_mimicgen/rl/trainer.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/trainer.py)
  - add replay construction and insertion
  - route AWAC through replay-based updates
  - leave PPO / DPPO rollout flow unchanged

- [rl_mimicgen/rl/config.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/config.py)
  - add replay and target-critic parameters

- [scripts/train_online_rl.py](/home/nelly/projects/rl-mimicgen/scripts/train_online_rl.py)
  - expose any new AWAC tuning knobs if needed

## Replication Recipe

To replicate the paper as closely as this repository allows, the training procedure should be:

1. Start from a BC-initialized actor checkpoint.
2. Build an offline transition replay buffer from dataset trajectories.
3. Initialize Q critics and target critics.
4. Run offline pretraining using replay-only batches.
5. Begin online environment interaction.
6. Continuously append online transitions to the same replay buffer.
7. Keep updating actor and critics from replay batches that include both offline and online experience.
8. Evaluate periodically on held-out rollouts.

The current repo already has:

- the BC checkpoint bootstrap
- the online environment and evaluation stack
- the demo dataset loader

It does not yet have:

- replay-backed transition learning
- Q critics
- target critics
- paper-faithful actor weights

## Validation Checklist

After implementation, AWAC should be validated with tests and metrics that specifically confirm the off-policy algorithm is being run correctly.

Unit tests:

- replay buffer insert / sample correctness
- target Q computation with terminal and non-terminal transitions
- actor weighting uses `Q - V`, not GAE
- target-network Polyak update
- checkpoint round-trip for actor, critic, target critic, optimizer

Training diagnostics to log:

- `q1_mean`, `q2_mean`, and optional `q_gap`
- `target_q_mean`
- `awac_advantage_mean/std/min/max`
- `awac_weight_mean/std/min/max`
- `awac_weight_clipped_frac`
- replay size
- fraction of offline vs online transitions in sampled batch

Smoke-run expectations:

- early eval should reflect BC prior
- critic targets should be finite and stable
- actor weights should not collapse to all-ones
- online fine-tuning should improve over the BC starting point if reward shaping and data are sane

## Minimal Accuracy Standard

If the goal is "accurate AWAC," the following are non-negotiable:

- replay buffer
- action-conditioned Q critic
- TD target
- actor weights based on `Q - V`

If any of those are missing, the result is still an AWAC-inspired method, not the paper algorithm.

## Practical Recommendation

Do not retrofit paper-faithful AWAC into the current PPO storage abstraction.

Instead:

1. Keep PPO / DPPO on their current rollout-based path.
2. Build AWAC as a separate off-policy branch sharing only:
   - actor initialization
   - env setup
   - eval loop
   - logging shell

That split is the cleanest design and the one least likely to break the existing PPO / DPPO implementations.
