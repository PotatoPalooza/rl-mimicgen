# AWAC Refactor Status 2026-04-17

This note summarizes the AWAC experiments and refactor work completed so far, what the logs indicate, and the recommended next steps.

## What We Changed

- Created a repository tag at the start of the refactor:
  - `awac-refactor-start-2026-04-17`
- Fixed AWAC / PPO imports so non-diffusion runs do not require `diffusers`.
- Added fallback from parallel robosuite envs to serial envs when worker startup fails.
- Added medium and stable AWAC configs for coffee and square task variants.
- Began a sequence-aware AWAC refactor for recurrent BC checkpoints:
  - replay buffer now stores trajectories and can sample sequence windows
  - offline demos are added as trajectories, not only flat transitions
  - online AWAC rollouts are accumulated into trajectories
  - recurrent actor log-probs can now be computed over replayed sequence windows
- Added AWAC log-prob metrics to trainer output for fresh runs:
  - `awac_log_prob_mean`
  - `awac_log_prob_std`
  - `awac_log_prob_min`
  - `awac_log_prob_max`
- Fixed a recurrent rollout shape bug where entropy retained an extra axis and broke masking in the sequence path.
- Standardized square AWAC configs to use `robosuite.start_method = "spawn"` so CUDA and multiprocessing do not conflict.

## What We Learned

### Coffee runs

- `online_rl_coffee_awac_refactor_big`
  - best eval success: `0.8203125`
  - stable, but mostly plateaued
- `online_rl_coffee_awac_refactor_medium`
  - best eval success: `0.9375`
  - improved faster than `big`, but had large actor-loss spikes
- `online_rl_coffee_awac_refactor_medium_stable`
  - best eval success: `0.96875`
  - best coffee result so far
  - still drifted later, ending around `0.71875`

Interpretation:

- Coffee is workable with the current AWAC-style implementation.
- The main problem on coffee is not capacity; it is actor drift after reaching a strong checkpoint.

### Square runs

- `online_rl_square_awac_refactor_medium_stable`
  - best eval success: `0.375`
  - very unstable almost immediately
- `online_rl_square_awac_refactor_medium_stable_v2`
  - best eval success: `0.59375`
  - softer actor settings helped, but loss spikes remained severe
- `online_rl_square_awac_refactor_medium_stable_v2_seq`
  - best eval success: `0.5`
  - sequence-aware replay clearly improved early behavior
  - actor loss still spikes later, so the root issue is reduced but not fully solved

Interpretation:

- Square is materially harder than coffee under the current design.
- The previous single-step replay path for recurrent actors was invalid and was contributing to the failure mode.
- Sequence-aware replay improved early training behavior:
  - lower early `policy_loss`
  - lower and steadier `awac_weight_max`
  - lower early `awac_advantage_std`
- But the refactor is incomplete. Later actor-loss spikes show that sequence replay alone is not enough.

## Most Important Log Findings

1. The largest square `policy_loss` spikes were not explained by AWAC weights alone.
   - In several runs, `awac_weight_max` stayed modest while `policy_loss` still jumped into the thousands.
   - This strongly suggests the replay action log-prob term remains the unstable part of the objective.

2. Sequence-aware replay helped, but did not eliminate instability.
   - `stable_v2_seq` had much healthier updates `0-5` than `stable_v2`.
   - Later spikes still occurred.

3. Eval logging changed across runs and made comparisons harder than necessary.
   - Older runs often exposed `eval/best_success_rate` and `eval/success_rate`.
   - Some analysis assumed `eval/success_rate_mean`, which was not consistently present.

4. The logging schema still contains noise.
   - `algorithm` is written as `1.0`, which is not meaningful.
   - Fresh `awac_log_prob_*` metrics were added after `stable_v2_seq` had already started, so that run cannot answer the log-prob question cleanly.

## Recommended Improvements

### Highest priority

1. Rerun square sequence-aware AWAC from a fresh output dir with the new log-prob metrics enabled.
   - This is the shortest path to confirming whether bad replay log-probs still drive the spikes.

2. Add stronger actor regularization for AWAC fine-tuning.
   - Best candidates:
     - explicit KL penalty to the BC policy
     - or a BC anchor term on replay actions
   - The logs suggest the actor is still departing too aggressively from the behavior policy, especially on square.

3. Normalize the AWAC actor objective more conservatively.
   - Consider:
     - `normalize_weights = true`
     - tighter `max_weight`
     - optional clipping on very low replay log-probs
   - Current weights are not exploding, but these changes still reduce update variance.

### Medium priority

4. Make eval metrics schema consistent.
   - Always emit the same keys for eval success and last eval success.
   - This avoids silent confusion when comparing runs.

5. Fix low-value metrics.
   - Replace `algorithm = 1.0` with a string-like identifier in logs or remove it.

6. Keep `spawn` as the default multiprocessing mode for CUDA training runs.
   - `fork` caused the expected CUDA reinitialization failure.

### Research direction

7. Continue the AWAC refactor toward a more faithful recurrent implementation.
   - Current state:
     - recurrent actor uses sequence replay windows
     - critic is still transition-based
   - Likely next refactor slices:
     - better recurrent actor diagnostics
     - explicit behavior-policy anchoring
     - clearer offline / online replay mixing controls

## Suggested Immediate Next Step

Run a fresh square sequence-aware experiment with the current code and inspect:

- `awac_log_prob_mean`
- `awac_log_prob_min`
- `policy_loss`
- `awac_weight_max`
- `eval/success_rate`

If `policy_loss` spikes still coincide with extremely bad replay log-probs while weights remain moderate, the next concrete code change should be actor anchoring rather than more hyperparameter tuning.

## Where We Are Leaving Off

- Coffee has a usable baseline and a strong best checkpoint.
- Square improved from the refactor, but is still unstable enough that more config-only tuning is unlikely to solve it cleanly.
- The current codebase now supports sequence-aware recurrent AWAC actor updates, but the refactor is not complete.
- The next decision point is whether to:
  - keep hardening this practical AWAC-style method with better regularization
  - or continue toward a more paper-faithful recurrent AWAC implementation
