# Diffusion Aligned BC Failure Summary

Date: 2026-04-17

## Why This Matters

The aligned diffusion BC pipeline was introduced to reduce the gap between robomimic BC training and the RL-stack evaluator. The intended outcome was:

1. train diffusion BC under the same runtime profile used for DPPO
2. evaluate saved BC checkpoints with the RL-stack evaluator
3. start online RL from a checkpoint that does not collapse at initial evaluation

The current aligned run did not achieve that goal.

## Run Investigated

- Run dir:
  [runs/aligned_bc_20260417_140041/core_training_results/coffee_d0_aligned_bc_v2_dppo_ddim5_ft5_act8/20260417140045](/home/nelly/projects/rl-mimicgen/runs/aligned_bc_20260417_140041/core_training_results/coffee_d0_aligned_bc_v2_dppo_ddim5_ft5_act8/20260417140045)
- Final checkpoint:
  [model_epoch_2000.pth](/home/nelly/projects/rl-mimicgen/runs/aligned_bc_20260417_140041/core_training_results/coffee_d0_aligned_bc_v2_dppo_ddim5_ft5_act8/20260417140045/models/model_epoch_2000.pth)
- RL-stack checkpoint sweep:
  [checkpoint_metrics.jsonl](/home/nelly/projects/rl-mimicgen/runs/aligned_bc_20260417_140041/core_training_results/coffee_d0_aligned_bc_v2_dppo_ddim5_ft5_act8/20260417140045/rl_aligned_eval/checkpoint_metrics.jsonl)

## What We Confirmed

### 1. The run is mislabeled

The experiment name says `coffee_d0_aligned_bc_v2`, but the saved config points at `coffee_d1.hdf5`.

- Evidence:
  [config.json](/home/nelly/projects/rl-mimicgen/runs/aligned_bc_20260417_140041/core_training_results/coffee_d0_aligned_bc_v2_dppo_ddim5_ft5_act8/20260417140045/config.json)
- Relevant lines:
  `experiment.name = coffee_d0_aligned_bc_v2_dppo_ddim5_ft5_act8`
  `train.data[0].path = /home/nelly/projects/rl-mimicgen/runs/datasets/core/coffee_d1.hdf5`

The checkpoint env metadata also resolves to `Coffee_D1`. This is a D1 run with a D0-looking name.

### 2. The model was bad from the first saved checkpoint

Every saved checkpoint in the RL-stack sweep scored zero:

- epoch 200: success 0.0
- epoch 400: success 0.0
- epoch 600: success 0.0
- ...
- epoch 2000: success 0.0

See:
[checkpoint_metrics.jsonl](/home/nelly/projects/rl-mimicgen/runs/aligned_bc_20260417_140041/core_training_results/coffee_d0_aligned_bc_v2_dppo_ddim5_ft5_act8/20260417140045/rl_aligned_eval/checkpoint_metrics.jsonl)

This means the issue is not late-training collapse. The run never produced a usable checkpoint under RL-stack evaluation.

### 3. The BC optimization itself did not numerically explode

The robomimic training log finishes cleanly at epoch 2000 with:

- loss around `0.04`
- reasonable gradient norms
- no NaNs or exceptions
- `finished run successfully`

Evidence:
[logs/log.txt](/home/nelly/projects/rl-mimicgen/runs/aligned_bc_20260417_140041/core_training_results/coffee_d0_aligned_bc_v2_dppo_ddim5_ft5_act8/20260417140045/logs/log.txt)

So this does not look like a numerical training failure. It looks like a semantically wrong or ineffective run.

### 4. Early rendered checkpoints are also dead

Rendered one-episode videos for the earliest available checkpoints:

- epoch 200:
  [eval.mp4](/home/nelly/projects/rl-mimicgen/logs/debug_eval_coffee_d1_epoch200_osmesa/videos/eval.mp4)
  [eval_metrics.json](/home/nelly/projects/rl-mimicgen/logs/debug_eval_coffee_d1_epoch200_osmesa/eval_metrics.json)
- epoch 400:
  [eval.mp4](/home/nelly/projects/rl-mimicgen/logs/debug_eval_coffee_d1_epoch400_osmesa2/videos/eval.mp4)
  [eval_metrics.json](/home/nelly/projects/rl-mimicgen/logs/debug_eval_coffee_d1_epoch400_osmesa2/eval_metrics.json)

Both runs ended with:

- `success_rate = 0.0`
- `return_mean = 0.0`
- `length_mean = 400.0`

This confirms the failure is present at the earliest saved checkpoints, not only at epoch 2000.

## Most Likely Interpretation

The current evidence supports the following:

1. We accidentally launched a D1 aligned run while naming it as D0.
2. That D1 aligned diffusion run trained to completion but never learned a policy that survives RL-stack evaluation.
3. Therefore the aligned diffusion pipeline, in its current form, did not solve the BC-to-RL eval collapse problem for this run.

What we do not yet know:

- whether the failure is specific to Coffee D1
- whether the failure is caused by the aligned runtime profile itself
- whether the failure comes from the aligned BC config overrides, rather than diffusion BC generally
- whether a correctly named and freshly launched D0 aligned run would behave differently

## Current Process Issues

### Issue A: experiment naming is unsafe

The run name can encode the wrong task variant, which makes later analysis error-prone.

### Issue B: success is only checked post hoc

The aligned runner disables internal robomimic rollout evaluation and only checks RL-stack performance after training. That means a fully bad run can consume 2000 epochs before we know it failed.

### Issue C: checkpoint cadence is too sparse for diagnosis

This run only saved every 200 epochs. We have no 50 or 100 epoch checkpoints, so we cannot pinpoint when the policy first became behaviorally wrong.

### Issue D: no hard pre-RL gate

We intended to avoid initial eval collapse before starting RL, but there is currently no enforced rule like:

- do not start RL unless BC checkpoint success is at least X under RL-stack eval

## What To Do Now With Diffusion

### Immediate operational decision

Do not start DPPO from this aligned checkpoint.

It already fails the initial-eval requirement the aligned pipeline was meant to guarantee.

### Recommended next experiment sequence

1. Relaunch aligned BC with the correct variant in the experiment name.
2. Save more often, at least every 50 epochs for diagnosis.
3. Run RL-stack checkpoint eval immediately after training.
4. Only promote checkpoints to RL if they pass a minimum eval threshold.

Suggested launch shape:

```bash
bash scripts/mimicgen_train_bc_aligned.sh \
  --task coffee \
  --variant D0 \
  --modality low_dim \
  --diffusion-runtime-profile dppo_ddim5_ft5_act8 \
  --experiment-name coffee_d0_aligned_bc_v3 \
  --run-root runs/aligned_bc_fix \
  --data-dir runs/datasets \
  --save-every-n-epochs 50 \
  --no-download-datasets
```

Suggested post-train gate:

- sweep saved checkpoints with `scripts/eval_bc_checkpoints.py`
- reject the run if all checkpoints have `success_rate == 0.0`
- reject the run if the best checkpoint is materially below the baseline diffusion BC checkpoint

### Baseline comparison that still needs to be done

Before trusting aligned diffusion again, compare:

1. standard diffusion BC checkpoint under RL-stack eval
2. aligned diffusion BC checkpoint under RL-stack eval

If standard diffusion survives but aligned diffusion does not, the problem is in the aligned pipeline or runtime-profile setup, not diffusion BC in general.

## Recommended Code / Workflow Fixes

1. Make the generated experiment name include the actual task variant automatically and refuse mismatched user-supplied names.
2. Add a `--save-every-n-epochs 50` default for aligned debugging runs.
3. Add a post-train failure summary that prints:
   best checkpoint, best success, and whether the run is RL-eligible.
4. Add an explicit promotion threshold before writing a “best checkpoint” symlink for RL use.
5. Add a small rendered eval at the first saved checkpoint during debugging runs.

## Bottom Line

This training run failed at the exact job it was supposed to do.

The aligned diffusion BC pipeline was supposed to prevent “initial RL eval collapse,” but this run produced only zero-success checkpoints under RL-stack evaluation. The run also carried a misleading D0 name while actually training Coffee D1, which made the failure harder to interpret quickly.

For now, diffusion is still viable as a direction, but this aligned BC path is not yet trustworthy enough to be the default source of RL starting checkpoints.
