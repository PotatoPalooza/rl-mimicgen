# Midterm Progress Summary

As of March 13, 2026, the project has moved from setup and reproduction work into a functioning online fine-tuning pipeline for MimicGen policies. The repository now supports both the offline MimicGen behavioral cloning workflow and a custom online RL stage that starts from a robomimic BC checkpoint and fine-tunes it in robosuite.

## What We Have Built

We completed the basic experimental infrastructure needed to run the project end to end.

- Environment and dependency setup for the Blackwell machine is documented in [README.md](/home/nelly/projects/rl-mimicgen/README.md), with verification through [scripts/verify_blackwell_env.py](/home/nelly/projects/rl-mimicgen/scripts/verify_blackwell_env.py).
- MimicGen BC training is automated through [scripts/mimicgen_train_bc.sh](/home/nelly/projects/rl-mimicgen/scripts/mimicgen_train_bc.sh) and the one-task runner in [rl_mimicgen/mimicgen/paper_bc_one_task.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/mimicgen/paper_bc_one_task.py).
- We implemented an online RL training stack in [rl_mimicgen/rl/trainer.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/trainer.py), [rl_mimicgen/rl/policy.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/policy.py), [rl_mimicgen/rl/ppo.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/ppo.py), and [rl_mimicgen/rl/storage.py](/home/nelly/projects/rl-mimicgen/rl_mimicgen/rl/storage.py).
- The online RL method is documented mathematically in [docs/online_rl_math.md](/home/nelly/projects/rl-mimicgen/docs/online_rl_math.md).
- We added analysis tooling in [scripts/analyze_rl_logs.py](/home/nelly/projects/rl-mimicgen/scripts/analyze_rl_logs.py) to convert raw `metrics.jsonl` logs into summary tables and dashboards in [analysis/online_rl](/home/nelly/projects/rl-mimicgen/analysis/online_rl).
- We added unit coverage for config serialization, rollout storage behavior, and log summarization in [tests/test_online_rl_config.py](/home/nelly/projects/rl-mimicgen/tests/test_online_rl_config.py), [tests/test_online_rl_storage.py](/home/nelly/projects/rl-mimicgen/tests/test_online_rl_storage.py), and [tests/test_analyze_rl_logs.py](/home/nelly/projects/rl-mimicgen/tests/test_analyze_rl_logs.py).

## Online RL Method Implemented

The current online RL stage is not generic PPO from scratch. It is a MimicGen-specific fine-tuning setup with several project-specific pieces already implemented.

- The actor is initialized from a trained robomimic BC checkpoint instead of random initialization.
- A separate critic is trained online.
- PPO is augmented with a demonstration loss so policy updates remain anchored to the offline dataset.
- Recurrent robomimic policies are supported, including recurrent state handling during rollout and PPO replay.
- Several stabilization mechanisms are now in the codebase: KL-based early stopping, critic warmup, actor freezing by environment steps, decaying demo-loss weight, and an optional residual policy path.

This means the main algorithmic contribution for the midterm is already implemented and runnable, not just planned.

## Experimental Progress So Far

Our main empirical work so far has focused on online fine-tuning the `Coffee_D1` low-dimensional recurrent BC checkpoint:

`model_epoch_450_Coffee_D1_success_0.9.pth`

The analyzed runs are summarized in [analysis/online_rl/summary.md](/home/nelly/projects/rl-mimicgen/analysis/online_rl/summary.md). The most important results are:

- `coffee_d1_v1`: reached a best evaluation success rate of `1.0` by update `49`, but ended at `0.4`. This run showed severe instability. Its logged approximate KL reached values above `12`, indicating overly aggressive policy updates.
- `coffee_d1_rsl`: reached a best evaluation success rate of `1.0` by update `9`, but collapsed to `0.2` by the end of the 50-update run.
- `coffee_d1_stable`: with more conservative PPO settings, stronger demo regularization, and 20-episode evaluations, the run reached `0.9` best evaluation success and ended at `0.8`, which was much more stable than `v1`.
- `coffee_d1_v2`: with lower actor LR, tighter clipping, more frequent evaluation, and 20 evaluation episodes, the run reached `0.9` best evaluation success at update `54` and finished at `0.7`. Unlike `v1`, this run also ended with `1.0` training success rate, showing substantially better online adaptation.
- `coffee_d1_rsl_stable_v1`: reached `1.0` best evaluation success at update `39` and finished at `0.8`, which is currently one of the strongest and most stable runs in the repo.

Across these runs, we have already established an important midterm result: online fine-tuning can preserve or recover high task success on Coffee D1, but naive PPO settings are unstable and can destroy a strong BC prior. The project has therefore progressed from “can we run this?” to “which stabilization choices make BC-to-RL fine-tuning reliable?”

## What We Learned

Several concrete findings have come out of the experiments.

- Starting from a strong BC checkpoint is useful, but it also makes the training fragile because large PPO updates can quickly move the policy off the demonstration manifold.
- The unstable runs (`coffee_d1_v1`, `coffee_d1_rsl`) suggest that larger actor step sizes and looser trust-region settings cause policy collapse even when the run temporarily hits perfect evaluation success.
- More conservative settings improve reliability. The better runs use lower actor learning rates, smaller PPO clip ratios, stronger or slower-decaying demo regularization, and more careful evaluation.
- Evaluation protocol matters. Moving from 5 evaluation episodes to 20 episodes gave a clearer picture of whether gains were real or just noisy spikes.
- We now have enough tooling to compare runs systematically rather than relying on anecdotal observations from raw logs.

## Current Status

At midterm, the project has completed the following milestones:

1. Reproduced the MimicGen training environment and integrated the required local dependencies.
2. Built and documented an online RL fine-tuning pipeline on top of robomimic BC checkpoints.
3. Added experiment configs for multiple Coffee D1 fine-tuning variants under [config](/home/nelly/projects/rl-mimicgen/config).
4. Ran and analyzed multiple Coffee D1 online RL experiments, including both failed and stabilized settings.
5. Identified instability as the main technical challenge and implemented concrete mitigation mechanisms in the training code.

## Remaining Work for the Final Report

The main work left is empirical rather than infrastructural.

- Run the newer recovery and residual-policy configs to determine whether they outperform the current stable baselines.
- Extend experiments beyond the initial Coffee D1 setting to test whether the stabilization strategy transfers to other MimicGen tasks or modalities.
- Quantify results more rigorously with repeated seeds and cleaner comparisons against the offline BC baseline.
- Turn the current experimental findings into a more formal ablation story around actor LR, PPO clipping, KL control, and demo regularization.

## Recommended Visuals for the Midterm Report

The clearest set of visuals is already available in [analysis/online_rl](/home/nelly/projects/rl-mimicgen/analysis/online_rl). I would show the following:

1. [comparison_eval_success_rate.png](/home/nelly/projects/rl-mimicgen/analysis/online_rl/comparison_eval_success_rate.png)
Talk about:
This should be the main figure. Use it to show the central result that all methods can reach high success temporarily, but only the stabilized variants maintain that performance. Emphasize that `coffee_d1_v1` and `coffee_d1_rsl` spike early and then degrade, while `coffee_d1_stable`, `coffee_d1_v2`, and `coffee_d1_rsl_stable_v1` are more reliable.

2. [comparison_eval_return_mean.png](/home/nelly/projects/rl-mimicgen/analysis/online_rl/comparison_eval_return_mean.png)
Talk about:
Use this to support the success-rate plot with a reward-based metric. The point is that the more stable runs are not only succeeding more often, but also achieving stronger average evaluation returns, which suggests the policy quality is genuinely better rather than exploiting a noisy success metric.

3. [comparison_train_success_rate.png](/home/nelly/projects/rl-mimicgen/analysis/online_rl/comparison_train_success_rate.png)
Talk about:
Use this figure to discuss the difference between training-time rollouts and held-out evaluation. In particular, it helps explain why train success alone is not enough: some runs look better online than they do at evaluation time, and `coffee_d1_v2` is interesting because it finishes with strong train success while still retaining reasonably high eval success.

4. [coffee_d1_v1_dashboard.png](/home/nelly/projects/rl-mimicgen/analysis/online_rl/coffee_d1_v1_dashboard.png)
Talk about:
Use one unstable dashboard as a case study. This is the best figure for explaining collapse. Point out that the run reaches perfect evaluation success at one stage, but ends much worse, which motivates the claim that naive PPO settings are too aggressive for fine-tuning a strong BC prior.

5. [coffee_d1_rsl_stable_v1_dashboard.png](/home/nelly/projects/rl-mimicgen/analysis/online_rl/coffee_d1_rsl_stable_v1_dashboard.png) or [coffee_d1_stable_dashboard.png](/home/nelly/projects/rl-mimicgen/analysis/online_rl/coffee_d1_stable_dashboard.png)
Talk about:
Use one stable dashboard as the contrast to `v1`. Highlight the smaller performance swings and the much stronger final evaluation performance. This is where you can say that reduced actor step size, tighter PPO updates, and stronger regularization improved stability.

6. [summary.md](/home/nelly/projects/rl-mimicgen/analysis/online_rl/summary.md)
Talk about:
Include a compact table derived from this file in the report. The most useful columns are `best_eval_success_rate`, `best_eval_update`, `last_eval_success_rate`, `actor_lr`, `value_lr`, and `eval_episodes`. This table makes the experimental comparison explicit and helps connect behavior in the plots to concrete hyperparameter changes.

If you want a minimal presentation, show items 1, 4, 5, and a condensed table from item 6. That combination is enough to communicate the problem, the failure mode, and the current best progress.

In short, by midterm we have already built the full experimental pipeline, verified that online BC-to-RL fine-tuning works in this codebase, and identified the main stability bottleneck along with promising mitigation strategies.
