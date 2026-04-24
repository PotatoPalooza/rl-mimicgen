# Resumable Official DPPO Pipeline Plan

## Goal

Add bounded-loss resume for the official MimicGen DPPO pipeline:

- if a stage is complete, skip it
- if a stage has not started, start it
- if a stage is partially complete, resume the same run in place
- reuse the same `run_dir` and W&B run instead of creating a new run
- allow losing work since the last checkpoint boundary

This plan targets resume from the latest saved checkpoint, not exact crash-point replay.

## Accepted Tradeoffs

- Losing up to the last checkpoint interval is acceptable
- We do not try to preserve dataloader cursor state
- We do not try to preserve async env state or in-flight rollout state
- Eval stages may restart from scratch if they do not have a clean partial-output story
- Resume support is only guaranteed for runs created after the new checkpoint format lands

## Current Gaps

### Launcher

- `scripts/run_official_dppo_mimicgen.py` always creates a fresh `run_id`
- stage selection is based on latest checkpoint discovery, not canonical in-progress runs
- sweep always creates a fresh output directory, so `--skip-existing` cannot resume prior partial sweeps

### Pretrain

- `dppo/agent/pretrain/train_agent.py` only saves:
  - `epoch`
  - `model`
  - `ema`
- optimizer, scheduler, and resume metadata are not saved
- `dppo/agent/pretrain/train_diffusion_agent.py` always starts from `epoch = 1`

### Finetune

- `dppo/agent/finetune/train_agent.py` only saves:
  - `itr`
  - `model`
- PPO optimizer and scheduler state are not saved
- best-metric state is not restored
- W&B resume is not wired

### Sweep

- `dppo/script/eval_checkpoint_sweep.py` already supports `--skip-existing`
- the launcher must reuse the same sweep output dir for this to matter

## Target Behavior

Each task stage gets one canonical run directory while it is active. Example:

- `pretrain/<dataset>/<run_name>/<stable_run_id>/`
- `finetune/<dataset>/<run_name>/<stable_run_id>/`
- `sweep/<dataset>/<run_name>/<stable_run_id>/`

Each run dir contains:

- `run_manifest.json`
- `run_state.json`
- `run_config.yaml`
- checkpoints
- stage outputs

`run_state.json` should track:

- `stage`
- `dataset_id`
- `status`: `running`, `completed`, `failed`, `interrupted`
- `target_progress`
- `completed_progress`
- `checkpoint_path`
- `wandb_run_id`
- `updated_at`

## Phase 1: Stable Run Identity

Files:

- `scripts/run_official_dppo_mimicgen.py`
- `scripts/run_dppo_bc_to_rl.sh`
- `scripts/run_dppo_batch_worker.sh`

Changes:

- Add stage-level run discovery helpers:
  - find latest incomplete run for `dataset_id + stage + seed + relevant config signature`
  - otherwise create a new run
- Stop creating a fresh `run_id` when resumable mode is enabled
- Add `--resume` or make resume the default behavior for pipeline stages
- Update batch worker semantics:
  - completed task: mark `DONE`
  - interrupted but resumable task: leave task available for re-claim or mark `FAILED` and allow retry

Deliverable:

- rerunning `pretrain`, `finetune`, or `sweep` reuses the same run dir if the prior run is incomplete

## Phase 2: W&B Reattach

Files:

- `dppo/agent/pretrain/train_agent.py`
- `dppo/agent/finetune/train_agent.py`

Changes:

- Persist a stable `wandb_run_id` in `run_manifest.json` or `run_state.json`
- Initialize W&B with:
  - `id=<stable id>`
  - `resume="allow"` or `resume="must"`
  - current run name
- On resumed runs, append to the same W&B run rather than creating a new run name-only session

Deliverable:

- resumed jobs continue logging into the same W&B run

## Phase 3: Pretrain Resume

Files:

- `dppo/agent/pretrain/train_agent.py`
- `dppo/agent/pretrain/train_diffusion_agent.py`

Changes:

- Extend pretrain checkpoint payload with:
  - `optimizer`
  - `lr_scheduler`
  - `epoch`
  - `cnt_batch`
  - optional RNG snapshots for better continuity
- Add helper to load the latest checkpoint in the current run dir
- Resume loop from `epoch + 1`
- Keep logging/checkpoint numbering monotonic inside the same run
- Mark stage complete when final epoch checkpoint exists

Deliverable:

- if pretrain crashes at epoch 1200 of 1500, rerun continues from the latest saved epoch in the same run dir

## Phase 4: Finetune Resume

Files:

- `dppo/agent/finetune/train_agent.py`
- `dppo/agent/finetune/train_ppo_agent.py`
- `dppo/agent/finetune/train_ppo_diffusion_agent.py`

Changes:

- Extend finetune checkpoint payload with:
  - `itr`
  - `model`
  - `actor_optimizer`
  - `critic_optimizer`
  - `actor_lr_scheduler`
  - `critic_lr_scheduler`
  - `eta_optimizer` and `eta_lr_scheduler` when enabled
  - `best_eval_success`
  - `best_eval_metrics`
  - `best_eval_itr`
  - `cnt_train_step`
- Add a `load_resume_checkpoint()` path
- Resume from the last saved iteration and continue until `n_train_itr`
- Ensure checkpoint file names stay aligned with `itr`

Deliverable:

- if finetune crashes at checkpointed iteration 100 of 200, rerun continues to 200 in the same run

## Phase 5: Sweep Resume

Files:

- `scripts/run_official_dppo_mimicgen.py`
- `dppo/script/eval_checkpoint_sweep.py`

Changes:

- Reuse the canonical sweep run dir
- Keep `--skip-existing`
- Load existing `best_checkpoint.json` and per-checkpoint metrics in place
- Recompute final summary from existing + newly evaluated checkpoints

Deliverable:

- a partially completed sweep continues instead of starting a second sweep run

## Phase 6: Completion Detection and Pipeline Orchestration

Files:

- `scripts/run_dppo_bc_to_rl.sh`
- `scripts/run_official_dppo_mimicgen.py`

Changes:

- Add explicit stage completion checks:
  - pretrain complete if final checkpoint exists
  - sweep complete if `best_checkpoint.json` exists and all intended checkpoints are accounted for
  - eval complete if expected result file exists
  - finetune complete if final target iteration has been reached
- Pipeline behavior:
  - skip completed stages
  - resume incomplete stages
  - start missing stages

Deliverable:

- `run_dppo_bc_to_rl.sh --task X` becomes restart-safe

## Phase 7: Tests

Files:

- `tests/test_run_official_dppo_mimicgen.py`
- new targeted tests under `tests/`

Add tests for:

- launcher reuses incomplete run dir
- launcher skips completed run
- pretrain resume from latest saved epoch
- finetune resume from latest saved itr
- sweep resume reuses prior output and `--skip-existing`
- W&B metadata is stable across resume
- stale interrupted batch tasks can be retried without duplicating runs

## Recommended Checkpoint Frequency

To keep resume cheap while limiting lost work:

- pretrain: every `25` or `50` epochs
- finetune: every `10` iterations

That keeps implementation simple while meeting the tolerance for small losses.

## Suggested Execution Order

1. Stable run identity in launcher
2. W&B stable run ids
3. Pretrain resume
4. Finetune resume
5. Sweep reuse
6. Pipeline completion detection
7. Batch-worker recovery polish

## Expected Outcome

After these changes:

- restarting the same task no longer creates a fresh run when an incomplete run already exists
- completed stages auto-skip
- interrupted training resumes from the last saved checkpoint
- W&B shows one continuous run per stage
- some work may be lost since the last checkpoint, but not the full stage
