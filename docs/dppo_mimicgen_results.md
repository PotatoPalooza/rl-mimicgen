# DPPO MimicGen Results

Central scorecard for low-dim official `dppo` runs on MimicGen tasks.

Use one row per `task_dx` variant. Update the BC columns after the checkpoint sweep / BC eval pass, then update the RL columns after fine-tuning.

Recommended conventions:

- `BC success` should come from the selected swept BC checkpoint.
- `BC eval episodes` should record the episode budget used for that reported BC number.
- `RL init success` should be the first RL evaluation point when useful.
- `RL best success` should be the best observed online RL success for that run.
- `Status` can be `todo`, `prepared`, `bc`, `rl`, or `done`.
- Prefer short run identifiers or relative paths in `BC source` / `RL source`, not full local absolute paths.

## Progress Snapshot

- Done: `square_d0`
- In progress (`rl`): `coffee_d1`, `coffee_d2`, `square_d1`, `square_d2`
- Todo: `coffee_d0`, `stack_d0`, `stack_d1`, `threading_d0`, `threading_d1`, `threading_d2`, `hammer_cleanup_d0`, `hammer_cleanup_d1`, `three_piece_assembly_d0`, `three_piece_assembly_d1`, `three_piece_assembly_d2`

| Task | Status | BC checkpoint | BC eval episodes | BC success | RL init success | RL best success | BC source | RL source | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| coffee_d0 | todo |  |  |  |  |  |  |  |  |
| coffee_d1 | rl | state_1500.pt | 1 | 1.0000 | 0.8500 | 1.0000 | sweep/coffee_d1/coffee_d1_sweep_bc/sweep_bc_coffee_d1_20260419_135337_s42_e26ce8 | finetune/coffee_d1/coffee_d1_ft_diffusion_mlp_ta4_td20_tdf10/finetune_coffee_d1_20260418_232657_s42_b6414c | Sweep moved to a 1-episode screening after previous longer sweep; BC eval on this checkpoint is 0.8150 over 200 episodes. |
| coffee_d2 | rl | state_800.pt | 10 | 0.7000 | 0.6050 | 0.9900 | sweep/coffee_d2/coffee_d2_sweep_bc/sweep_bc_coffee_d2_20260419_015956_s42_0de597 | finetune/coffee_d2/finetune_coffee_d2_20260419_022909_s42_92ef86 | Best swept BC checkpoint came from a 10-episode sweep; BC eval on that checkpoint is 0.6050 over 200 episodes. |
| square_d0 | done | state_1500.pt | 32 | 0.90625 | 0.7550 | 0.9950 | sweep/square_d0/square_d0_sweep_bc/2026-04-18_19-30-16_42 | finetune/square_d0/square_d0_ft_diffusion_mlp_ta4_td20_tdf10/2026-04-18_18-57-34_42 | RL best reached 0.9950; RL init was 0.7550. |
| square_d1 | rl | state_1250.pt | 10 | 0.6000 | 0.6300 | 0.8650 | sweep/square_d1/square_d1_sweep_bc/sweep_bc_square_d1_20260419_141411_s42_b9c5c8 | finetune/square_d1/square_d1_ft_diffusion_mlp_ta4_td20_tdf10/finetune_square_d1_20260419_145306_s42_f85ea0 | BC sweep at 10 episodes; first RL eval was 0.6300, best RL eval 0.8650. |
| square_d2 | rl | state_1250.pt | 10 | 0.7000 | 0.3350 | 0.5200 | sweep/square_d2/square_d2_sweep_bc/sweep_bc_square_d2_20260419_141434_s42_9bcec9 | finetune/square_d2/square_d2_ft_diffusion_mlp_ta4_td20_tdf10/finetune_square_d2_20260419_145433_s42_5e6b91 | BC sweep at 10 episodes; first RL eval was 0.3350, best RL eval 0.5200. |
| stack_d0 | todo |  |  |  |  |  |  |  |  |
| stack_d1 | todo |  |  |  |  |  |  |  |  |
| threading_d0 | todo |  |  |  |  |  |  |  |  |
| threading_d1 | todo |  |  |  |  |  |  |  |  |
| threading_d2 | todo |  |  |  |  |  |  |  |  |
| hammer_cleanup_d0 | todo |  |  |  |  |  |  |  |  |
| hammer_cleanup_d1 | todo |  |  |  |  |  |  |  |  |
| three_piece_assembly_d0 | todo |  |  |  |  |  |  |  |  |
| three_piece_assembly_d1 | todo |  |  |  |  |  |  |  |  |
| three_piece_assembly_d2 | todo |  |  |  |  |  |  |  |  |
