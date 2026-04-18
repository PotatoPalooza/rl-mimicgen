


# BC training diffusion

```
export MUJOCO_GL=glx
bash scripts/mimicgen_train_bc.sh \
    --task square \
    --variant D1 \
    --modality low_dim \
    --algo diffusion_policy \
    --max-parallel 1 \
    --data-dir runs/datasets \
    --no-download-datasets
```

# BC training diffusion aligned
```
export MUJOCO_GL=glx
bash scripts/mimicgen_train_bc.sh \
    --task coffee \
    --variant D1 \
    --modality low_dim \
    --algo diffusion_policy \
    --diffusion-runtime-profile dppo_ddim5_ft5_act8 \
    --experiment-name coffee_d0_dp_bc_aligned \
    --max-parallel 1 \
    --data-dir runs/datasets \
    --no-download-datasets
```

version 2
```

  bash scripts/mimicgen_train_bc_aligned.sh \
    --task coffee \
    --variant D1 \
    --modality low_dim \
    --diffusion-runtime-profile dppo_ddim5_ft5_act8 \
    --experiment-name coffee_d0_aligned_bc_v2
    --max-parallel 1 \
    --data-dir runs/datasets \
    --no-download-datasets
```

# Eval Bc model (and select best)
  MUJOCO_GL=glx \
  UV_CACHE_DIR=.uv-cache \
  uv run python scripts/eval_bc_checkpoints.py \
    --config config/online_rl_coffee_v11_dppo_profile_eval.json \
    --run-dir runs/aligned_bc_YYYYMMDD_HHMMSS/core_training_results/coffee_d0_aligned_bc_dppo_ddim5_ft5_act8/20260417123456 \
    --output-dir analysis/eval_bc/coffee_d0_aligned_bc \
    --episodes 20 \
    --eval-num-envs 4

# Training diffusion RL

Smoke - TODO make this a json conf
```
  MUJOCO_GL=glx \
  UV_CACHE_DIR=.uv-cache \
  uv run python scripts/train_online_rl.py \
    --config config/online_rl_coffee_vc2.json \
    --checkpoint runs/coffee_parallel/D0_low_dim/core_training_results/core_coffee_d0_low_dim_diffusion_policy/20260401163037/models/model_epoch_1950_coffee_d0_success_0.9.pth \
    --output_dir logs/online_rl_coffee_d0_diffusion_v3_smoke \
    --total_updates 2 \
    --num_envs 4 \
    --rollout_steps 32
```

Full run
```
  MUJOCO_GL=glx \
  UV_CACHE_DIR=.uv-cache \
  uv run python scripts/train_online_rl.py \
    --config config/online_rl_coffee_vc2.json \
    --checkpoint runs/coffee_parallel/D0_low_dim/core_training_results/core_coffee_d0_low_dim_diffusion_policy/20260401163037/models/model_epoch_1950_coffee_d0_success_0.9.pth \
    --output_dir logs/online_rl_coffee_d0_diffusion_v3 \
    --diffusion_use_ema
```


Diffusion Refac
```
  Baseline:

  MUJOCO_GL=glx \
  UV_CACHE_DIR=.uv-cache \
  uv run python scripts/train_online_rl.py \
    --config config/online_rl_coffee_v6_smoke.json \
    --checkpoint runs/coffee_parallel/D0_low_dim/core_training_results/core_coffee_d0_low_dim_diffusion_policy/20260401163037/models/model_epoch_1950_coffee_d0_success_0.9.pth \
    --output_dir logs/online_rl_coffee_d0_diffusion_v6_smoke

  ft_denoising_steps=5:

  MUJOCO_GL=glx \
  UV_CACHE_DIR=.uv-cache \
  uv run python scripts/train_online_rl.py \
    --config config/online_rl_coffee_v6_smoke_ft5.json \
    --checkpoint runs/coffee_parallel/D0_low_dim/core_training_results/core_coffee_d0_low_dim_diffusion_policy/20260401163037/models/model_epoch_1950_coffee_d0_success_0.9.pth \
    --output_dir logs/online_rl_coffee_d0_diffusion_v6_smoke_ft5

  ft_denoising_steps=1:

  MUJOCO_GL=glx \
  UV_CACHE_DIR=.uv-cache \
  uv run python scripts/train_online_rl.py \
    --config config/online_rl_coffee_v6_smoke_ft1.json \
    --checkpoint runs/coffee_parallel/D0_low_dim/core_training_results/core_coffee_d0_low_dim_diffusion_policy/20260401163037/models/model_epoch_1950_coffee_d0_success_0.9.pth \
    --output_dir logs/online_rl_coffee_d0_diffusion_v6_smoke_ft1

```


### AWAC
```
  MUJOCO_GL=glx PYTHONPATH=. .venv/bin/python scripts/train_online_rl.py \
    --config config/online_rl_coffee_awac.json
```


### Current runs

Aligned diffusion
```
MUJOCO_GL=glx PYTHONPATH=. python3 -m rl_mimicgen.mimicgen.train_robomimic \
    --config runs/aligned_bc_20260417_140041/core_training_results/coffee_d0_aligned_bc_v2_dppo_ddim5_ft5_act8/20260417140045/config.json \
    --resume

```

TODO - run eval script
```

  bash scripts/mimicgen_train_bc_aligned.sh \
    --task coffee \
    --variant D0 \
    --modality low_dim \
    --diffusion-runtime-profile dppo_ddim5_ft5_act8 \
    --experiment-name coffee_d0_aligned_bc_v2 \
    --run-root runs/aligned_bc_20260417_140041 \
    --no-run-training
  python3 scripts/eval_bc_checkpoints.py \
    --run-dir runs/aligned_bc_20260417_140041/core_training_results/coffee_d0_aligned_bc_v2_dppo_ddim5_ft5_act8/20260417140045 \
    --output-dir analysis/eval_bc/coffee_d0_aligned_bc_v2
```
