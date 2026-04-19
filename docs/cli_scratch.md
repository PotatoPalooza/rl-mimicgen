# CLI Scratch

## Official DPPO: MimicGen `coffee_d0` low-dim

Preferred task-spec workflow:

```bash
cd /home/nelly/projects/rl-mimicgen
.venv/bin/python scripts/run_official_dppo_mimicgen.py prepare --task coffee_d0
.venv/bin/python scripts/run_official_dppo_mimicgen.py pretrain --task coffee_d0
```

Materialize the official-DPPO-compatible dataset bundle and generated config:

```bash
cd /home/nelly/projects/rl-mimicgen
MUJOCO_GL=glx .venv/bin/python scripts/prepare_mimicgen_dppo_lowdim.py \
  --dataset runs/datasets/core/coffee_d0.hdf5 \
  --output-root runs/official_dppo_mimicgen \
  --verify-env-reset
```

Run the full official DPPO demo pretraining job for `coffee_d0`:

```bash
cd /home/nelly/projects/rl-mimicgen/dppo
MUJOCO_GL=glx ../.venv/bin/python script/run.py \
  --config-dir=cfg/mimicgen/generated/coffee_d0 \
  --config-name=pre_diffusion_mlp
```
