from __future__ import annotations

from pathlib import Path

from rl_mimicgen.adapters.dppo_lowdim import MimicGenLowDimSpec, write_official_dppo_lowdim_configs


def test_generated_configs_use_spec_env_name(tmp_path: Path) -> None:
    spec = MimicGenLowDimSpec(
        dataset_id="hammer_cleanup_d1",
        source_hdf5=tmp_path / "hammer_cleanup_d1.hdf5",
        env_name="HammerCleanup_D1",
        low_dim_keys=("robot0_eef_pos",),
        obs_shapes={"robot0_eef_pos": (3,)},
        obs_dim=3,
        action_dim=7,
        horizon=500,
        num_demos=1,
        num_transitions=10,
    )

    artifact_dir = tmp_path / "artifacts"
    config_dir = tmp_path / "configs"
    log_root = tmp_path / "logs"
    artifact_dir.mkdir()
    (artifact_dir / "normalization.npz").write_text("", encoding="utf-8")
    (artifact_dir / "env_meta.json").write_text("{}", encoding="utf-8")

    configs = write_official_dppo_lowdim_configs(
        spec,
        artifact_dir=artifact_dir,
        config_dir=config_dir,
        log_root=log_root,
        wandb_entity=None,
    )

    pretrain_text = configs["pre_diffusion_mlp.yaml"].read_text(encoding="utf-8")
    finetune_text = configs["ft_ppo_diffusion_mlp.yaml"].read_text(encoding="utf-8")
    eval_bc_text = configs["eval_bc_diffusion_mlp.yaml"].read_text(encoding="utf-8")
    eval_rl_text = configs["eval_rl_init_diffusion_mlp.yaml"].read_text(encoding="utf-8")

    assert "env: HammerCleanup_D1" in pretrain_text
    assert "env_name: HammerCleanup_D1" in finetune_text
    assert "env_name: HammerCleanup_D1" in eval_bc_text
    assert "env_name: HammerCleanup_D1" in eval_rl_text
