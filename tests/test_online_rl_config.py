from rl_mimicgen.rl.config import OnlineRLConfig


def test_online_rl_config_round_trip(tmp_path) -> None:
    config = OnlineRLConfig(checkpoint_path="/tmp/model.pth", num_envs=16)
    config_path = tmp_path / "config.json"
    config.dump_json(config_path)

    loaded = OnlineRLConfig.from_json(config_path)

    assert loaded.checkpoint_path == "/tmp/model.pth"
    assert loaded.num_envs == 16
    assert loaded.algorithm == "ppo"
    assert loaded.robosuite.parallel_envs is True
    assert loaded.optimizer.actor_lr == 1e-4
    assert loaded.optimizer.value_lr == 1e-3
    assert loaded.ppo.entropy_coef == 0.006
    assert loaded.ppo.min_log_std == -2.0
    assert loaded.ppo.critic_warmup_updates == 0
    assert loaded.ppo.actor_freeze_env_steps == 8192
    assert loaded.awac.beta == 1.0
    assert loaded.awac.max_weight == 20.0
    assert loaded.awac.normalize_weights is True
    assert loaded.awac.actor_batch_size == 256
    assert loaded.awac.replay_capacity == 500000
    assert loaded.awac.discount == 0.99
    assert loaded.awac.num_q_networks == 2
    assert loaded.awac.target_tau == 0.005
    assert loaded.residual.enabled is False
    assert loaded.residual.scale == 0.2
    assert loaded.evaluation.num_envs == 8
    assert loaded.diffusion.enabled is True
    assert loaded.diffusion.num_inference_timesteps is None
    assert loaded.diffusion.use_ema is False


def test_online_rl_config_runtime_profile_applies(tmp_path) -> None:
    config = OnlineRLConfig()
    config.diffusion.runtime_profile = "dppo_ddim5_ft5_act8"
    config_path = tmp_path / "config.json"
    config.dump_json(config_path)

    loaded = OnlineRLConfig.from_json(config_path)

    assert loaded.diffusion.runtime_profile == "dppo_ddim5_ft5_act8"
    assert loaded.diffusion.use_ddim is True
    assert loaded.diffusion.ddim_steps == 5
    assert loaded.diffusion.num_inference_timesteps is None
    assert loaded.diffusion.use_ema is False
    assert loaded.diffusion.ft_denoising_steps == 5
    assert loaded.diffusion.gamma_denoising == 0.99
    assert loaded.diffusion.min_sampling_denoising_std == 0.01
    assert loaded.diffusion.min_logprob_denoising_std == 0.01
    assert loaded.diffusion.act_steps == 8


def test_online_rl_config_awac_round_trip(tmp_path) -> None:
    config = OnlineRLConfig(algorithm="awac")
    config.awac.beta = 0.3
    config.awac.max_weight = 10.0
    config.awac.normalize_weights = False
    config_path = tmp_path / "config_awac.json"
    config.dump_json(config_path)

    loaded = OnlineRLConfig.from_json(config_path)

    assert loaded.algorithm == "awac"
    assert loaded.awac.beta == 0.3
    assert loaded.awac.max_weight == 10.0
    assert loaded.awac.normalize_weights is False
    assert loaded.awac.actor_batch_size == 256


def test_online_rl_config_rejects_unknown_algorithm(tmp_path) -> None:
    config_path = tmp_path / "bad_config.json"
    config_path.write_text('{"algorithm":"invalid"}', encoding="utf-8")

    try:
        OnlineRLConfig.from_json(config_path)
    except ValueError as exc:
        assert "Unsupported online RL algorithm" in str(exc)
    else:
        raise AssertionError("Expected invalid algorithm to raise ValueError.")
