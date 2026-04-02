from rl_mimicgen.rl.config import OnlineRLConfig


def test_online_rl_config_round_trip(tmp_path) -> None:
    config = OnlineRLConfig(checkpoint_path="/tmp/model.pth", num_envs=16)
    config_path = tmp_path / "config.json"
    config.dump_json(config_path)

    loaded = OnlineRLConfig.from_json(config_path)

    assert loaded.checkpoint_path == "/tmp/model.pth"
    assert loaded.num_envs == 16
    assert loaded.robosuite.parallel_envs is True
    assert loaded.optimizer.actor_lr == 1e-4
    assert loaded.optimizer.value_lr == 1e-3
    assert loaded.ppo.entropy_coef == 0.006
    assert loaded.ppo.min_log_std == -2.0
    assert loaded.ppo.critic_warmup_updates == 0
    assert loaded.ppo.actor_freeze_env_steps == 8192
    assert loaded.residual.enabled is False
    assert loaded.residual.scale == 0.2
    assert loaded.evaluation.num_envs == 8
