from rl_mimicgen.rl.config import OnlineRLConfig
from rl_mimicgen.rl.storage import RolloutBatch


def test_online_rl_config_round_trip(tmp_path):
    config = OnlineRLConfig(checkpoint_path="model.pth", output_dir="logs/test")
    path = tmp_path / "config.json"
    config.dump_json(path)
    loaded = OnlineRLConfig.from_json(path)
    assert loaded.checkpoint_path == "model.pth"
    assert loaded.output_dir == "logs/test"


def test_rollout_batch_finish():
    batch = RolloutBatch()
    for _ in range(4):
        batch.add(
            observation={"obs": 1},
            goal=None,
            action=[0.0],
            reward=1.0,
            done=False,
            episode_start=False,
            value=0.0,
            log_prob=0.0,
        )
    batch.finish(last_value=0.0, gamma=0.99, gae_lambda=0.95)
    tensors = batch.as_numpy()
    assert tensors["advantages"].shape == (4,)
    assert tensors["returns"].shape == (4,)
