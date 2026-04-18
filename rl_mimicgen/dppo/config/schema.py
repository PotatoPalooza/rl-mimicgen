from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DPPODatasetConfig:
    source_hdf5: str = ""
    bundle_dir: str = "runs/dppo_dataset"
    task: str = ""
    variant: str = ""
    obs_keys: tuple[str, ...] = ()
    max_demos: int | None = None


@dataclass
class DPPORunConfig:
    task: str = ""
    variant: str = ""
    seed: int = 0
    device: str = "cuda"
    dataset: DPPODatasetConfig = field(default_factory=DPPODatasetConfig)
    output_dir: str = "logs/dppo"
    train_steps: int = 0
    batch_size: int = 256
    num_envs: int = 1
    checkpoint_path: str | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "DPPORunConfig":
        with open(path, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        return cls(
            **{
                **data,
                "dataset": DPPODatasetConfig(**data.get("dataset", {})),
            }
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def dump_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as file_obj:
            json.dump(self.to_dict(), file_obj, indent=2)
