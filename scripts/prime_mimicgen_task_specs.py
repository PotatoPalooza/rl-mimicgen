from __future__ import annotations

import argparse
import sys
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_SPECS_ROOT = REPO_ROOT / "configs" / "mimicgen_tasks"


def _ensure_repo_paths() -> None:
    for repo_name in ("mimicgen", "robomimic", "robosuite", "robosuite-task-zoo"):
        repo_path = REPO_ROOT / repo_name
        if repo_path.exists():
            repo_path_str = str(repo_path)
            if repo_path_str not in sys.path:
                sys.path.insert(0, repo_path_str)


def _build_task_spec(dataset_id: str, dataset_link: str, horizon: int) -> str:
    return f"""dataset_id: {dataset_id}

dataset:
  path: runs/datasets/{dataset_link}

materialize:
  output_root: runs/official_dppo_mimicgen
  val_split: 0.0
  split_seed: 0

logging:
  root: logs/official_dppo/mimicgen

runtime:
  mujoco_gl: glx
  video_mujoco_gl: osmesa

wandb:
  entity: null

pretrain:
  config: {{}}

finetune:
  config: {{}}

eval_bc:
  config:
    env:
      n_envs: 8

eval_rl_init:
  config:
    env:
      n_envs: 8

sweep:
  eval_mode: bc
  device: cpu
  n_envs: 8
  n_steps: {horizon}
  max_episode_steps: {horizon}
  every_n: 1
  video_checkpoints: none
  render_num: 1
  skip_existing: true
"""


def _dataset_family(dataset_id: str) -> str:
    match = re.match(r"(.+)_d\d+$", dataset_id)
    return match.group(1) if match else dataset_id


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prime missing MimicGen task-spec configs from the MimicGen dataset registry."
    )
    parser.add_argument(
        "--dataset-type",
        default="core",
        help="Dataset registry bucket to use. Defaults to core.",
    )
    parser.add_argument(
        "--family",
        action="append",
        dest="families",
        default=None,
        help="Task family prefix to include, for example square or threading. Repeatable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing task-spec files instead of only filling missing ones.",
    )
    return parser


def main() -> None:
    _ensure_repo_paths()
    import mimicgen

    args = build_argument_parser().parse_args()
    registry = mimicgen.DATASET_REGISTRY.get(args.dataset_type, {})
    TASK_SPECS_ROOT.mkdir(parents=True, exist_ok=True)

    families = tuple(args.families) if args.families else ()
    written = 0
    skipped = 0
    filtered_out = 0

    for dataset_id, payload in sorted(registry.items()):
        if families and _dataset_family(dataset_id) not in families:
            filtered_out += 1
            continue
        target = TASK_SPECS_ROOT / f"{dataset_id}.yaml"
        if target.exists() and not args.overwrite:
            skipped += 1
            continue
        target.write_text(
            _build_task_spec(
                dataset_id=dataset_id,
                dataset_link=str(payload["url"]),
                horizon=int(payload["horizon"]),
            ),
            encoding="utf-8",
        )
        written += 1

    print(
        f"Primed task specs in {TASK_SPECS_ROOT} | written={written} skipped={skipped} filtered_out={filtered_out}"
    )


if __name__ == "__main__":
    main()
