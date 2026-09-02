"""Train one method at one seed and write its checkpoint.

    uv run python -m pusht_drake.train --method coupled_a2 --seed 0

Every method trains under the frozen recipe in pusht_drake.fm.recipe, which no
method can override, so the arms differ in their source construction and their
flow-time schedule and in nothing else. The step budget is derived from the
corpus (400 epochs at the realized updates per epoch; 42,400 steps on the
shipped corpus), never typed.

The corpus ships with the repository, so this command needs no download. On
the reference GPU one run takes about 90 seconds; training on another device
or torch version produces statistically equivalent rather than bit-identical
weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pusht_drake import paths, store
from pusht_drake.config import METHODS
from pusht_drake.fm.recipe import TrainConfig
from pusht_drake.fm.runner import Runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="cfm_restart", choices=list(METHODS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path, default=paths.CORPUS)
    parser.add_argument("--output-dir", type=Path, default=paths.CHECKPOINT_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override the derived step budget. Smoke tests only.",
    )
    args = parser.parse_args()

    dataset = store.load_dataset(args.data)
    problems = store.validate(dataset)
    if problems:
        listing = "\n  ".join(problems)
        raise SystemExit(f"The corpus violates its contract:\n  {listing}")
    print(f"corpus: {dataset.n_episodes} episodes, {dataset.n_frames} frames ({args.data})")

    method = METHODS[args.method]
    config = TrainConfig(
        source_type=method.source_type,
        warm_depth=method.warm_depth,
        alpha=method.alpha if method.alpha is not None else 2.0,
        warmprior_sigma=method.warm_sigma if method.warm_sigma is not None else 1.0,
        seed=args.seed,
        max_train_steps=args.steps,
        budget_override=args.steps is not None,
    )
    device = torch.device(args.device) if args.device else None

    runner = Runner(
        method,
        config,
        states=dataset.arrays[store.STATE_KEY],
        slider_states=dataset.arrays[store.SLIDER_STATE_KEY],
        actions=dataset.arrays[store.ACTION_KEY],
        episode_ends=dataset.episode_ends,
        device=device,
    )
    runner.learn(paths.checkpoint_path(method.key, args.seed, args.output_dir))


if __name__ == "__main__":
    main()
