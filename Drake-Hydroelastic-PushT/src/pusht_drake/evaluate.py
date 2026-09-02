"""Evaluate one cell closed-loop in Drake: a method, a training seed, a block.

    uv run python -m pusht_drake.evaluate --method coupled_a2 --seed 0 --block 1000

A cell is 300 episodes. Episode i starts from the frozen evaluation init
distribution (pusht_drake.sim.spawn, SeedSequence((1000, i))) and samples
actions with a fresh per-episode generator seeded block + i, so the same
index always replays the same episode. The artifact is one JSON file holding
the per-episode records plus their aggregate metrics.

The default device is cpu and the default worker count is 24; both are part
of the protocol, not tuning knobs. The per-episode noise generator is device
bound, so CPU and CUDA draw different streams, and each worker reuses one
simulator rig across its stripe of episodes, so a different worker count
moves outcomes at the third decimal. Every published episode ran with both
defaults.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from pusht_drake import paths
from pusht_drake.config import (
    ACTION_SEED_BLOCKS,
    ENV_SEED_BASE,
    MAX_EPISODE_STEPS,
    METHODS,
    NFE,
    NUM_EVAL_EPISODES,
    SCORE_TAU,
    TERMINATE_COVERAGE,
    WORKERS,
)


def evaluate(
    checkpoint: Path,
    *,
    method_key: str,
    seed: int,
    block: int,
    output: Path,
    episodes: int = NUM_EVAL_EPISODES,
    workers: int = WORKERS,
    device: str = "cpu",
) -> dict:
    """Run one cell and write its artifact atomically. Returns the artifact."""

    # Imported here so --help costs no pydrake.
    from pusht_drake.sim.episode_stats import aggregate_records
    from pusht_drake.sim.harness import PolicySpec, evaluate_policy
    from pusht_drake.sim.interface import RolloutLimits

    limits = RolloutLimits(
        max_steps=MAX_EPISODE_STEPS,
        terminate_coverage=TERMINATE_COVERAGE,
        score_tau=SCORE_TAU,
    )
    spec = PolicySpec(
        # Dispatched on the payload: the flow source carries no state_dict
        # keys, so a persistent checkpoint driven by the restart sampler would
        # load cleanly and produce plausible, wrong actions.
        module="pusht_drake.fm.adapter",
        factory="build_rollout_policy",
        kwargs={
            "checkpoint_path": str(checkpoint),
            "device": device,
            "num_integration_steps": NFE,
        },
    )

    wall_start = time.perf_counter()
    records = evaluate_policy(
        paths.ENV_CONFIG,
        spec,
        episodes=episodes,
        workers=workers,
        env_seed=ENV_SEED_BASE,
        action_seed=block,
        limits=limits,
        execution_mode="bspline",
    )
    wall = time.perf_counter() - wall_start

    metrics = aggregate_records(records, score_tau=limits.score_tau)
    artifact = {
        "method": method_key,
        "train_seed": seed,
        "block": block,
        "env_seed": ENV_SEED_BASE,
        "nfe": NFE,
        "workers": workers,
        "device": device,
        "checkpoint": _relative(checkpoint),
        "execution_mode": "bspline",
        "backend": "drake",
        "limits": {
            "max_steps": limits.max_steps,
            "terminate_coverage": limits.terminate_coverage,
            "score_tau": limits.score_tau,
        },
        # Two artifacts from bitwise-identical rollouts have differed by 30
        # percent in wall_time_s; recorded for orientation, never comparable.
        "wall_clock_comparable": False,
        "wall_time_s": wall,
        "episodes": episodes,
        "metrics": metrics,
        "records": records,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".tmp")
    partial.write_text(json.dumps(artifact, indent=2) + "\n")
    os.replace(partial, output)

    print(
        f"  {method_key} s{seed} b{block}: {len(records)} episodes, "
        f"mean max coverage {metrics['eval/mean_max_coverage']:.3f}, "
        f"success at 0.90 {metrics['eval/success_rate']:.3f} -> {output}"
    )
    return artifact


def _relative(checkpoint: Path) -> str:
    try:
        return str(Path(checkpoint).resolve().relative_to(paths.ROOT))
    except ValueError:
        return str(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="cfm_restart", choices=list(METHODS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block", type=int, default=ACTION_SEED_BLOCKS[0])
    parser.add_argument("--episodes", type=int, default=NUM_EVAL_EPISODES)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Anything but cpu is off-protocol: the noise stream is device bound.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=paths.CHECKPOINT_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing artifact.")
    args = parser.parse_args()

    if args.device != "cpu":
        print(f"  WARNING: device={args.device} is off-protocol; numbers will not pair.")
    checkpoint = paths.checkpoint_path(args.method, args.seed, args.checkpoint_dir)
    if not checkpoint.exists():
        raise SystemExit(f"No checkpoint at {checkpoint}; train it first.")
    output = args.out or paths.rollout_json_path(args.method, args.seed, args.block)
    if output.exists() and not args.force:
        raise SystemExit(f"{output} exists; pass --force to overwrite.")
    evaluate(
        checkpoint,
        method_key=args.method,
        seed=args.seed,
        block=args.block,
        output=output,
        episodes=args.episodes,
        workers=args.workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
