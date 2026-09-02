"""Run evaluation rollouts and write one CSV row per rollout.

Receding-horizon control: query the policy, execute the first H_e actions of
the chunk it returns, query again. H_e is fixed, so the horizon always slides
by the same amount and the schedules line up with what actually happens.

Every method is evaluated on identical environment seeds and identical action
seeds. The two are separate: env_seed fixes the initial scene, action_seed
fixes the policy's sampling noise. Every cell runs at the protocol's inference
budget of config.NFE network evaluations per replan.

The default device is cpu, and that is part of the protocol rather than a
fallback: the action-noise generator is device-bound, so CPU and CUDA draw
different streams, and every published episode ran on CPU. Another --device
gives valid numbers that are not comparable with the published ones.

    uv run python -m pusht_flow.evaluate --method coupled_a2 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch

from pusht_flow import paths
from pusht_flow.checkpoint import load_checkpoint
from pusht_flow.config import (
    ACTION_SEED_BLOCKS,
    AUXILIARY_THRESHOLD,
    ENV_SEED_START,
    METHODS,
    NFE,
    NUM_EVAL_EPISODES,
    SUCCESS_THRESHOLD,
)
from pusht_flow.env import EpisodeResult, make_env
from pusht_flow.rollout import CallCounter, build_policy

CSV_FIELDS = (
    "method",
    "train_seed",
    "env_seed",
    "action_seed",
    "nfe",
    "alpha",
    "max_coverage",
    "success_090",
    "success_095",
    "episode_length",
    "checkpoint",
    # Debug-only below. Recorded because they are free and diagnostic; never
    # promoted into a table.
    "final_coverage",
    "mean_reward",
    "terminated",
    "truncated",
    "num_replans",
)


def cell_is_done(csv_path: Path, expected_rows: int) -> bool:
    """A finished cell: the CSV exists with the header and every episode row.

    Writes are atomic (temp file plus rename below), so a file failing this
    shape check was not truncated mid-write; it is an off-protocol run, and the
    caller re-runs it rather than trusting it.
    """

    if not csv_path.exists():
        return False
    with csv_path.open() as handle:
        header = handle.readline().strip()
        rows = sum(1 for _ in handle)
    return header == ",".join(CSV_FIELDS) and rows == expected_rows


def run_episode(
    env,
    policy,
    *,
    env_seed: int,
    action_seed: int,
    nfe: int,
    execution_horizon: int,
) -> EpisodeResult:
    """One rollout under a fixed scene seed and a fixed sampling seed."""

    observation, info = env.reset(seed=env_seed)
    policy.reset()
    policy.counter.reset()
    generator = torch.Generator(device=policy.device).manual_seed(action_seed)

    coverages = [float(info.get("coverage", 0.0))]
    rewards: list[float] = []
    length = 0
    terminated = truncated = False

    while not (terminated or truncated):
        chunk = policy.predict(
            np.asarray(observation, dtype=np.float32),
            nfe=nfe,
            generator=generator,
        )
        for index in range(execution_horizon):
            observation, reward, terminated, truncated, info = env.step(chunk[index])
            coverages.append(float(info.get("coverage", 0.0)))
            rewards.append(float(reward))
            length += 1
            if terminated or truncated:
                break

    return EpisodeResult(
        env_seed=env_seed,
        action_seed=action_seed,
        max_coverage=max(coverages),
        final_coverage=coverages[-1],
        mean_reward=float(np.mean(rewards)) if rewards else 0.0,
        length=length,
        terminated=bool(terminated),
        truncated=bool(truncated),
        num_replans=policy.counter.replans,
    )


def evaluate(
    checkpoint: Path,
    *,
    output: Path,
    num_episodes: int = NUM_EVAL_EPISODES,
    env_seed_start: int = ENV_SEED_START,
    action_blocks: tuple[int, ...] = ACTION_SEED_BLOCKS,
    device: torch.device | str = "cpu",
) -> Path:
    loaded = load_checkpoint(checkpoint, device=device)
    method, recipe, train_seed = loaded.method, loaded.recipe, loaded.seed
    counter = CallCounter()
    policy = build_policy(
        loaded.model,
        loaded.normalizer,
        method,
        recipe,
        device=device,
        counter=counter,
    )
    env = make_env()

    # Record the checkpoint by a stable relative path where possible, so the
    # CSV does not embed one machine's home directory.
    try:
        recorded = str(Path(checkpoint).resolve().relative_to(paths.ROOT))
    except ValueError:
        recorded = str(checkpoint)

    output.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling temp file and rename at the end, so an interrupted run
    # leaves no truncated CSV for the resume logic to trust.
    partial = output.with_name(output.name + ".tmp")
    written = 0
    coverage_sum = 0.0
    success_sum = 0
    with partial.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for block in action_blocks:
            for index in range(num_episodes):
                result = run_episode(
                    env,
                    policy,
                    env_seed=env_seed_start + index,
                    action_seed=block + index,
                    nfe=NFE,
                    execution_horizon=recipe.execution_horizon,
                )
                success = int(result.success(SUCCESS_THRESHOLD))
                writer.writerow(
                    {
                        "method": method.key,
                        "train_seed": train_seed,
                        "env_seed": result.env_seed,
                        "action_seed": result.action_seed,
                        "nfe": NFE,
                        "alpha": method.alpha,
                        "max_coverage": f"{result.max_coverage:.6f}",
                        "success_090": success,
                        "success_095": int(result.success(AUXILIARY_THRESHOLD)),
                        "episode_length": result.length,
                        "checkpoint": recorded,
                        "final_coverage": f"{result.final_coverage:.6f}",
                        "mean_reward": f"{result.mean_reward:.6f}",
                        "terminated": int(result.terminated),
                        "truncated": int(result.truncated),
                        "num_replans": result.num_replans,
                    }
                )
                coverage_sum += result.max_coverage
                success_sum += success
                written += 1
    env.close()
    os.replace(partial, output)
    count = written or 1
    print(
        f"  {method.key} s{train_seed}: {written} episodes, "
        f"mean max coverage {coverage_sum / count:.3f}, "
        f"success at 0.90 {success_sum / count:.3f} -> {output}"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="cfm_restart", choices=list(METHODS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=paths.CHECKPOINT_DIR)
    parser.add_argument("--results-dir", type=Path, default=paths.RESULTS_DIR)
    parser.add_argument("--num-episodes", type=int, default=NUM_EVAL_EPISODES)
    parser.add_argument(
        "--action-blocks",
        type=int,
        nargs="+",
        default=list(ACTION_SEED_BLOCKS),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint = paths.checkpoint_path(args.method, args.seed, args.checkpoint_dir)
    if not checkpoint.exists():
        raise SystemExit(f"No checkpoint at {checkpoint}; train it first.")
    output = paths.rollout_csv_path(
        args.method, args.seed, NFE, args.results_dir / "rollouts"
    )
    if cell_is_done(output, args.num_episodes * len(args.action_blocks)):
        print(f"  complete, skipping: {output}")
        return
    evaluate(
        checkpoint,
        output=output,
        num_episodes=args.num_episodes,
        action_blocks=tuple(args.action_blocks),
        device=args.device,
    )


if __name__ == "__main__":
    main()
