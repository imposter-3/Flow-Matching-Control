"""Replay evaluation episodes as standalone Meshcat HTML animations.

    uv run python -m pusht_drake.replay --method coupled_a2 --seed 0 --episode 0
    uv run python -m pusht_drake.replay --method coupled_a2 --seed 0 --pick worst
    uv run python -m pusht_drake.replay --method coupled_a2 --seed 0 --episode 3 --live

Episode i of a block is a pure function of its index: the scene comes from
SeedSequence (1000, i) and the action noise from a fresh generator seeded
block + i, so replaying an index reproduces exactly the scene and noise the
campaign evaluated. Each replay is saved as one self-contained HTML file with
a playback timeline; open it in any browser. Recording changes no numbers.
The replay includes the reset transient (the arm driving to its push-start
posture and the T dropping into place) before the policy takes over.

--pick reads the cell's evaluation artifact and selects episodes by their
recorded max coverage (worst, best, or median), which is the quick way to
look at failures.

A plain replay runs the episode on a fresh rig, so it can end differently
from the artifact's record: inside a campaign worker every episode starts
from the previous episode's microscopic reset residue, and contact dynamics
amplify that difference over 300 steps, occasionally across the success
threshold. Pass --exact to reproduce the recorded trajectory instead. It
first re-runs the episode's predecessors in its worker stripe (read from the
artifact), which restores the exact rig state the record was produced under.
The saved file then contains the whole stripe; the tool prints where on the
timeline the picked episode begins.

--live additionally runs the simulation at wall-clock speed and prints a
local URL to watch it as it happens.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pusht_drake import paths
from pusht_drake.config import (
    ACTION_SEED_BLOCKS,
    ENV_SEED_BASE,
    MAX_EPISODE_STEPS,
    METHODS,
    NFE,
    SCORE_TAU,
    TERMINATE_COVERAGE,
)


def pick_episodes(
    artifact_path: Path, mode: str, count: int, cell: tuple[str, int, int]
) -> list[int]:
    """Episode indices chosen by recorded max coverage from a cell artifact."""

    method_key, seed, block = cell
    if not artifact_path.exists():
        raise SystemExit(
            f"No evaluation artifact at {artifact_path}. Run the cell first:\n"
            f"  uv run python -m pusht_drake.evaluate --method {method_key} "
            f"--seed {seed} --block {block}\n"
            "or pass explicit episode indices with --episode."
        )
    artifact = json.loads(artifact_path.read_text())
    identity = (artifact.get("method"), artifact.get("train_seed"), artifact.get("block"))
    if identity != cell:
        raise SystemExit(
            f"{artifact_path} records cell {identity}, but the replay was asked "
            f"for {cell}; the episode indices would not correspond."
        )
    ranked = sorted(artifact["records"], key=lambda r: r["max_coverage"])
    if mode == "best":
        ranked = ranked[::-1]
    elif mode == "median":
        middle = len(ranked) // 2
        ranked = ranked[middle:] + ranked[:middle][::-1]
    chosen = ranked[:count]
    for record in chosen:
        print(
            f"  picked episode {record['episode']:3d}: recorded max coverage "
            f"{record['max_coverage']:.3f}, {record['termination_reason']}"
        )
    return [record["episode"] for record in chosen]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="coupled_a2", choices=list(METHODS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block", type=int, default=ACTION_SEED_BLOCKS[0])
    parser.add_argument(
        "--episode",
        type=int,
        nargs="+",
        default=None,
        help="Episode indices to replay (default: 0).",
    )
    parser.add_argument(
        "--pick",
        choices=("worst", "best", "median"),
        default=None,
        help="Choose episodes by recorded max coverage from the cell's artifact.",
    )
    parser.add_argument("--count", type=int, default=1, help="How many episodes --pick selects.")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="Artifact to pick from (default: the cell's file under results/rollouts/).",
    )
    parser.add_argument(
        "--fps", type=float, default=None, help="Recording frame rate (default 30)."
    )
    parser.add_argument("--out-dir", type=Path, default=paths.RESULTS_DIR / "replays")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run at wall-clock speed and print a URL to watch live.",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help=(
            "Reproduce the recorded trajectory by first re-running the episode's "
            "worker-stripe predecessors from the artifact (see the module doc)."
        ),
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=paths.CHECKPOINT_DIR)
    args = parser.parse_args()

    if args.episode is not None and args.pick is not None:
        raise SystemExit("Pass either --episode or --pick, not both.")

    cell = (args.method, args.seed, args.block)
    artifact_path = args.artifact or paths.rollout_json_path(args.method, args.seed, args.block)
    if args.pick is not None:
        episodes = pick_episodes(artifact_path, args.pick, args.count, cell)
    else:
        episodes = args.episode or [0]

    artifact_workers = None
    if args.exact:
        if not artifact_path.exists():
            raise SystemExit(
                f"--exact needs the cell's artifact for its worker count; none at {artifact_path}."
            )
        artifact_workers = int(json.loads(artifact_path.read_text())["workers"])

    checkpoint = paths.checkpoint_path(args.method, args.seed, args.checkpoint_dir)
    if not checkpoint.exists():
        raise SystemExit(f"No checkpoint at {checkpoint}; train it first.")

    # Pin the math-library thread pools before torch is imported, exactly as
    # the evaluation harness does; a replay should behave like one worker.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    from pusht_drake.fm.adapter import build_rollout_policy
    from pusht_drake.sim.harness import _check_clip_box, _check_commitment, _episode_init
    from pusht_drake.sim.interface import RolloutLimits
    from pusht_drake.sim.recording import RECORDING_FPS, EpisodeRecorder
    from pusht_drake.sim.rig import build_rig
    from pusht_drake.sim.rollout import rollout_episode

    policy = build_rollout_policy(checkpoint, device="cpu", num_integration_steps=NFE)
    limits = RolloutLimits(
        max_steps=MAX_EPISODE_STEPS,
        terminate_coverage=TERMINATE_COVERAGE,
        score_tau=SCORE_TAU,
    )
    rig = None
    live_announced = False

    def fresh_rig():
        # One fresh rig per target, so every replay starts its recording at
        # sim time zero. Recording an episode that begins deep into a
        # long-lived rig's clock trips a stale-timestamp assertion inside
        # Drake's contact visualizer the moment contact evolves, and a
        # nonzero animation start would prepend that much frozen timeline
        # anyway. A fresh rig also makes each saved file independent of what
        # else the same command replayed.
        nonlocal live_announced
        built = build_rig(paths.ENV_CONFIG)
        _check_clip_box(policy, built.action_square)
        built.source.set_policy(policy)
        _check_commitment(policy, built.source)
        if args.live:
            built.station.simulator.set_target_realtime_rate(1.0)
            if not live_announced:
                # The meshcat server is one per process, so the URL is stable
                # across rig rebuilds; announce it once.
                url = built.station.meshcat.web_url()
                print(f"watch live: {url}")
                if sys.stdin.isatty():
                    input("open the URL, then press Enter to start...")
                live_announced = True
        return built

    def run_one(index: int):
        return rollout_episode(
            rig,
            policy,
            _episode_init(rig.cfg, ENV_SEED_BASE, index),
            env_seed=ENV_SEED_BASE,
            action_seed=args.block + index,
            limits=limits,
        )

    saved: list[Path] = []
    for index in episodes:
        rig = fresh_rig()
        recorder = EpisodeRecorder(rig, fps=args.fps or RECORDING_FPS)
        recorder.start()

        if artifact_workers is not None:
            # Restore the exact rig state the record was produced under by
            # running the episode's stripe predecessors, in order, inside the
            # same recording; the target's start on the timeline is printed.
            prefix = list(range(index % artifact_workers, index, artifact_workers))
            if prefix:
                print(
                    f"  episode {index:3d}: re-running {len(prefix)} stripe "
                    f"predecessor(s) to restore the recorded rig state..."
                )
            for predecessor in prefix:
                run_one(predecessor)
            if prefix:
                print(
                    f"  episode {index:3d}: begins at {rig.sim_time():.1f} s on the replay timeline"
                )

        error = None
        try:
            result = run_one(index)
        except Exception as failure:  # noqa: BLE001 -- a partial replay still helps
            error = failure
            result = None
        path = recorder.save(
            args.out_dir / f"{args.method}-s{args.seed}-b{args.block}-ep{index}.html"
        )
        saved.append(path)
        if result is not None:
            print(
                f"  episode {index:3d}: max coverage {result.max_coverage:.3f}, "
                f"{result.length} steps, {result.termination_reason} -> {path}"
            )
        else:
            print(f"  episode {index:3d}: SIM ERROR ({error}); partial replay -> {path}")

    print(f"\nsaved {len(saved)} replay(s) under {args.out_dir}. Open in any browser;")
    print("the console warning about gamepads is expected in every saved replay.")


if __name__ == "__main__":
    main()
