"""Run the whole evaluation grid in one command: 4 methods x 3 training seeds.

    uv run python -m pusht_flow.sweep

One process per cell, all on CPU. Each cell writes its per-episode records to
results/rollouts/ and reports its own summary line as it finishes; the CSVs
are the data of record, and the sweep ends by printing the numbers behind the
README's table. The run is resumable: a cell whose CSV already exists with
the right shape is skipped, and every CSV is written atomically, so an
interrupted sweep can simply be started again.

CPU is part of the protocol rather than a fallback: the action-noise generator
is device-bound, so the device selects the noise stream, and every published
episode ran on CPU. It is also the faster device for a model this small
(144,394 parameters at batch size 1), where GPU launch overhead dominates.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import time
from pathlib import Path
from statistics import mean, stdev

from pusht_flow import paths
from pusht_flow.config import (
    ACTION_SEED_BLOCKS,
    METHODS,
    NFE,
    NUM_EVAL_EPISODES,
    TRAIN_SEEDS,
)

Cell = tuple[str, int]


def all_cells() -> list[Cell]:
    return [(key, seed) for key in METHODS for seed in TRAIN_SEEDS]


def _stat(values: list[float]) -> str:
    if len(values) < 2:
        return f"{mean(values):.3f}"
    return f"{mean(values):.3f} +/- {stdev(values):.3f}"


def print_summary(results_dir: Path) -> None:
    """The table's numbers: per-seed means reduced to mean +/- sample std.

    The per-cell lines printed as the sweep runs are the per-seed values;
    this is their cross-seed reduction, the same numbers the README
    tabulates. Seeds without a CSV are simply left out.
    """

    width = max(len(method.name) for method in METHODS.values())
    print(f"\nmean +/- sample std over {len(TRAIN_SEEDS)} training seeds:")
    for key, method in METHODS.items():
        coverage: list[float] = []
        success: list[float] = []
        for seed in TRAIN_SEEDS:
            cell = paths.rollout_csv_path(key, seed, NFE, results_dir / "rollouts")
            if not cell.exists():
                continue
            with cell.open() as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                continue
            coverage.append(sum(float(r["max_coverage"]) for r in rows) / len(rows))
            success.append(sum(int(r["success_090"]) for r in rows) / len(rows))
        if not coverage:
            print(f"  {method.name:<{width}}  no finished cells")
            continue
        note = ""
        if len(coverage) < len(TRAIN_SEEDS):
            note = f"  ({len(coverage)} of {len(TRAIN_SEEDS)} seeds)"
        print(
            f"  {method.name:<{width}}  max coverage {_stat(coverage)}   "
            f"success at 0.90 {_stat(success)}{note}"
        )


def _worker_init() -> None:
    # One torch thread per worker: the parallelism lives across cells, and a
    # model this small gains nothing from intra-op threads.
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Already fixed for this process; the env vars set by main() applied.
        pass


def run_cell(job: tuple[Cell, str, int, tuple[int, ...]]) -> tuple[Cell, float]:
    (key, seed), results_dir, num_episodes, action_blocks = job
    from pusht_flow.evaluate import evaluate

    started = time.time()
    evaluate(
        paths.checkpoint_path(key, seed),
        output=paths.rollout_csv_path(key, seed, NFE, Path(results_dir) / "rollouts"),
        num_episodes=num_episodes,
        action_blocks=action_blocks,
        device="cpu",
    )
    return (key, seed), time.time() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(len(all_cells()), (os.cpu_count() or 2) - 2)),
        help="Concurrent evaluation processes; more than one per cell is idle.",
    )
    parser.add_argument("--results-dir", type=Path, default=paths.RESULTS_DIR)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="20 episodes and one action block per cell, into results-smoke/.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    num_episodes, action_blocks = NUM_EVAL_EPISODES, ACTION_SEED_BLOCKS
    if args.smoke:
        results_dir = results_dir.parent / (results_dir.name + "-smoke")
        num_episodes, action_blocks = 20, ACTION_SEED_BLOCKS[:1]
    expected_rows = num_episodes * len(action_blocks)

    missing = [
        paths.checkpoint_path(key, seed)
        for key in METHODS
        for seed in TRAIN_SEEDS
        if not paths.checkpoint_path(key, seed).exists()
    ]
    if missing:
        listing = "\n  ".join(str(path) for path in missing)
        raise SystemExit(f"Missing checkpoints:\n  {listing}")

    from pusht_flow.evaluate import cell_is_done

    cells = all_cells()
    pending = [
        (key, seed)
        for key, seed in cells
        if not cell_is_done(
            paths.rollout_csv_path(key, seed, NFE, results_dir / "rollouts"),
            expected_rows,
        )
    ]
    print(
        f"{len(cells)} cells, {len(cells) - len(pending)} already done, "
        f"{len(pending)} to run with {args.workers} workers"
    )
    if not pending:
        print(f"every cell is present under {results_dir / 'rollouts'}")
        print_summary(results_dir)
        return

    # The env vars must be set before any worker imports torch.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"

    jobs = [(cell, str(results_dir), num_episodes, action_blocks) for cell in pending]
    started = time.time()
    finished = 0
    context = mp.get_context("spawn")
    with context.Pool(args.workers, initializer=_worker_init) as pool:
        for (key, seed), seconds in pool.imap_unordered(run_cell, jobs):
            finished += 1
            print(f"done {finished:>3d}/{len(pending)}  {key} s{seed}  {seconds:6.0f}s")
    total = time.time() - started
    print(f"swept {len(pending)} cells in {total / 60:.1f} min")
    print_summary(results_dir)


if __name__ == "__main__":
    main()
