"""Default locations, anchored to the project root, not the working directory.

Every CLI default comes from here, so the commands work from any directory.
The layout:

    checkpoints/         provided and retrained policies, flat key-s-seed names
    config/              the Drake environment description
    data/demos_all.zarr  the teleoperated training corpus, tracked in git
    results/rollouts/    one JSON artifact per evaluation cell
"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """The directory that holds pyproject.toml, found by walking up."""

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"No pyproject.toml above {here}; the package layout is broken.")


ROOT = repo_root()
DATA_DIR = ROOT / "data"
CORPUS = DATA_DIR / "demos_all.zarr"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
ROLLOUT_DIR = RESULTS_DIR / "rollouts"
ENV_CONFIG = ROOT / "config" / "pusht_iiwa_workspace_optimized.yaml"


def checkpoint_path(key: str, seed: int, base: Path | None = None) -> Path:
    """Path to a trained policy: checkpoints/<key>-s<seed>.pt."""

    return (base or CHECKPOINT_DIR) / f"{key}-s{seed}.pt"


def rollout_json_path(key: str, seed: int, block: int, base: Path | None = None) -> Path:
    """Path to one cell's artifact: results/rollouts/<key>-s<seed>-b<block>.json."""

    return (base or ROLLOUT_DIR) / f"{key}-s{seed}-b{block}.json"
