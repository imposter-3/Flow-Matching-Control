"""Default file locations, anchored to the project root rather than the CWD.

Every CLI default comes from here, so the commands work from any directory:

    checkpoints/         shipped and retrained policies, <key>-s<seed>.pt
    data/                the Push-T replay, downloaded on the first training run
    results/rollouts/    one CSV per (method, train seed) cell
"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the directory holding pyproject.toml, found by walking up."""

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"No pyproject.toml above {here}; the package layout is broken.")


ROOT = repo_root()
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
ROLLOUT_DIR = RESULTS_DIR / "rollouts"


def checkpoint_path(key: str, seed: int, base: Path | None = None) -> Path:
    """Path of a policy checkpoint: checkpoints/<key>-s<seed>.pt."""

    return (base or CHECKPOINT_DIR) / f"{key}-s{seed}.pt"


def rollout_csv_path(key: str, seed: int, nfe: int, base: Path | None = None) -> Path:
    """Path of one evaluation cell: results/rollouts/<key>-s<seed>-nfe<n>.csv."""

    return (base or ROLLOUT_DIR) / f"{key}-s{seed}-nfe{nfe}.csv"
