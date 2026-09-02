"""Read and validate the human demonstration corpus.

The corpus ships with the repository at data/demos_all.zarr: 342 episodes
recorded in the Drake environment with an Xbox gamepad, SI units, Drake world
frame, 10 Hz. 301 are free-form teleoperation; the other 41 were recorded as
corrections from start states an earlier policy failed on.

Keys under data/: state (pusher x, y, theta), slider_state (T pose about its
area centroid), action (absolute pusher target one control period ahead),
target (goal pose, constant per episode). meta/episode_ends holds exclusive
episode boundaries.

validate() checks the whole contract before any training starts. Training on a
corpus with non-monotone boundaries, NaNs, or an action column that is not the
shifted state produces a policy that fails in ways that look like bad
hyperparameters, so every violation is reported here instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

DATA_GROUP = "data"
META_GROUP = "meta"
EPISODE_ENDS_KEY = "episode_ends"
SIDECAR_NAME = "episode_metadata.json"

STATE_KEY = "state"
SLIDER_STATE_KEY = "slider_state"
ACTION_KEY = "action"
TARGET_KEY = "target"

#: (key, feature dimension) of every required array.
LOWDIM_SPECS = (
    (STATE_KEY, 3),
    (SLIDER_STATE_KEY, 3),
    (ACTION_KEY, 2),
    (TARGET_KEY, 3),
)


@dataclass
class Dataset:
    path: Path
    arrays: dict[str, np.ndarray]
    episode_ends: np.ndarray
    sidecar: dict

    @property
    def n_episodes(self) -> int:
        return len(self.episode_ends)

    @property
    def n_frames(self) -> int:
        return int(self.episode_ends[-1]) if len(self.episode_ends) else 0

    def episode_slice(self, i: int) -> slice:
        start = 0 if i == 0 else int(self.episode_ends[i - 1])
        return slice(start, int(self.episode_ends[i]))


def load_dataset(path: str | Path) -> Dataset:
    """Load every array into memory; the corpus is a few megabytes."""

    path = Path(path)
    root = zarr.open(str(path), mode="r")
    data = root[DATA_GROUP]
    arrays = {key: np.asarray(data[key]) for key in sorted(data.array_keys())}
    episode_ends = np.asarray(root[f"{META_GROUP}/{EPISODE_ENDS_KEY}"])

    sidecar_path = path / SIDECAR_NAME
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
    return Dataset(path=path, arrays=arrays, episode_ends=episode_ends, sidecar=sidecar)


def validate(dataset: Dataset) -> list[str]:
    """Return a list of problems; empty means the corpus satisfies the contract."""

    problems: list[str] = []
    arrays, ends = dataset.arrays, dataset.episode_ends

    for key, _ in LOWDIM_SPECS:
        if key not in arrays:
            problems.append(f"missing required array data/{key}")
    if problems:
        return problems

    if ends.ndim != 1 or len(ends) == 0:
        problems.append(f"meta/episode_ends must be a non-empty 1-D array, got shape {ends.shape}")
    else:
        if not np.all(np.diff(ends) > 0):
            problems.append(f"episode_ends must be strictly increasing, got {ends.tolist()}")
        if ends[0] <= 0:
            problems.append(f"first episode_end must be > 0, got {ends[0]}")
        n = int(ends[-1])
        for key, arr in arrays.items():
            if len(arr) != n:
                problems.append(f"data/{key} has {len(arr)} rows but episode_ends[-1] = {n}")

    for key, dim in LOWDIM_SPECS:
        arr = arrays[key]
        if arr.ndim != 2 or arr.shape[1] != dim:
            problems.append(f"data/{key} should be (N, {dim}), got {arr.shape}")

    for key, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
            bad = int(np.count_nonzero(~np.isfinite(arr)))
            problems.append(f"data/{key} contains {bad} non-finite values")

    # The action contract: action at t equals the pusher state at t+1 within
    # each episode, with the final action repeated.
    state, action = arrays[STATE_KEY], arrays[ACTION_KEY]
    for i in range(len(ends)):
        sl = dataset.episode_slice(i)
        s, a = state[sl][:, :2], action[sl]
        if len(s) < 2:
            continue
        if not np.allclose(a[:-1], s[1:], atol=1e-6):
            worst = float(np.max(np.abs(a[:-1] - s[1:])))
            problems.append(
                f"episode {i}: action does not lead the state (max abs diff {worst:.2e})"
            )
        if not np.allclose(a[-1], s[-1], atol=1e-6):
            problems.append(f"episode {i}: final action should repeat the final state xy")

    target = arrays[TARGET_KEY]
    for i in range(len(ends)):
        t = target[dataset.episode_slice(i)]
        if len(t) and not np.allclose(t, t[0], atol=1e-6):
            problems.append(f"episode {i}: target is not constant within the episode")

    return problems
