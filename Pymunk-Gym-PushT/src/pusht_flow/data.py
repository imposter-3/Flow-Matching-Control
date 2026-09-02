"""The Push-T replay: fetching it, normalizing it, and slicing it into chunks.

This is the only module that imports a storage backend, so it is the only one
whose import costs one.

The coordinate contract lives here, in Normalizer. It is written into every
checkpoint and read back whenever one is scored, because it decides what the
network's numbers mean: a checkpoint restored against different statistics
produces wrong predictions and raises nothing.

Actions are relative: every waypoint in a chunk has the replan-time agent
position subtracted before z-scoring. That is why rollout.py has to re-express
a carried state when the horizon slides; see the frame note there.
"""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pusht_flow.config import (
    ACTION_HIGH,
    ACTION_LOW,
    PUSHT_DATASET_URL,
    PUSHT_ZARR_RELPATH,
    PUSHT_ZIP_SHA256,
    STATE_DIM,
    agent_position,
)


def download_replay(data_dir: Path) -> Path:
    """Download and extract the Push-T replay unless it is already present.

    The archive's sha256 is checked against the pin in config.py before
    extraction, so a corrupted or substituted download raises. A manually
    supplied data/pusht.zip goes through the same check.
    """

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = data_dir / PUSHT_ZARR_RELPATH
    if zarr_path.exists():
        return zarr_path

    archive = data_dir / "pusht.zip"
    if not archive.exists():
        print(f"downloading {PUSHT_DATASET_URL} -> {archive}")
        urllib.request.urlretrieve(PUSHT_DATASET_URL, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != PUSHT_ZIP_SHA256:
        raise ValueError(
            f"{archive} has sha256 {digest}, expected {PUSHT_ZIP_SHA256}. "
            "Delete the file and download again."
        )
    with zipfile.ZipFile(archive, "r") as handle:
        handle.extractall(data_dir)
    if not zarr_path.exists():
        raise FileNotFoundError(
            f"Extracted {archive} but {zarr_path} is still missing; the archive "
            "layout is not what this code expects."
        )
    return zarr_path


def load_replay(zarr_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (states, actions, episode_ends) from the replay."""

    import zarr

    root = zarr.open(str(zarr_path), mode="r")
    states = np.asarray(root["data"]["state"][:], dtype=np.float32)
    actions = np.asarray(root["data"]["action"][:], dtype=np.float32)
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if states.shape[-1] != STATE_DIM:
        raise ValueError(
            f"Expected Push-T states of width {STATE_DIM}, got {tuple(states.shape)}."
        )
    return states, actions, episode_ends


def episode_bounds(episode_ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-episode start and end transition indices, ends exclusive."""

    starts = np.concatenate(([0], episode_ends[:-1])).astype(np.int64)
    return starts, np.asarray(episode_ends, dtype=np.int64)


def valid_chunk_starts(episode_ends: np.ndarray, chunk_size: int) -> np.ndarray:
    """Chunk start indices whose whole window stays inside one episode.

    A chunk that straddled an episode boundary would ask the model to predict a
    continuation across a scene reset, so those starts are dropped.
    """

    starts, ends = episode_bounds(episode_ends)
    indices: list[int] = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        last = end - chunk_size
        if last >= start:
            indices.extend(range(start, last + 1))
    return np.asarray(indices, dtype=np.int64)


def _floored_std(std: np.ndarray) -> np.ndarray:
    """Std with a small floor, so a constant column cannot divide by zero."""

    return np.maximum(std, 1e-6).astype(np.float32)


@dataclass(frozen=True)
class Normalizer:
    """Observation and action coordinates, fitted once on the training replay.

    Action statistics are fitted on chunks rather than on the raw action
    stream, because the statistic that matters for a relative representation is
    the spread of waypoints around a replan-time anchor, not the spread of
    absolute positions.
    """

    feature_mean: np.ndarray
    feature_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    chunk_size: int

    @property
    def condition_dim(self) -> int:
        return int(self.feature_mean.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.action_mean.shape[0])

    @classmethod
    def fit(
        cls,
        states: np.ndarray,
        actions: np.ndarray,
        episode_ends: np.ndarray,
        *,
        chunk_size: int,
    ) -> Normalizer:
        starts = valid_chunk_starts(episode_ends, chunk_size)
        if len(starts) == 0:
            raise ValueError(
                f"No chunk of length {chunk_size} fits inside any episode."
            )
        offsets = np.arange(chunk_size)
        chunks = actions[starts[:, None] + offsets[None, :]]
        relative = chunks - agent_position(states[starts])[:, None, :]
        relative = relative.reshape(-1, actions.shape[-1])
        return cls(
            feature_mean=states.mean(axis=0).astype(np.float32),
            feature_std=_floored_std(states.std(axis=0)),
            action_mean=relative.mean(axis=0).astype(np.float32),
            action_std=_floored_std(relative.std(axis=0)),
            chunk_size=chunk_size,
        )

    def encode_observation(self, raw_state: np.ndarray) -> np.ndarray:
        raw_state = np.asarray(raw_state, dtype=np.float32)
        return ((raw_state - self.feature_mean) / self.feature_std).astype(np.float32)

    def encode_action(self, raw_chunk: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        """Absolute waypoints to normalized relative coordinates.

        anchor is the agent position at the replan that produced the chunk; it
        broadcasts over the horizon axis.
        """

        raw_chunk = np.asarray(raw_chunk, dtype=np.float32)
        centred = raw_chunk - np.asarray(anchor, dtype=np.float32)[..., None, :]
        return ((centred - self.action_mean) / self.action_std).astype(np.float32)

    def decode_action(
        self, normalized_chunk: np.ndarray, anchor: np.ndarray
    ) -> np.ndarray:
        """Inverse of encode_action, against the same anchor."""

        raw = (
            np.asarray(normalized_chunk, dtype=np.float32) * self.action_std
            + self.action_mean
        )
        return (raw + np.asarray(anchor, dtype=np.float32)[..., None, :]).astype(
            np.float32
        )

    def clip_action(self, raw_chunk: np.ndarray) -> np.ndarray:
        """Clip decoded absolute waypoints into the task's action box."""

        return np.clip(raw_chunk, ACTION_LOW, ACTION_HIGH)

    def to_metadata(self) -> dict:
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
            "chunk_size": int(self.chunk_size),
            "action_representation": "relative",
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> Normalizer:
        representation = metadata.get("action_representation")
        if representation != "relative":
            raise ValueError(
                "This package only reads relative-coordinate checkpoints; got "
                f"action_representation={representation!r}."
            )

        def as_array(key: str) -> np.ndarray:
            return np.asarray(metadata[key], dtype=np.float32)

        return cls(
            feature_mean=as_array("feature_mean"),
            feature_std=as_array("feature_std"),
            action_mean=as_array("action_mean"),
            action_std=as_array("action_std"),
            chunk_size=int(metadata["chunk_size"]),
        )


class ChunkDataset(Dataset):
    """Conditioning frames paired with normalized action chunks.

    Encoding is done once, here, rather than per __getitem__: it is a fixed
    function of the replay, and repeating it every epoch made the loader
    rather than the GPU the binding constraint. The arithmetic is identical
    either way; both encoders are vectorized over the leading axis.
    """

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        episode_ends: np.ndarray,
        *,
        normalizer: Normalizer,
    ) -> None:
        chunk_size = normalizer.chunk_size
        self.starts = valid_chunk_starts(episode_ends, chunk_size)
        offsets = np.arange(chunk_size)
        raw_chunks = actions[self.starts[:, None] + offsets[None, :]]
        conditions = normalizer.encode_observation(states[self.starts])
        chunks = normalizer.encode_action(
            raw_chunks, agent_position(states[self.starts])
        )
        self.conditions = torch.from_numpy(np.ascontiguousarray(conditions)).float()
        self.chunks = torch.from_numpy(np.ascontiguousarray(chunks)).float()

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.conditions[index], self.chunks[index]


def build_dataset(
    data_dir: Path, *, chunk_size: int
) -> tuple[ChunkDataset, Normalizer]:
    """Download if needed, fit coordinates, and return the training dataset."""

    zarr_path = download_replay(Path(data_dir))
    states, actions, episode_ends = load_replay(zarr_path)
    normalizer = Normalizer.fit(states, actions, episode_ends, chunk_size=chunk_size)
    dataset = ChunkDataset(states, actions, episode_ends, normalizer=normalizer)
    return dataset, normalizer
