"""The versioned coordinate contract for observations and actions.

The action bounds are per-axis vectors carried as fitted state: the action set
here is the frozen 480 mm square and its x and y bounds differ, where the
canonical 512-px Push-T task uses the same scalar [0, 512] range on both axes.
The bounds ride in the metadata, so a checkpoint stays self-describing about
the box it was trained to clip to.

This is the only versioned contract that crosses process boundaries. It is
written into every checkpoint and read back whenever one is scored, so a change
here invalidates trained artifacts.

The observation is a single raw 5-D frame (pusht_drake.observation), z-scored
per feature. Actions are agent-relative: one replan-time pusher position is
subtracted from every future waypoint before the pooled z-score. That relative
form is used unconditionally.

Free of storage and of torch: fitting, encoding, and decoding coordinates has
nothing to do with where a replay is stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pusht_drake.observation import agent_position

METADATA_VERSION = 1


def episode_bounds(episode_ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-episode [start, end) transition indices."""

    starts = np.concatenate(([0], episode_ends[:-1])).astype(np.int64)
    return starts, np.asarray(episode_ends, dtype=np.int64)


def build_valid_chunk_indices(
    episode_ends: np.ndarray,
    chunk_size: int,
    episode_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Chunk starts that do not cross episode boundaries; a short tail is dropped, not padded."""

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    starts, ends = episode_bounds(episode_ends)
    selected = range(len(ends)) if episode_ids is None else sorted(map(int, episode_ids))
    indices: list[int] = []
    for episode in selected:
        start, end = int(starts[episode]), int(ends[episode])
        last_start = end - chunk_size
        if last_start >= start:
            indices.extend(range(start, last_start + 1))
    return np.asarray(indices, dtype=np.int64)


@dataclass(frozen=True)
class Representation:
    """Immutable, train-only observation and action coordinates for one experiment."""

    prediction_horizon: int
    feature_mean: np.ndarray
    feature_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    action_low: np.ndarray  # (A,) absolute clip box, world meters
    action_high: np.ndarray  # (A,)

    def __post_init__(self) -> None:
        if self.feature_mean.ndim != 1 or self.feature_std.shape != self.feature_mean.shape:
            raise ValueError(
                "Feature statistics must be matching 1-D arrays, got "
                f"{tuple(self.feature_mean.shape)} and {tuple(self.feature_std.shape)}."
            )
        if not (
            self.action_low.shape
            == self.action_high.shape
            == self.action_mean.shape
            == self.action_std.shape
        ):
            raise ValueError(
                "Action statistics and bounds must share one shape, got "
                f"mean {tuple(self.action_mean.shape)}, low {tuple(self.action_low.shape)}, "
                f"high {tuple(self.action_high.shape)}."
            )
        if np.any(self.action_low >= self.action_high):
            raise ValueError("action_low must be strictly below action_high per axis.")

    @property
    def condition_dim(self) -> int:
        """Velocity-model conditioning width, derived from the fitted statistics."""

        return int(self.feature_mean.shape[0])

    @property
    def action_dim(self) -> int:
        """Width of one action. Derived from the statistics, never serialized."""

        return int(self.action_mean.shape[0])

    @staticmethod
    def _safe_std(std: np.ndarray) -> np.ndarray:
        return np.maximum(std, 1e-6).astype(np.float32)

    @classmethod
    def fit(
        cls,
        observations: np.ndarray,
        actions: np.ndarray,
        episode_ends: np.ndarray,
        *,
        prediction_horizon: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        train_episodes: np.ndarray | None = None,
    ) -> Representation:
        """Fit every statistic on training episodes only.

        observations are the 5-D frames from
        pusht_drake.observation.build_observations; their first two
        coordinates are the pusher position, which anchors the relative chunks.
        """

        starts, ends = episode_bounds(episode_ends)
        episodes = np.arange(len(ends)) if train_episodes is None else np.asarray(train_episodes)
        timesteps = np.concatenate([np.arange(int(starts[e]), int(ends[e])) for e in episodes])
        features = np.asarray(observations, dtype=np.float32)[timesteps]

        chunk_starts = build_valid_chunk_indices(episode_ends, prediction_horizon, episodes)
        if len(chunk_starts) == 0:
            raise ValueError(
                f"No training chunk of length {prediction_horizon} fits inside "
                "the selected episodes."
            )
        offsets = np.arange(prediction_horizon)
        chunks = actions[chunk_starts[:, None] + offsets[None, :]]
        # Subtract one replan-time pusher position from every future waypoint.
        anchors = agent_position(observations[chunk_starts])[:, None, :]
        relative_actions = (chunks - anchors).reshape(-1, actions.shape[-1])

        return cls(
            prediction_horizon=prediction_horizon,
            feature_mean=features.mean(axis=0).astype(np.float32),
            feature_std=cls._safe_std(features.std(axis=0)),
            action_mean=relative_actions.mean(axis=0).astype(np.float32),
            action_std=cls._safe_std(relative_actions.std(axis=0)),
            action_low=np.asarray(action_low, dtype=np.float32).reshape(-1),
            action_high=np.asarray(action_high, dtype=np.float32).reshape(-1),
        )

    def encode_observation(self, raw_state: np.ndarray) -> np.ndarray:
        """Map raw observation frames to normalized conditioning features."""

        raw_state = np.asarray(raw_state, dtype=np.float32)
        if raw_state.shape[-1] != self.feature_mean.shape[-1]:
            raise ValueError(
                f"Expected raw state shape (..., {self.feature_mean.shape[-1]}), "
                f"got {tuple(raw_state.shape)}."
            )
        return ((raw_state - self.feature_mean) / self.feature_std).astype(np.float32)

    def encode_action(
        self,
        raw_chunk: np.ndarray,
        agent_position: np.ndarray,
    ) -> np.ndarray:
        """Map raw absolute action chunks into normalized learning coordinates.

        The anchor is required: encode and decode must use the same replan-time
        pusher position, and keeping the argument on both makes a
        desynchronized pair raise TypeError instead of skewing every decoded
        waypoint with no error.
        """

        raw_chunk = np.asarray(raw_chunk, dtype=np.float32)
        raw_chunk = raw_chunk - np.asarray(agent_position, dtype=np.float32)[..., None, :]
        return ((raw_chunk - self.action_mean) / self.action_std).astype(np.float32)

    def decode_action(
        self,
        normalized_chunk: np.ndarray,
        agent_position: np.ndarray,
    ) -> np.ndarray:
        """Invert encode_action against the original replan-time position."""

        raw = np.asarray(normalized_chunk, dtype=np.float32) * self.action_std + self.action_mean
        raw = raw + np.asarray(agent_position, dtype=np.float32)[..., None, :]
        return raw.astype(np.float32)

    def clip_action(self, raw_chunk: np.ndarray) -> np.ndarray:
        """Clip decoded absolute actions into the task's action box (per axis)."""

        return np.clip(raw_chunk, self.action_low, self.action_high)

    def to_metadata(self) -> dict[str, Any]:
        """Serialize everything needed to rebuild identical coordinates."""

        return {
            "metadata_version": METADATA_VERSION,
            "observation_representation": "raw_pose",
            "observation_horizon": 1,
            "action_representation": "relative",
            "prediction_horizon": self.prediction_horizon,
            "condition_dim": self.condition_dim,
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> Representation:
        """Rebuild a representation, rejecting metadata this code cannot honor.

        Strict on purpose: this package wrote every payload it will ever read,
        so a missing key is a bug and raises KeyError rather than being papered
        over with a default.
        """

        version = metadata["metadata_version"]
        if version != METADATA_VERSION:
            raise ValueError(
                f"Unsupported representation metadata version {version!r}; expected "
                f"{METADATA_VERSION}."
            )
        if (
            metadata["observation_representation"] != "raw_pose"
            or int(metadata["observation_horizon"]) != 1
        ):
            raise ValueError("This checkpoint uses an observation representation this code lacks.")
        if metadata["action_representation"] != "relative":
            raise ValueError(
                "This package only rebuilds agent-relative action coordinates; got "
                f"action_representation={metadata['action_representation']!r}."
            )
        representation = cls(
            prediction_horizon=int(metadata["prediction_horizon"]),
            feature_mean=np.asarray(metadata["feature_mean"], dtype=np.float32),
            feature_std=np.asarray(metadata["feature_std"], dtype=np.float32),
            action_mean=np.asarray(metadata["action_mean"], dtype=np.float32),
            action_std=np.asarray(metadata["action_std"], dtype=np.float32),
            action_low=np.asarray(metadata["action_low"], dtype=np.float32),
            action_high=np.asarray(metadata["action_high"], dtype=np.float32),
        )
        stored_dim = int(metadata["condition_dim"])
        if stored_dim != representation.condition_dim:
            raise ValueError(
                f"Checkpoint condition_dim {stored_dim} disagrees with the rebuilt "
                f"representation ({representation.condition_dim})."
            )
        return representation
