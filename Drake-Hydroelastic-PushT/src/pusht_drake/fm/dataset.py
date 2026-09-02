"""Sliding-window conditioning and normalized action chunks, from plain numpy.

The layering rule forbids pusht_drake.fm from importing zarr, so this dataset
takes arrays: the entry point loads the store through pusht_drake.store and
hands the columns in.

Observations are built here with the same build_observations the eval loop
uses (pusht_drake.observation), so a train/eval feature skew is impossible.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from pusht_drake.fm.representation import (
    Representation,
    build_valid_chunk_indices,
    episode_bounds,
)
from pusht_drake.observation import agent_position, build_observations


def observations_from_columns(states: np.ndarray, slider_states: np.ndarray) -> np.ndarray:
    """Dataset columns -> (T, 5) observation frames.

    states is the recorded pusher planar pose (T, 3); the third column is the
    cylinder's free yaw, which carries no information and is dropped.
    slider_states is the T's planar pose (T, 3) about its area centroid.
    """

    states = np.asarray(states)
    slider_states = np.asarray(slider_states)
    if states.ndim != 2 or states.shape[-1] < 2:
        raise ValueError(f"states must be (T, >=2), got {states.shape}")
    if slider_states.shape != (states.shape[0], 3):
        raise ValueError(f"slider_states must be ({states.shape[0]}, 3), got {slider_states.shape}")
    return build_observations(states[:, :2], slider_states)


class ChunkDataset(Dataset):
    """Fixed 3-tuple samples: (observation, relative chunk, warm flag).

    Chunks that would cross an episode boundary are dropped, never padded.
    Encoding is eager, a fixed function of the replay paid once instead of
    once per sample per epoch; measured, the lazy version was the loader
    bottleneck.

    The warm flag says whether a previous replan existed to warm-start from:
    1.0 when this chunk starts at least execution_horizon steps into its
    own episode, 0.0 otherwise. It is per-sample rather than global because
    every episode's first replan is cold, and a warm source must reduce to
    the cold one there exactly rather than be handed a preview of nothing.
    execution_horizon=None is open loop: nothing to warm from, so every row
    is flagged warm and no source reads it.
    """

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        episode_ends: np.ndarray,
        chunk_size: int,
        *,
        representation: Representation,
        episode_ids: np.ndarray | None = None,
        execution_horizon: int | None = None,
    ) -> None:
        if representation.prediction_horizon != chunk_size:
            raise ValueError(
                f"Representation horizon {representation.prediction_horizon} does not "
                f"match chunk_size {chunk_size}."
            )
        if execution_horizon is not None and not 1 <= execution_horizon <= chunk_size:
            raise ValueError(
                f"execution_horizon must be in [1, {chunk_size}], got {execution_horizon}."
            )
        self.chunk_size = chunk_size
        self.execution_horizon = execution_horizon
        self.representation = representation
        self.indices = build_valid_chunk_indices(episode_ends, chunk_size, episode_ids)
        if len(self.indices) == 0:
            raise ValueError("No valid chunks: every episode is shorter than the horizon.")

        raw_chunks = actions[self.indices[:, None] + np.arange(chunk_size)[None, :]]
        conditions = representation.encode_observation(observations[self.indices])
        chunks = representation.encode_action(
            raw_chunks, agent_position(observations[self.indices])
        )
        self.conditions = torch.from_numpy(np.ascontiguousarray(conditions)).float()
        self.chunks = torch.from_numpy(np.ascontiguousarray(chunks)).float()
        self.warm_flags = torch.from_numpy(self._warm_flags(episode_ends, execution_horizon))

    def _warm_flags(self, episode_ends: np.ndarray, execution_horizon: int | None) -> np.ndarray:
        """1.0 where a previous replan existed H_e steps earlier, same episode."""

        if execution_horizon is None:
            return np.ones(len(self.indices), dtype=np.float32)
        starts, ends = episode_bounds(episode_ends)
        # self.indices are global timesteps even when episode_ids subsets the
        # store, so the owning episode is found by search rather than assumed.
        owner = np.searchsorted(ends, self.indices, side="right")
        return (self.indices - execution_horizon >= starts[owner]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # One sample is a (condition_dim,) vector, an (H, A) chunk and a scalar
        # warm flag; B comes later.
        return (self.conditions[index], self.chunks[index], self.warm_flags[index])
