"""The seam a rollout crosses: observation in, absolute action chunk out.

The sim tier consumes policies only through this Protocol, and the check suite
forbids it from importing torch, so the torch-side adapter in pusht_drake.fm
and any future policy implementation meet the simulator here, in numpy, in
world meters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Policy(Protocol):
    """A chunk-predicting policy, as the evaluator sees it.

    reset(seed) starts a new episode: any per-episode sampling state (e.g.
    the flow-matching source generator) must be re-seeded from seed so an
    episode is a pure function of (checkpoint, env seed, action seed).

    predict_chunk(observation) maps one 5-D observation
    (pusht_drake.observation.build_observation) to a chunk of absolute pusher
    (x, y) targets in world meters, shape (horizon, 2), already clipped
    to the action set. The rollout executes it at the policy rate.

    How the chunk is physically executed (staircase hold, B-spline
    interpolation, a Cartesian servo) is the simulator layer's concern,
    selected by execution_mode at rig build time; nothing execution- or
    spline-specific belongs in this package or in any Policy implementation.

    Optional members, duck-typed by the executor. They are kept out of the
    Protocol so a policy that declares none of them stays a valid Policy and
    isinstance keeps its current meaning, the same pattern
    pusht_drake.sim.harness uses for policy.representation:

    execution_horizon: int
        Re-query the policy after this many waypoints. If it is absent,
        execution is open loop: the whole chunk runs, then a fresh prediction.
        This is a property of the trained artifact rather than of the
        simulator, because a policy that warm-starts from its own previous
        forecast anchors that cache on being called at exactly this cadence.
        So the policy owns the value, and the executor raises when a rig-side
        setting disagrees.
    notify_committed(n: int) -> None
        Called once per replan with the number of waypoints actually
        committed. Pushed rather than inferred: nothing observable downstream
        distinguishes an off-by-one in it from ordinary motion.
    invalidate_warm_cache() -> None
        Called when the executing plan is discarded out of band (an episode
        reset), so any cached forecast of it is dropped.
    """

    def reset(self, seed: int) -> None: ...

    def predict_chunk(self, observation: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class RolloutLimits:
    """Episode termination and scoring constants, taken from gym-pusht.

    max_steps: truncation, in policy steps (300 @ 10 Hz = 30 s, gym-pusht's
    TimeLimit).
    terminate_coverage: env-native early termination, 0.95 as in gym-pusht.
    score_tau: the scoring threshold. reward = clip(coverage/tau, 0, 1) and
    success = max_coverage > tau. It is 0.90, chosen against the human-demo
    coverage ceiling and re-calibrated against this repo's own demonstrations
    before being frozen; see the README's scoring section.
    """

    max_steps: int = 300
    terminate_coverage: float = 0.95
    score_tau: float = 0.90

    def __post_init__(self) -> None:
        if not 0.0 < self.score_tau <= self.terminate_coverage <= 1.0:
            raise ValueError(
                f"need 0 < score_tau <= terminate_coverage <= 1, got "
                f"{self.score_tau} / {self.terminate_coverage}"
            )
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
