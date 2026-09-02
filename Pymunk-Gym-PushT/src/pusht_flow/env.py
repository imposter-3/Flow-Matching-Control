"""The Push-T environment and its scoring wrapper.

gym_pusht is imported inside make_env, so importing this module does not pull in
the simulator.

The wrapper changes the reported numbers, not the dynamics. Reward becomes
clip(coverage / 0.90, 0, 1), and success at 0.90 is computed downstream from the
recorded coverages. Termination is left at the environment's native
coverage > 0.95; overriding it would change episode lengths, and with them the
trajectories. An episode run here is step for step the episode the unmodified
environment produces, with a different score attached.

The rollout loop records raw per-step coverage, so both the 0.90 reading and the
stricter 0.95 one can be recovered from the same run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pusht_flow.config import ENV_ID, SUCCESS_THRESHOLD


def make_env():
    """Construct the scored Push-T environment."""

    import gym_pusht  # noqa: F401  (registers the env id)
    import gymnasium as gym

    class RescoredPushT(gym.Wrapper):
        def step(self, action):
            observation, _, terminated, truncated, info = self.env.step(action)
            # Indexed rather than .get(): a missing coverage key would
            # otherwise score every episode zero.
            coverage = info["coverage"]
            reward = float(np.clip(coverage / SUCCESS_THRESHOLD, 0.0, 1.0))
            return observation, reward, terminated, truncated, info

    return RescoredPushT(gym.make(ENV_ID, obs_type="state", render_mode="rgb_array"))


@dataclass
class EpisodeResult:
    """One rollout's contribution to the results table."""

    env_seed: int
    action_seed: int
    max_coverage: float
    final_coverage: float
    mean_reward: float
    length: int
    terminated: bool
    truncated: bool
    num_replans: int

    def success(self, threshold: float) -> bool:
        return bool(self.max_coverage > threshold)
