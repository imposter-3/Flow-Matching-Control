"""The 5-D observation vector, defined once for training and evaluation.

The policy conditions on a single frame that matches the gym-pusht state
observation field-for-field (gym-pusht returns
[agent_x, agent_y, block_x, block_y, block_angle % 2pi] in pixels; this is the
same tuple in world meters and radians):

    [pusher_x, pusher_y, slider_x, slider_y, slider_theta mod 2pi]

Two choices in that layout are load-bearing:

- The goal is not a feature. It is frozen at (0.500, 0.000, 0) for every
  episode, the way gym-pusht freezes its goal at (256, 256, pi/4). A constant
  feature would z-score to garbage: the std floors at 1e-6 and the normalized
  value explodes on the first float wobble.
- theta is reduced mod 2pi into [0, 2pi). The PlanarPose convention used
  elsewhere in this package wraps to (-pi, pi]; gym-pusht feeds
  angle % (2*np.pi). The discontinuity exists either way, and reducing mod 2pi
  keeps the feature distribution identical to the dataset the math was
  validated on, and identical between training frames and rollout frames.

Training (pusht_drake.fm.dataset) and rollout (pusht_drake.sim.rollout) both
call this function, so a train/eval observation skew is impossible by
construction.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi

#: Length of the observation vector.
OBSERVATION_DIM = 5


def build_observation(pusher_xy: np.ndarray, slider_pose: np.ndarray) -> np.ndarray:
    """[pusher_x, pusher_y, slider_x, slider_y, slider_theta mod 2pi] as float32.

    Args:
        pusher_xy: shape (2,), world meters. The dataset's state rows carry a
            third component (the pusher cylinder's free yaw); callers slice it
            off.
        slider_pose: shape (3,), world meters and radians, origin at the T's
            area centroid, which is the slider pose convention used throughout
            this package.
    """
    pusher_xy = np.asarray(pusher_xy, dtype=np.float64).reshape(2)
    slider_pose = np.asarray(slider_pose, dtype=np.float64).reshape(3)
    return np.array(
        [
            pusher_xy[0],
            pusher_xy[1],
            slider_pose[0],
            slider_pose[1],
            slider_pose[2] % TWO_PI,
        ],
        dtype=np.float32,
    )


def build_observations(pusher_xy: np.ndarray, slider_pose: np.ndarray) -> np.ndarray:
    """Vectorized build_observation over leading batch dimensions.

    pusher_xy: (..., 2); slider_pose: (..., 3). Returns (..., 5) float32.
    """
    pusher_xy = np.asarray(pusher_xy, dtype=np.float64)
    slider_pose = np.asarray(slider_pose, dtype=np.float64)
    if pusher_xy.shape[:-1] != slider_pose.shape[:-1]:
        raise ValueError(
            f"batch shapes differ: pusher {pusher_xy.shape} vs slider {slider_pose.shape}"
        )
    out = np.empty((*pusher_xy.shape[:-1], OBSERVATION_DIM), dtype=np.float32)
    out[..., 0:2] = pusher_xy[..., 0:2]
    out[..., 2:4] = slider_pose[..., 0:2]
    out[..., 4] = slider_pose[..., 2] % TWO_PI
    return out


def agent_position(observation: np.ndarray) -> np.ndarray:
    """The pusher position inside an observation: the action-decode anchor.

    Action chunks are learned relative to the pusher position at replan time,
    so the encoder and the decoder have to agree on where that anchor sits in
    the vector. Keeping the slice in one place is what makes them agree.
    """
    return np.asarray(observation)[..., 0:2]
