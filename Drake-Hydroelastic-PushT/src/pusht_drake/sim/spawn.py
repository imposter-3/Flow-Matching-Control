"""The frozen task-initialization distribution, shared by teleop and evaluation.

One definition, used identically for data collection and every future evaluation
run:

* T slider: its COM uniform in the 400 mm t_spawn square, yaw uniform in
  (-pi, pi). Only the COM is bounded, so the body may overhang the square and
  even the action set's edge; the pusher, not the T, is what stays confined.
* pusher: uniform in the 440 mm pusher_spawn square.
* Both squares share the action square's centre. The only rejection is a
  physical T-pusher overlap, resolved by resampling the pusher, so the T's
  marginal distribution stays exactly uniform and the pusher's is uniform
  conditioned on not starting inside the T.

Everything is derived from the config's task_init block; there are no module
constants to drift. Determinism: pass a seeded numpy.random.Generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pusht_drake.sim.tblock import PUSHT_T

__all__ = ["TaskInit", "sample_task_init"]

# The pusher must start clear of the T by this much (surface to surface). Zero
# would allow spawning in marginal contact; one pusher radius of slack keeps the
# first commanded motion contact-free without visibly biasing the distribution.
_PUSHER_CLEARANCE_M = 0.015


@dataclass(frozen=True)
class TaskInit:
    """One sampled episode initialization."""

    slider_pose: tuple[float, float, float]  # x, y, theta; COM pose, world
    pusher_xy: tuple[float, float]


def _point_to_polygon_distance(point: np.ndarray, poly: np.ndarray) -> float:
    """Distance from a point to a polygon boundary; negative would mean inside.

    Returns 0.0 for interior points (we only need "closer than clearance").
    """
    n = len(poly)
    inside = False
    best = np.inf
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        t = float(np.clip(np.dot(point - a, e) / max(float(np.dot(e, e)), 1e-15), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(point - (a + t * e))))
        if (a[1] > point[1]) != (b[1] > point[1]):
            x_cross = a[0] + (point[1] - a[1]) / (b[1] - a[1] + 1e-30) * (b[0] - a[0])
            if point[0] < x_cross:
                inside = not inside
    return 0.0 if inside else best


def _pusher_overlaps_t(pusher_xy, slider_pose, pusher_radius: float) -> bool:
    x, y, th = slider_pose
    c, s = np.cos(th), np.sin(th)
    world = (np.array([[c, -s], [s, c]]) @ PUSHT_T.outline().T).T + [x, y]
    d = _point_to_polygon_distance(np.asarray(pusher_xy, dtype=float), world)
    return d < pusher_radius + _PUSHER_CLEARANCE_M


def sample_task_init(cfg, rng: np.random.Generator, max_pusher_tries: int = 1000) -> TaskInit:
    """Sample one episode initialization from the config's task_init block."""
    block = cfg["task_init"] if isinstance(cfg, dict) else cfg.task_init
    t_c = [float(v) for v in block["t_spawn"]["center"]]
    t_h = float(block["t_spawn"]["side"]) / 2.0
    p_c = [float(v) for v in block["pusher_spawn"]["center"]]
    p_h = float(block["pusher_spawn"]["side"]) / 2.0
    pusher_radius = float(cfg["pusher_radius"] if isinstance(cfg, dict) else cfg.pusher_radius)

    slider = (
        float(rng.uniform(t_c[0] - t_h, t_c[0] + t_h)),
        float(rng.uniform(t_c[1] - t_h, t_c[1] + t_h)),
        float(rng.uniform(-np.pi + 1e-3, np.pi - 1e-3)),
    )
    for _ in range(max_pusher_tries):
        pusher = (
            float(rng.uniform(p_c[0] - p_h, p_c[0] + p_h)),
            float(rng.uniform(p_c[1] - p_h, p_c[1] + p_h)),
        )
        if not _pusher_overlaps_t(pusher, slider, pusher_radius):
            return TaskInit(slider_pose=slider, pusher_xy=pusher)
    raise RuntimeError(
        f"no clear pusher start in {max_pusher_tries} draws -- geometrically implausible "
        f"(T covers < 4% of the pusher spawn box); investigate the config"
    )
