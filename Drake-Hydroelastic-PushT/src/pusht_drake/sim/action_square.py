"""The rectangular Cartesian action set: the square teleop and policies live in.

Three nested regions:

    W_robot   the certified irregular kinematic fence (SafeWorkspace), safety net
      contains
    R_action  this module, a clean axis-aligned square: the normal command domain
      contains
    W_T       the T task region, derived later from R_action and the T's envelope

clamp is a per-axis clip, which is the exact Euclidean projection onto an
axis-aligned box. Because each axis is clipped independently, the boundary
behaviour the teleop needs falls out with no special cases: pushing outward
saturates on the wall, the tangential component passes through untouched (the
target slides along the wall), and the first inward increment leaves the wall
immediately. Combined with the guard's store-after-clamp rule there is no windup.

The square does not extend SafeWorkspace. The fence's alternating projection is
approximate near its non-convex inner rim while a box projection is exact, so
the two stay separate layers with separate guarantees. The invariant that the
square sits inside the fence is certified offline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ActionSquare", "add_action_square_visualization"]


@dataclass(frozen=True)
class ActionSquare:
    """An axis-aligned square action region in world coordinates, meters."""

    center: tuple[float, float]
    side: float

    @property
    def x_min(self) -> float:
        return self.center[0] - self.side / 2.0

    @property
    def x_max(self) -> float:
        return self.center[0] + self.side / 2.0

    @property
    def y_min(self) -> float:
        return self.center[1] - self.side / 2.0

    @property
    def y_max(self) -> float:
        return self.center[1] + self.side / 2.0

    @classmethod
    def from_config(cls, cfg) -> ActionSquare:
        """Build from the config's action_square block (omegaconf or dict)."""
        block = cfg["action_square"] if isinstance(cfg, dict) else cfg.action_square
        centre = tuple(float(v) for v in block["center"])
        return cls(center=(centre[0], centre[1]), side=float(block["side"]))

    def clamp(self, xy) -> np.ndarray:
        """Exact projection onto the square: independent per-axis clip."""
        p = np.asarray(xy, dtype=float).reshape(2)
        return np.array(
            [
                min(max(p[0], self.x_min), self.x_max),
                min(max(p[1], self.y_min), self.y_max),
            ]
        )

    def contains(self, xy, tol: float = 0.0) -> bool:
        p = np.asarray(xy, dtype=float).reshape(2)
        return bool(
            self.x_min - tol <= p[0] <= self.x_max + tol
            and self.y_min - tol <= p[1] <= self.y_max + tol
        )

    def boundary_polyline(self) -> np.ndarray:
        """Closed (5, 2) loop of the square's corners, world coordinates."""
        return np.array(
            [
                [self.x_min, self.y_min],
                [self.x_max, self.y_min],
                [self.x_max, self.y_max],
                [self.x_min, self.y_max],
                [self.x_min, self.y_min],
            ]
        )


def add_action_square_visualization(
    meshcat,
    square: ActionSquare,
    *,
    path: str = "workspace/action_square",
    z_m: float = 0.0015,
    line_width: float = 2.0,
) -> None:
    """Draw the action square: thin light-green boundary, no fill, no collision.

    This is the only region normal teleoperation shows. It is drawn from the same
    ActionSquare instance the guard clamps against, so the visible wall and the
    enforced wall cannot disagree. ~1.5 mm above the tabletop to avoid z-fighting.
    """
    from pydrake.all import Rgba

    loop = square.boundary_polyline()
    vertices = np.vstack([loop.T, np.full(len(loop), z_m)])
    meshcat.SetLine(path, vertices, line_width=line_width, rgba=Rgba(0.60, 0.95, 0.60, 1.0))
