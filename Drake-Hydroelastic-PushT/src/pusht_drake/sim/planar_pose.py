"""A pose on the table: ``(x, y, theta)``, and its conversions to Drake.

The task is planar, but Drake is not, so something has to carry the convention
for how a 3-vector becomes a ``RigidTransform``. This is that thing.

What the three numbers mean depends on whose pose it is, and conflating the two
readings is an easy mistake:

- For the slider, it is the pose of the T's area centroid, which sits 25.714 mm
  below the crossbar centre. Drake reports the body pose in the body frame, and
  the generated SDF puts that frame on the centroid, so this is what comes out
  of the plant and what goes into coverage, the dataset and the observation.
- For the pusher, only ``x`` and ``y`` mean anything. Its ``theta`` is the
  cylinder's free yaw, which rotates with the arm's last link and carries no
  task information. Nothing reads it.

``to_pose`` points the z-axis down by default (roll = pi): the pusher's frame is
the arm's, reaching down at the table, and differential IK commands a pose in
that convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydrake.common.eigen_geometry import Quaternion
from pydrake.math import RigidTransform, RollPitchYaw

_Z_AXIS = 2


@dataclass(frozen=True)
class PlanarPose:
    """A planar pose. Frozen: these are values, and they get stored in records."""

    x: float
    y: float
    theta: float

    # -- from Drake -------------------------------------------------------

    @classmethod
    def from_pose(cls, pose: RigidTransform) -> PlanarPose:
        """Project a ``RigidTransform``: drop z, keep the rotation about z."""
        translation = pose.translation()
        theta = RollPitchYaw(pose.rotation()).vector()[_Z_AXIS]
        return cls(float(translation[0]), float(translation[1]), float(theta))

    @classmethod
    def from_generalized_coords(cls, q: np.ndarray) -> PlanarPose:
        """Project a free body's ``q = [quaternion, translation]``.

        The quaternion is renormalized first: a plant integrated for thousands
        of steps drifts off the unit sphere, and ``Quaternion`` rejects that.
        """
        q = np.asarray(q, dtype=float)
        if len(q) == 2:
            return cls(float(q[0]), float(q[1]), 0.0)
        wxyz = q[0:4] / np.linalg.norm(q[0:4])
        theta = RollPitchYaw(Quaternion(wxyz)).vector()[_Z_AXIS]
        return cls(float(q[4]), float(q[5]), float(theta))

    # -- to Drake ---------------------------------------------------------

    def to_pose(self, z_value: float, z_axis_is_positive: bool = False) -> RigidTransform:
        """Lift to a ``RigidTransform`` at height ``z_value``.

        ``z_axis_is_positive=False`` (the default) points the z-axis down, which
        is the pusher/end-effector convention. Pass ``True`` for a body that
        sits on the table the right way up, such as the T.
        """
        roll = 0.0 if z_axis_is_positive else np.pi
        return RigidTransform(
            RollPitchYaw(np.array([roll, 0.0, self.theta])),
            np.array([self.x, self.y, z_value]),
        )

    def to_generalized_coords(self, z_value: float, z_axis_is_positive: bool = False) -> np.ndarray:
        """The free-body ``q = [quaternion, translation]`` for this pose."""
        pose = self.to_pose(z_value, z_axis_is_positive)
        return np.concatenate((pose.rotation().ToQuaternion().wxyz(), pose.translation()))

    # -- as numbers -------------------------------------------------------

    def vector(self) -> np.ndarray:
        """``[x, y, theta]``, the form the dataset and the observation use."""
        return np.array([self.x, self.y, self.theta])

    def pos(self) -> np.ndarray:
        """``[[x], [y]]``, a column, for matrix arithmetic."""
        return np.array([[self.x], [self.y]])

    def two_d_rot_matrix(self) -> np.ndarray:
        c, s = np.cos(self.theta), np.sin(self.theta)
        return np.array([[c, -s], [s, c]])

    def __str__(self) -> str:
        return f"x: {self.x}, y: {self.y}, theta: {self.theta}"

    def __eq__(self, other: object) -> bool:
        """Approximate on purpose: these come out of a solver, not a literal."""
        if not isinstance(other, PlanarPose):
            return NotImplemented
        return bool(np.allclose(self.vector(), other.vector()))
