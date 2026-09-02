"""The push-start joint configuration: solved once, conditioned, cached.

pusher_start_pose is fixed, so its IK solution is a constant of the task, and there
is no reason to re-solve it every episode. This module solves it once per process
using upstream's own yaw-free IK formulation (pusht_drake.sim.ik with
free_yaw=True: position of pusher_end within 1e-3 m, pusher z-axis down, yaw
unconstrained) with the wrist nominal mid-range, and reports how
well-conditioned the resulting configuration is for planar pushing:

- sigma_min of the pusher's planar Jacobian J_xy (the 2x7 translational x/y
  rows): how well the arm can realize commanded planar velocities. Larger is better.
- distance to joint limits: teleop drives the arm around; a start pose hugging a
  limit invites Differential IK saturation.
- elbow sign: the evaluation's ELBOW_DOWN failure fires when q[3] > +5 deg, so
  the start configuration must keep the elbow well clear of that.

Why yaw-free with J7 = 0 rather than upstream's full-orientation solve: the pusher
is a cylinder, so its yaw is physically meaningless, and upstream's solution parks
J7 at 171.2 deg of its +/-175 deg range, 3.8 deg from the limit the fixed-yaw
DiffIK then drives it into. Under the yaw-relaxed controller
(pusht_drake.sim.diffik) J7 is the free self-motion DOF; starting it mid-range
gives the wrist its full travel. The pusher's (x, y, z) and verticality are
identical to upstream; only the wrist self-motion differs, along with the observed
pusher theta (link-7-dependent, consumed by nothing downstream).
the check suite pins the solution and a conditioning floor so an unnoticed
model drift fails CI instead of degrading teleop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydrake.all import JacobianWrtVariable


@dataclass(frozen=True)
class PushStartConfiguration:
    """The once-solved push-start joint configuration and its conditioning."""

    q: np.ndarray  # (7,) joint positions
    sigma_min_jxy: float  # smallest singular value of the 2x7 planar Jacobian
    min_limit_margin_rad: float  # min over joints of distance to the nearer limit
    elbow_angle_rad: float  # q[3]; must stay well below +5 deg (ELBOW_DOWN)

    def summary(self) -> str:
        return (
            f"q_push_start = {np.round(self.q, 5).tolist()}\n"
            f"  sigma_min(J_xy)    = {self.sigma_min_jxy:.4f}  (m/s per rad/s)\n"
            f"  joint-limit margin = {np.degrees(self.min_limit_margin_rad):.1f} deg\n"
            f"  elbow angle q[3]   = {np.degrees(self.elbow_angle_rad):.1f} deg "
            f"(ELBOW_DOWN fires above +5 deg)"
        )


def solve_push_start_configuration(plant, config) -> PushStartConfiguration:
    """Solve the push-start IK once and measure its conditioning.

    plant is the controller plant: the arm and its welded pusher, nothing
    else. Handing it the full scene adds the T's degrees of freedom to the
    decision variables, with no error to say so. The nominal is
    default_joint_positions with J7 zeroed (see module docstring).
    """
    from pusht_drake.sim.ik import solve_pusher_ik

    nominal = np.asarray(config.default_joint_positions, dtype=float).copy()
    nominal[6] = 0.0
    q = np.asarray(
        solve_pusher_ik(
            plant,
            config.pusher_start_pose.to_pose(config.pusher_z_offset),
            nominal,
            free_yaw=True,
        )
    ).ravel()

    context = plant.CreateDefaultContext()
    plant.SetPositions(context, q)
    jacobian = plant.CalcJacobianTranslationalVelocity(
        context,
        JacobianWrtVariable.kQDot,
        plant.GetFrameByName("pusher_end"),
        np.zeros(3),
        plant.world_frame(),
        plant.world_frame(),
    )
    sigma_min = float(np.linalg.svd(jacobian[:2, :], compute_uv=False)[-1])

    lower, upper = plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
    margin = float(np.min(np.minimum(q - lower, upper - q)))

    return PushStartConfiguration(
        q=q,
        sigma_min_jxy=sigma_min,
        min_limit_margin_rad=margin,
        elbow_angle_rad=float(q[3]),
    )
