"""Inverse kinematics for the pusher tip.

All that survives of ``IiwaPlanner`` in the upstream Drake station
(Michaelszeng/diffusion-policy-drake; see NOTICE.md), a two-mode ``LeafSystem``
state machine (HOLD, PUSHING). The machine is dead weight here: the reset forces
PUSHING before the first ``AdvanceTo``, so HOLD is never entered, its update is
a no-op after the first reset, and its hold-position output raises
``AssertionError`` in the only reachable mode. It also cost a ``PortSwitch``, a
``RunFlagSystem`` and about thirty lines of ``builder.Connect`` to implement a
constant. The solve below is what it genuinely provided, and it needs no state
from the machine, only a plant every caller already holds.
"""

from __future__ import annotations

import numpy as np
from pydrake.all import InverseKinematics, MultibodyPlant, RigidTransform, RotationMatrix, Solve

#: Position and angle tolerance for the IK constraints. Tight enough that the
#: solution is the pose asked for, loose enough that the solver has room.
IK_TOLERANCE = 1e-3

PUSHER_TIP_FRAME = "pusher_end"


def solve_pusher_ik(
    plant: MultibodyPlant,
    pose: RigidTransform,
    q_nominal: np.ndarray,
    *,
    free_yaw: bool = False,
) -> np.ndarray:
    """Joint positions putting the pusher tip at ``pose``, nearest to ``q_nominal``.

    ``plant`` must be the controller plant: the arm and its pusher, nothing
    else. Handing it the full scene adds the T's degrees of freedom to the
    decision variables, with no error to say so.

    ``free_yaw`` relaxes the orientation constraint from "match this rotation"
    to "keep the tip pointing down", leaving rotation about the tool axis free.
    That is the right constraint for a cylindrical pusher, which has no
    meaningful yaw, and it is what makes J7 a usable redundancy rather than a
    joint that saturates: the fixed-yaw solve drives J7 into its limit after
    13-38 cm of planar travel.

    ``q_nominal`` is both the initial guess and the centre of a quadratic cost,
    so this returns the nearby solution rather than an arbitrary one. Seeding it
    badly lands the wrist on a different IK branch, and the arm then has to
    cross branches to get back.
    """
    ik = InverseKinematics(plant, with_joint_limits=True)
    tip = plant.GetFrameByName(PUSHER_TIP_FRAME)
    world = plant.world_frame()
    target = pose.translation()

    ik.AddPositionConstraint(
        tip,
        np.zeros(3),
        world,
        target - IK_TOLERANCE,
        target + IK_TOLERANCE,
    )

    if free_yaw:
        z_axis = np.array([0.0, 0.0, 1.0])
        # The tip's own z points up out of the cylinder, so "pointing down at the
        # table" is zero angle between it and world -z.
        ik.AddAngleBetweenVectorsConstraint(tip, z_axis, world, -z_axis, 0.0, IK_TOLERANCE)
    else:
        ik.AddOrientationConstraint(tip, RotationMatrix(), world, pose.rotation(), IK_TOLERANCE)

    program = ik.get_mutable_prog()
    q = ik.q()
    q_nominal = np.asarray(q_nominal, dtype=float)
    program.AddQuadraticErrorCost(np.identity(len(q)), q_nominal, q)
    program.SetInitialGuess(q, q_nominal)

    result = Solve(ik.prog())
    if not result.is_success():
        raise RuntimeError(
            f"no configuration reaches pusher tip {np.round(target, 4).tolist()} "
            f"(free_yaw={free_yaw}); the pose is probably outside the certified workspace"
        )
    return result.GetSolution(q)
