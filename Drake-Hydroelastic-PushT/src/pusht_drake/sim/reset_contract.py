"""What "settled" means, shared by every reset path.

These numbers are a property of the task, not of a simulator: an episode may
begin only once the slider has stopped, the arm has stopped, and the pusher is
where it was asked to be. Two reset paths that each defined them separately
would agree until one drifted, and the disagreement would then be
indistinguishable from physics. So they live here, numpy only, and
pusht_drake.sim.reset re-exports them unchanged.

The tolerances themselves are not arbitrary and should not be rounded off:

- SLIDER_SPEED_TOL and ROBOT_SPEED_TOL differ by an order of magnitude
  because they are different units on different bodies: a slider drifting at
  1 mm/s is moving, an arm joint creeping at 1 mrad/s is not.
- PUSHER_POSITION_TOL is 5 mm against an IK that places the pusher within
  1 mm: the slack is for the settle, not for the solve.
- PUSHER_EXCURSION_LIMIT is the sweep guard. A pusher that strays 3 cm during
  a settle is the post-teleport sweep defect, not a slow settle, and it must
  raise rather than time out.
- QUATERNION_TOL catches a slider whose orientation has stopped being a
  rotation, which is how a diverged step first shows up in a free body.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Settled means: everything still, pusher where it should be.
SLIDER_SPEED_TOL = 1e-3  # m/s and rad/s, generalized velocity infinity-norm
ROBOT_SPEED_TOL = 1e-2  # rad/s
PUSHER_POSITION_TOL = 5e-3  # m; IK places the pusher within 1e-3 of the start pose
#: The pusher must never stray during settling; that is the sweep defect.
PUSHER_EXCURSION_LIMIT = 0.03  # m
QUATERNION_TOL = 1e-6

SETTLE_SLICE_S = 0.05
SETTLE_BUDGET_S = 0.25  # expected settle time (slider drops 5e-4 m); informational
SETTLE_TIMEOUT_S = 2.0  # hard failure beyond this


@dataclass
class ResetReport:
    """What one direct reset did and how long it took.

    forced_pushing reports Drake's planner state machine. A simulator with no
    such state machine reports False, so the record shape is the same either
    way.
    """

    settle_time_s: float
    max_pusher_excursion_m: float
    forced_pushing: bool
    checks: list[str] = field(default_factory=list)
