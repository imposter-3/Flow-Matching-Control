"""The one reset used by simulation: teleport to push-start, then settle.

This is Zeng's own newest reset pattern (rl_push_t/envs/push_t_gym_env.py::reset,
2026-03) promoted to the primary path for teleop and evaluation, instead of the
GCS+Toppra startup sequence. The two paths reach an equivalent pushing-entry state
(same IK configuration, zero velocities, same slider distribution, DiffIK synced),
but the direct reset is exact, deterministic, has no Toppra exposure, and lets us
close a real upstream defect:

The post-teleport command sweep. JointVelocityClamp keeps its previous
command in discrete state. Upstream's env.set_robot_position syncs the plant and
the Differential IK integrator on a teleport, but not the clamp. Its stale
last_command makes the command walk from the old arm pose back to push-start at
the joint velocity limit while the plant is already teleported, so the driver yanks
the physical arm away from push-start and sweeps it across the table where the new
slider was just placed. The reset restores the clamp's first-call sentinel,
which makes the next command pass through unclamped (the clamp's own startup
behavior).

What used to be here and is not any more: the hand-off to Drake's planner state
machine. Upstream's `IiwaPlanner` spent an initial delay holding before it
yielded to differential IK, and this reset forced it past that hold before the
first `AdvanceTo`. With the planner gone the hand-off
is unconditional, and the only thing that survives from that branch is calling
`Simulator.Initialize()` exactly once, on the first reset, so the initialization
events run against the teleported state rather than the spawn state. (Drake would
call it from the first `AdvanceTo` anyway; doing it here keeps it explicit and
keeps `_reanchor_realtime_pacing`'s reasoning about the pacing anchor true.)

Settling is state-based with a timeout, not a fixed pause: we advance in small
slices until the slider and arm are quiescent and the pusher is at its start, and
raise if that does not happen within the timeout. Stability invariants
(finite state, normalized slider quaternion, no pusher excursion) are asserted on
every slice, so a violent reset raises rather than being recorded as settled.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# What "settled" means is a property of the task, so it lives in
# pusht_drake.sim.reset_contract and is shared with every other backend's reset rather
# than transcribed into it. Re-exported here so every existing import site and
# the check suite keep working unchanged.
from pusht_drake.sim.reset_contract import (  # noqa: E402
    PUSHER_EXCURSION_LIMIT,
    PUSHER_POSITION_TOL,
    QUATERNION_TOL,
    ROBOT_SPEED_TOL,
    SETTLE_BUDGET_S,
    SETTLE_SLICE_S,
    SETTLE_TIMEOUT_S,
    SLIDER_SPEED_TOL,
    ResetReport,
)

__all__ = [
    "PUSHER_EXCURSION_LIMIT",
    "PUSHER_POSITION_TOL",
    "QUATERNION_TOL",
    "ROBOT_SPEED_TOL",
    "SETTLE_BUDGET_S",
    "SETTLE_SLICE_S",
    "SETTLE_TIMEOUT_S",
    "SLIDER_SPEED_TOL",
    "DirectResetter",
    "ResetReport",
]

#: Largest elbow-family step the reset IK may take away from the scheduled
#: nominal before the solution is treated as a different IK branch.
ELBOW_BRANCH_TOL_RAD = 0.35


class DirectResetter:
    """Resets the environment straight to the push-start configuration."""

    def __init__(self, station, config, push_start_q: np.ndarray) -> None:
        self.station = station
        self.config = config
        self.q_push_start = np.asarray(push_start_q).ravel()

        self._simulator = station.simulator
        self._start_xy = np.array([config.pusher_start_pose.x, config.pusher_start_pose.y])
        self._initialized = False

    # ---- the reset ---------------------------------------------------------

    def _reset_clamp_state(self) -> None:
        """Restore the clamp's first-call sentinel so the next command passes through.

        Closes the post-teleport command sweep (module docstring).
        """
        clamp = self.station.clamp
        clamp.reset(clamp.GetMyMutableContextFromRoot(self.station.context))

    def _initialize_once(self) -> bool:
        """Run the diagram's initialization events, on the first reset only.

        Reports whether this reset was the one that did it, which is what the
        `forced_pushing` field of the report has always meant: the first reset
        does the one-time hand-off into closed-loop control.
        """
        if self._initialized:
            return False
        self._simulator.Initialize()
        self._initialized = True
        return True

    def solve_start_configuration(self, pusher_xy) -> np.ndarray:
        """Arm posture for an arbitrary pusher start: the centering equilibrium.

        Per-episode pusher starts are sampled (pusht_drake.sim.spawn), so the reset
        posture must be solved per reset. The posture that settles is not free to
        choose: the controller's joint-centering term relaxes the null space
        toward the radius-only scheduled nominal (diffik.py), so the only
        zero-velocity posture at a given pusher target is the constrained
        minimizer of ||q - q_nom(r)||^2 on the task manifold. Solving the IK with
        the uncorrected nominal as both cost reference and initial guess yields
        exactly that minimizer: the arm starts where the pull is already zero
        and settles immediately. Seeding anything else (measured with the
        azimuth-rotated branch posture) leaves a residual null-space pull and the
        settle criterion is never met at |azimuth| > ~15 deg.

        The guard checks the elbow-family joints (q2, q4, q6) against the
        schedule: those carry the branch identity. J1/J3/J5 absorb the target's
        azimuth by design and legitimately differ from the azimuth-0 schedule.
        """
        from pusht_drake.sim.ik import solve_pusher_ik
        from pusht_drake.sim.planar_pose import PlanarPose

        schedule = getattr(self.station.diff_ik, "_schedule", None)
        if schedule is None:
            return self.q_push_start
        radii, postures, base_pose = schedule
        dx = float(pusher_xy[0]) - float(base_pose[0])
        dy = float(pusher_xy[1]) - float(base_pose[1])
        r = float(np.hypot(dx, dy))
        q_nom = np.array([np.interp(r, radii, postures[:, j]) for j in range(postures.shape[1])])

        pose = PlanarPose(float(pusher_xy[0]), float(pusher_xy[1]), 0.0)
        q = np.asarray(
            solve_pusher_ik(
                self.station.scene.controller_plant,
                pose.to_pose(self.config.pusher_z_offset),
                q_nom,
                free_yaw=True,
            )
        ).ravel()
        elbow = [1, 3, 5]
        step = float(np.max(np.abs(q[elbow] - q_nom[elbow])))
        if step > ELBOW_BRANCH_TOL_RAD:
            raise RuntimeError(
                f"reset IK left the certified branch (elbow-family step {step:.3f} rad "
                f"at pusher {tuple(round(v, 3) for v in pusher_xy)})"
            )
        return q

    def reset(self, slider_pose, pusher_xy=None) -> ResetReport:
        """Teleport to a push-start and settle.

        pusher_xy = None uses the config's nominal start (deterministic resets in
        tests/acceptance); a sampled start comes from pusht_drake.sim.spawn.

        Raises RuntimeError if the state does not settle within the timeout or any
        stability invariant is violated while settling.
        """
        from pusht_drake.sim.planar_pose import PlanarPose

        if pusher_xy is None:
            start_xy = np.array([self.config.pusher_start_pose.x, self.config.pusher_start_pose.y])
            q_start = self.q_push_start
            pusher_pose = self.config.pusher_start_pose
        else:
            start_xy = np.asarray(pusher_xy, dtype=float).reshape(2)
            q_start = self.solve_start_configuration(start_xy)
            pusher_pose = PlanarPose(float(start_xy[0]), float(start_xy[1]), 0.0)
        self._start_xy = start_xy

        # Hold the operator's stick for the whole teleport+settle. The command
        # port is evaluated by DiffIK at 1 kHz while we advance the plant below,
        # so a still-deflected stick would integrate the target away from the
        # pose we just placed the pusher at, moving the episode's
        # start off the sampled point, and at full deflection tripping the
        # excursion guard within a third of a second. freed in the finally.
        self._hold_operator_input(True)
        try:
            return self._teleport_and_settle(q_start, slider_pose, pusher_pose)
        finally:
            self._hold_operator_input(False)
            # Resume from exactly where the reset put the pusher, so releasing the
            # hold cannot hand DiffIK a stale target to chase.
            self.station.source.reset(start_xy)
            self._reanchor_realtime_pacing()

    def _reanchor_realtime_pacing(self) -> None:
        """Re-anchor the wall-clock reference the realtime throttle paces against.

        Drake paces set_target_realtime_rate against a reference captured by
        Initialize()/ResetStatistics(), and the throttle is cumulative against
        it: once the simulation is behind, PauseIfTooFast stops sleeping
        entirely until the whole deficit is repaid. Measured on Drake 1.48 --
        one loaded second (wall 2.06 s) is followed by a second that runs in
        3 ms at zero throttling.

        Teleop is the only path that runs paced, for hours, and it called
        Initialize() exactly once per session: the initialization latch
        short-circuits forever after the first reset. Every reset then injects
        fresh deficit (a full nonlinear IK solve plus a 0.05-2.0 s settle, wall
        time against little sim time), so later episodes ran unthrottled
        at whatever the machine could deliver (~1.46x free space) while
        contact-heavy stretches managed ~0.91x. That is a ~60% swing in apparent
        speed at identical stick input, and it is invisible in the recorded
        data, which is stamped on an exact sim-time grid.

        Re-anchoring here, at the end of the reset, gives every episode its own
        reference and bounds the error within one episode instead of letting it
        compound across a session. It must run after the settle and after the IK
        solve, or the reset's own wall cost is baked into the new anchor.

        ResetStatistics() and not Initialize(): the narrow operation moves only
        the wall/sim reference, where Initialize() would re-run initialization
        events that _initialize_once exists to run exactly once.
        """
        simulator = self._simulator
        # Unpaced callers (evaluation runs at rate 0.0) have no reference to move.
        if simulator.get_target_realtime_rate() > 0.0:
            simulator.ResetStatistics()

    def _hold_operator_input(self, held: bool) -> None:
        hold = getattr(self.station.source, "hold_input", None)
        if hold is not None:  # policy/fixed sources integrate nothing to hold
            hold(held)

    def _teleport_and_settle(self, q_start, slider_pose, pusher_pose) -> ResetReport:
        # Upstream's own teleport: plant positions + zero velocities + DiffIK sync.
        self.station.teleport(q_start, slider_pose, pusher_pose)
        self._reset_clamp_state()
        forced = self._initialize_once()

        simulator = self._simulator
        station = self.station
        t0 = simulator.get_context().get_time()
        max_excursion = 0.0
        settle_time = None

        while True:
            simulator.AdvanceTo(simulator.get_context().get_time() + SETTLE_SLICE_S)
            elapsed = simulator.get_context().get_time() - t0

            if not station.state_finite():
                raise RuntimeError(f"non-finite plant state {elapsed:.2f}s into reset settling")
            quat_err = abs(station.slider_quaternion_norm() - 1.0)
            if quat_err > QUATERNION_TOL:
                raise RuntimeError(
                    f"slider quaternion drifted (|norm-1| = {quat_err:.2e}) during settling"
                )
            excursion = float(np.linalg.norm(station.pusher_xy() - self._start_xy))
            max_excursion = max(max_excursion, excursion)
            if excursion > PUSHER_EXCURSION_LIMIT:
                raise RuntimeError(
                    f"pusher swept {excursion * 100:.1f} cm from its start during settling "
                    "-- the post-teleport command sweep is back (see pusht_drake.sim.reset)"
                )

            slider_speed, robot_speed = station.speeds()
            if (
                slider_speed < SLIDER_SPEED_TOL
                and robot_speed < ROBOT_SPEED_TOL
                and excursion < PUSHER_POSITION_TOL
            ):
                settle_time = elapsed
                break
            if elapsed >= SETTLE_TIMEOUT_S:
                raise RuntimeError(
                    f"reset did not settle within {SETTLE_TIMEOUT_S}s: "
                    f"slider_speed={slider_speed:.2e}, robot_speed={robot_speed:.2e}, "
                    f"pusher_offset={excursion:.4f} m"
                )

        if settle_time > SETTLE_BUDGET_S:
            logger.info("reset settled in %.2fs (budget %.2fs)", settle_time, SETTLE_BUDGET_S)
        return ResetReport(
            settle_time_s=float(settle_time),
            max_pusher_excursion_m=max_excursion,
            forced_pushing=forced,
        )
