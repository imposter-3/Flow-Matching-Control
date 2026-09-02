"""Yaw-relaxed Differential IK for an axisymmetric pusher.

Why this exists instead of Drake's DifferentialInverseKinematicsIntegrator
(measured 2026-08-22):

- The pusher is a cylinder welded along link 7's axis, so rotation about its
  own axis (yaw) is physically meaningless. The upstream Drake station
  (Michaelszeng/diffusion-policy-drake; see NOTICE.md) nevertheless tracks the
  full 6-DOF pose with yaw fixed at 0, which presses J7, parked 3.8 deg from
  its +175 deg limit, into saturation after 13-38 cm of planar travel in most
  directions. That is the stuck/no solution chatter that blocked teleop.
- Drake 1.48's sanctioned relaxation, set_end_effector_velocity_flag, was
  tried and rejected on measurement: the integrator computes its error as the
  axis-angle of the full rotation difference and then masks Jacobian rows, so
  once the freed yaw drifts toward pi the yaw error pollutes the tilt
  components (measured: 5.9 deg accumulated tilt, ~73 rad/s demanded
  correction, alpha collapse at 12-14 cm, inside the true reachable set).
- The stock integrator also exposes no status: the only failure signal is a log
  line, so recovery and tests cannot see it.

This system owns the error computation instead:

    tilt error   e_t = z_pusher x z_down          (yaw-invariant by construction)
    V5 = [ k_tilt * e_t_xy ; (p_desired - p)/dt ]  in world frame
    J5 = spatial Jacobian rows [wx_W, wy_W, vx, vy, vz]
    v  = DoDifferentialInverseKinematics(q, v_prev, V5, J5, params)   (Drake QP)
    q += dt * v

With this formulation failures occur only at the true physical boundaries (the
r ~ 0.80 m arm-extension circle and the J4 elbow limit toward the base),
verified along 12 rays against the same boundaries found by nonlinear IK
mapping.

Ports and the SetPositions method mirror the Drake integrator, so the
station's wiring and the reset's DiffIK resync work against either.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from pydrake.all import (
    AbstractValue,
    DifferentialInverseKinematicsParameters,
    DifferentialInverseKinematicsStatus,
    DoDifferentialInverseKinematics,
    JacobianWrtVariable,
    LeafSystem,
    RigidTransform,
)

logger = logging.getLogger(__name__)

Z_DOWN = np.array([0.0, 0.0, -1.0])  # desired pusher z-axis in world frame

# Cap on the demanded Cartesian speed fed to the QP. Without it the demand is
# position_error/dt (250 m/s for a 25 cm step at 1 kHz), which makes the QP's
# achieved-fraction alpha meaninglessly small and reports kStuck on every large
# step even while the arm is moving at full speed. With the cap, alpha ~ 1 in
# free space and kStuck fires only when genuinely constrained (a wall). The cap
# is 4x the full-stick teleop speed and far below joint-limit capability.
MAX_TRANSLATIONAL_SPEED = 0.25  # m/s
MAX_TILT_RATE = 2.0  # rad/s


@dataclass
class DiffIkDiagnostics:
    """Separate failure counters plus the last status-transition snapshot."""

    stuck_ticks: int = 0
    no_solution_ticks: int = 0
    transitions: int = 0
    last_status: str = "kSolutionFound"
    last_transition: dict = field(default_factory=dict)


class YawRelaxedDiffIk(LeafSystem):
    """1 kHz differential IK holding the pusher vertical, yaw free.

    Parameters are the station's exact settings (velocity limits, position
    limits, identity joint-centering) except the nominal posture, whose J7
    entry is mid-range (0) so centering never drags the wrist toward the limit
    it sits against under upstream's fixed-yaw scheme.
    """

    def __init__(
        self,
        robot,
        frame_name: str,
        time_step: float,
        params,
        secondary=None,
        centering_gain: float = 1.0,
        nominal_schedule=None,
    ) -> None:
        super().__init__()
        # Secondary (null-space) objective. Drake's QP minimises
        #     |P (v_next - N^+ K (q_nominal - q))|^2
        # subject to the primary task equality, so whatever desired joint velocity
        # we express through the nominal-posture channel is projected by P onto the
        # null space of the primary task. Injecting qdot_0 as
        #     q_nominal_effective = q + qdot_0 / k
        # therefore adds a secondary behaviour that is structurally subordinate to
        # the 5D Push-T task: it can never trade tracking away, which a
        # weighted-least-squares formulation would not guarantee.
        self._secondary = secondary
        self._centering_gain = float(centering_gain)
        self._nominal = np.asarray(params.get_nominal_joint_position(), dtype=float).copy()
        # Optional radius-scheduled nominal posture: (radii, postures,
        # base_pose), base_pose = (x, y, yaw). The QP's joint-centering term
        # (K = I, always on) pulls the null space toward nominal. With a
        # constant nominal, the push-start posture, that pull points the wrong
        # way over most of a 0.21-0.73 m band and, near the fully-extended wall,
        # drags the wrist across IK branches until the pusher folds into
        # link 5/6 (the acceptance run measured -12 mm). The schedule
        # interpolates one continuous certified collision-free branch, keyed by
        # the target's distance from the base axis, so the centering pull points
        # toward a posture certified for where the pusher is going. This is the
        # baseline centering channel, not a Level-4 secondary objective; those
        # remain disabled.
        self._schedule = None
        if nominal_schedule is not None:
            radii, postures, base_pose = nominal_schedule
            self._schedule = (
                np.asarray(radii, dtype=float),
                np.asarray(postures, dtype=float),
                np.asarray(base_pose, dtype=float).reshape(3),
            )
        self._robot = robot
        self._frame = robot.GetFrameByName(frame_name)
        self._dt = float(time_step)
        self._params: DifferentialInverseKinematicsParameters = params
        self._n = robot.num_positions()
        self._robot_ctx = robot.CreateDefaultContext()

        self.diagnostics = DiffIkDiagnostics()

        # Mirror the Drake integrator's ports so the station wiring is unchanged.
        self._desired_port = self.DeclareAbstractInputPort(
            "X_WE_desired", AbstractValue.Make(RigidTransform())
        )
        self._robot_state_port = self.DeclareVectorInputPort("robot_state", 2 * self._n)
        self._use_robot_state_port = self.DeclareAbstractInputPort(
            "use_robot_state", AbstractValue.Make(False)
        )

        self._q_index = self.DeclareDiscreteState(self._n)
        self._v_index = self.DeclareDiscreteState(self._n)
        self.DeclarePeriodicDiscreteUpdateEvent(self._dt, 0.0, self._update)
        self.DeclareInitializationDiscreteUpdateEvent(self._initialize)
        self.DeclareStateOutputPort("joint_positions", self._q_index)

    # -- Drake-integrator-compatible API ------------------------------------

    def SetPositions(self, context, q) -> None:
        context.get_mutable_discrete_state(self._q_index).set_value(np.asarray(q).ravel())
        context.get_mutable_discrete_state(self._v_index).set_value(np.zeros(self._n))

    def ForwardKinematics(self, context):
        q = context.get_discrete_state(self._q_index).get_value()
        self._robot.SetPositions(self._robot_ctx, q)
        return self._robot.CalcRelativeTransform(
            self._robot_ctx, self._robot.world_frame(), self._frame
        )

    # -- dynamics ------------------------------------------------------------

    def _initialize(self, context, discrete_state) -> None:
        # Like the Drake integrator: start from the measured robot state.
        x = self._robot_state_port.Eval(context)
        discrete_state.get_mutable_vector(self._q_index).set_value(x[: self._n])
        discrete_state.get_mutable_vector(self._v_index).set_value(np.zeros(self._n))

    def _update(self, context, discrete_state) -> None:
        if self._use_robot_state_port.HasValue(context) and self._use_robot_state_port.Eval(
            context
        ):
            q = np.array(self._robot_state_port.Eval(context)[: self._n])
            v_prev = np.zeros(self._n)
        else:
            q = np.array(context.get_discrete_state(self._q_index).get_value())
            v_prev = np.array(context.get_discrete_state(self._v_index).get_value())

        target = self._desired_port.Eval(context).translation()

        # -- radius-scheduled nominal (see __init__) ----------------------------
        if self._schedule is not None:
            radii, postures, base_pose = self._schedule
            dx = target[0] - base_pose[0]
            dy = target[1] - base_pose[1]
            r = float(np.hypot(dx, dy))
            nominal = np.array(
                [np.interp(r, radii, postures[:, j]) for j in range(postures.shape[1])]
            )
            # Radius-only, with no azimuth term. An azimuth-corrected nominal
            # (J1 += beta, the exact same-branch rotation) was tried and
            # reverted: it re-aims the null-space pull mid-traverse, and on the
            # square's near wall, where the fold depth is task-locked and the
            # arm-arm clearance moves ~300 mm/rad of posture, it measurably
            # deepened the executed fold (acceptance p2 went from +0.4 mm to
            # -8 mm against the certification monitor). The settle problem that
            # correction was meant to fix lives in the reset path and is fixed
            # there instead: pusht_drake.sim.reset seeds each start at the
            # constrained equilibrium of this radius-only pull, so the arm has
            # nothing to slide toward.
            self._nominal = nominal
            self._params.set_nominal_joint_position(nominal)

        # -- secondary objective, injected through the nominal-posture channel --
        if self._secondary is not None and self._secondary.active:
            lower = self._robot.GetPositionLowerLimits()
            upper = self._robot.GetPositionUpperLimits()
            qdot0 = self._secondary.qdot0(
                self._robot, self._robot_ctx, self._frame, q, lower, upper
            )
            # Posture centring is retained unchanged as a weak regulariser; the
            # secondary objective is added to it, so the only difference between
            # ablation variants is the added term.
            nominal_eff = self._nominal + qdot0 / self._centering_gain
            self._params.set_nominal_joint_position(np.clip(nominal_eff, lower, upper))

        self._robot.SetPositions(self._robot_ctx, q)
        X = self._robot.CalcRelativeTransform(
            self._robot_ctx, self._robot.world_frame(), self._frame
        )
        z_now = X.rotation().matrix()[:, 2]
        tilt_rate = np.cross(z_now, Z_DOWN) / self._dt
        tilt_norm = float(np.linalg.norm(tilt_rate))
        if tilt_norm > MAX_TILT_RATE:
            tilt_rate *= MAX_TILT_RATE / tilt_norm
        translational_rate = (target - X.translation()) / self._dt
        speed = float(np.linalg.norm(translational_rate))
        if speed > MAX_TRANSLATIONAL_SPEED:
            translational_rate *= MAX_TRANSLATIONAL_SPEED / speed
        V5 = np.hstack([tilt_rate[:2], translational_rate])
        J = self._robot.CalcJacobianSpatialVelocity(
            self._robot_ctx,
            JacobianWrtVariable.kV,
            self._frame,
            np.zeros(3),
            self._robot.world_frame(),
            self._robot.world_frame(),
        )
        J5 = np.vstack([J[0:2, :], J[3:6, :]])

        result = DoDifferentialInverseKinematics(q, v_prev, V5, J5, self._params)
        self._record(context, result, target, X, q, V5)

        if result.status == DifferentialInverseKinematicsStatus.kNoSolutionFound:
            # No velocities are returned in this state; hold. The workspace
            # fence and leash ahead of this system make it unreachable in
            # normal operation.
            v = np.zeros(self._n)
        else:
            # kSolutionFound, and also kStuck: like Drake's integrator, stuck is
            # a report (achieved fraction alpha below threshold), not a freeze.
            # The QP's best-effort velocities still make what progress the
            # constraints allow (e.g. sliding along a boundary).
            v = np.asarray(result.joint_velocities)

        discrete_state.get_mutable_vector(self._q_index).set_value(q + self._dt * v)
        discrete_state.get_mutable_vector(self._v_index).set_value(v)

    # -- instrumentation ------------------------------------------------------

    def _record(self, context, result, target, X, q, V5) -> None:
        status = result.status
        name = str(status).split(".")[-1]
        diag = self.diagnostics
        if status == DifferentialInverseKinematicsStatus.kStuck:
            diag.stuck_ticks += 1
        elif status == DifferentialInverseKinematicsStatus.kNoSolutionFound:
            diag.no_solution_ticks += 1

        if name != diag.last_status:
            diag.transitions += 1
            lower = self._robot.GetPositionLowerLimits()
            upper = self._robot.GetPositionUpperLimits()
            J_xy = self._robot.CalcJacobianTranslationalVelocity(
                self._robot_ctx,
                JacobianWrtVariable.kQDot,
                self._frame,
                np.zeros(3),
                self._robot.world_frame(),
                self._robot.world_frame(),
            )[:2, :]
            sv = np.linalg.svd(J_xy, compute_uv=False)
            snapshot = {
                "time_s": float(context.get_time()),
                "status": name,
                "desired_xy": np.round(target[:2], 4).tolist(),
                "actual_xy": np.round(X.translation()[:2], 4).tolist(),
                "cartesian_error_m": float(np.linalg.norm(target - X.translation())),
                "q": np.round(q, 4).tolist(),
                "joint_limit_margins": np.round(np.minimum(q - lower, upper - q), 4).tolist(),
                "sigma_min_jxy": float(sv[-1]),
                "cond_jxy": float(sv[0] / sv[-1]),
                "requested_V5": np.round(V5, 3).tolist(),
            }
            diag.last_transition = snapshot
            level = logging.INFO if name == "kSolutionFound" else logging.WARNING
            logger.log(level, "DiffIK status -> %s: %s", name, snapshot)
        diag.last_status = name


def arm_velocity_limits(robot) -> np.ndarray:
    """The arm's joint velocity limits, read from the loaded plant.

    Raises on a model that declares none, rather than substituting another
    arm's numbers.
    """
    upper = np.asarray(robot.GetVelocityUpperLimits(), dtype=float)
    lower = np.asarray(robot.GetVelocityLowerLimits(), dtype=float)
    if not np.all(np.isfinite(upper)) or not np.all(np.isfinite(lower)):
        raise RuntimeError(
            "the loaded arm model declares no finite joint velocity limits; "
            "refusing to guess another arm's numbers"
        )
    if not np.allclose(upper, -lower):
        raise RuntimeError(f"asymmetric velocity limits {lower} .. {upper}; unexpected for an iiwa")
    return upper


def make_yaw_relaxed_params(robot, default_joint_positions, velocity_limit_factor: float = 1.0):
    """The station's exact DiffIK parameters, with the nominal wrist centered.

    Identical to upstream's setup (velocity limits, position limits, identity
    centering gain) except nominal J7 = 0: under yaw-free tracking J7 is the
    free self-motion DOF, and centering it mid-range is what keeps the wrist
    away from the limit upstream parks it against.
    """
    n = robot.num_positions()
    params = DifferentialInverseKinematicsParameters(n, n)
    params.set_time_step(1e-3)
    # Velocity limits come from the loaded model, not a transcribed table. A
    # hardcoded iiwa7 array ([1.7, 1.7, 1.7, 2.2, 2.4, 3.1, 3.1]) would
    # over-authorise an iiwa14 by up to 68% on J4 (1.309 rad/s actual), and
    # neither the QP nor the clamp would report it. The upstream iiwa7 numbers
    # were the SDF's own limits rounded down, so reading the plant reproduces
    # them there too.
    velocity_limits = arm_velocity_limits(robot)
    params.set_joint_velocity_limits(
        (
            -velocity_limit_factor * velocity_limits,
            velocity_limit_factor * velocity_limits,
        )
    )
    params.set_joint_position_limits(
        (robot.GetPositionLowerLimits(), robot.GetPositionUpperLimits())
    )
    nominal = np.asarray(default_joint_positions, dtype=float).copy()
    nominal[6] = 0.0
    params.set_nominal_joint_position(nominal)
    params.set_joint_centering_gain(np.eye(n))
    return params
