"""Drake position sources that are not the gamepad.

The simulator exposes one control seam, the planar_position_command port: a
2-vector of absolute pusher (x, y) in meters, world frame. The gamepad writes
to it during teleoperation; a policy writes to it during evaluation. They are
interchangeable there, and nothing else couples them.

FixedPositionSource holds one constant target, so the simulator can be
exercised end to end (built, stepped, checked for contact) with no policy, no
checkpoint and no gamepad attached. The arm-dynamics tests drive it directly.

ActionChunkPolicySource occupies the same seam during evaluation: a
sim.interface.Policy runs inside the diagram, driven by a periodic unrestricted
update event at the policy rate. The executing chunk spline lives in Drake abstract state and is
replaced only when the event scheduler says so, which pins the command pipeline
at exactly one policy period in every episode; swapping the plan from a Python
loop instead lets accumulated float drift decide which side of a latch a
command lands on. Between events the output port evaluates the spline at the
context time, so DiffIK's own 1 kHz evaluation samples a continuous trajectory
through the guarded waypoints instead of a 10 Hz staircase
(pusht_drake.sim.bspline_chunk, adapted from the official B-spline Policy
runtime).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from pydrake.all import (
    AbstractValue,
    Diagram,
    DiagramBuilder,
    LeafSystem,
    RigidTransform,
    ZeroOrderHold,
)

from pusht_drake.observation import build_observation
from pusht_drake.sim.bspline_chunk import ChunkSpline, build_chunk_spline
from pusht_drake.sim.guard import LEASH_M, GuardTicks, guard_command
from pusht_drake.sim.planar_pose import PlanarPose

# Execution strategies for the 10 Hz waypoints. The mode is the only
# experimental variable between them: everything before it (policy,
# observation, chunk, guard) and after it (DiffIK, driver, plant) is shared.
#   "staircase"    hold each guarded anchor for its period (the historical
#                  execution, kept for A/B/C comparison);
#   "bspline"      evaluate the C^1 chunk spline continuously (stage 1);
#   "cartesian_pd" canonical gym_pusht servo: a = KP*(anchor - measured)
#                  - KD*v_ref, v_ref += a*dt, command = measured + v_ref*dt.
EXECUTION_MODES = ("staircase", "bspline", "cartesian_pd")

# The canonical Push-T PD constants, verbatim from gym_pusht/envs/pusht.py
# (self.k_p, self.k_v = 100, 20): omega_n = 10 rad/s, zeta = 1 (critically
# damped). The linear servo is scale-free, so the pixel-space constants carry
# to SI meters unchanged. gym_pusht integrates at 100 Hz; we integrate at the
# native 1 kHz (identical dynamics to O(dt); Euler stability bound is 0.2 s).
#
# The servo simulates the pymunk agent in reference space: it owns
# (p_ref, v_ref), applies a = KP*(anchor - p_ref) - KD*v_ref exactly as the
# canonical env applies it to the physical body, and the arm servo-tracks
# p_ref through the existing DiffIK. v_ref is clamped to DiffIK's demand cap
# so the reference never asks what the arm cannot do; under contact p_ref
# converges to the anchor (a stable autonomous system, so no windup) and the
# pusher pushes with cap-bounded speed, the same blocked-target behavior as
# the other modes. A measured-feedback rebase (p_cmd = measured + v_ref*dt)
# was tried and rejected: DiffIK reads a fresher measured than the rebase
# did, so its demanded rate becomes v_ref minus the actual velocity, an
# extra effective pole that turned the critically damped pair into a
# sustained ~1 Hz, ~20 mm limit cycle around constant targets (measured).
CARTESIAN_PD_KP = 100.0  # 1/s^2
CARTESIAN_PD_KD = 20.0  # 1/s
_SERVO_DT = 0.001  # s
_SERVO_V_CAP = 0.25  # m/s, mirrors diffik.MAX_TRANSLATIONAL_SPEED


class _ConstantPlanarPosition(LeafSystem):
    def __init__(self, target_xy: np.ndarray) -> None:
        super().__init__()
        self._target = np.asarray(target_xy, dtype=float).reshape(2)
        self.DeclareVectorOutputPort("planar_position_command", 2, self._calc)

    def _calc(self, context, output) -> None:
        output.SetFromVector(self._target)

    def set_target(self, target_xy: np.ndarray) -> None:
        self._target = np.asarray(target_xy, dtype=float).reshape(2)


class FixedPositionSource(Diagram):
    """Commands a constant pusher position, held at freq like a real policy.

    The zero-order hold is not cosmetic: it is what makes the command a
    staircase at the policy rate rather than a continuously-evaluated signal,
    matching how DiffusionPolicySource drives the same port in the upstream
    Drake station (Michaelszeng/diffusion-policy-drake; see NOTICE.md).
    """

    # Read by the station's duck-typed source detection, so the station does
    # not have to import (and type-check against) this class.
    desired_position_source_type = "diffusion_policy"

    def __init__(self, target_xy: np.ndarray, freq: float = 10.0) -> None:
        super().__init__()
        builder = DiagramBuilder()
        self._source = builder.AddNamedSystem(
            "ConstantPlanarPosition", _ConstantPlanarPosition(target_xy)
        )
        zoh = builder.AddNamedSystem("ZeroOrderHold", ZeroOrderHold(1.0 / freq, 2))
        builder.Connect(self._source.get_output_port(), zoh.get_input_port())
        builder.ExportOutput(zoh.get_output_port(), "planar_position_command")
        builder.BuildInto(self)
        self._translation_scale = 1.0

    def set_target(self, target_xy: np.ndarray) -> None:
        self._source.set_target(target_xy)

    def reset(self, reset_xy: np.ndarray | None = None) -> None:
        if reset_xy is not None:
            self.set_target(reset_xy)

    # The environment and the teleop session treat every position source uniformly,
    # including the speed modifier the gamepad exposes. These make this source a
    # drop-in for that interface; they have no effect on a constant target.
    def get_translation_scale(self) -> float:
        return self._translation_scale

    def set_translation_scale(self, translation_scale: float) -> None:
        self._translation_scale = float(translation_scale)


@dataclass(frozen=True)
class CommandRecord:
    """One periodic-event command: the anchor scheduled at this event.

    executed is the guarded anchor the chunk spline interpolates, the absolute
    target attained exactly one policy period after this event (the same
    attainment timing the staircase mode has). The continuous command between
    anchors is the spline, not recorded here.
    """

    t: float
    requested: np.ndarray  # (2,) pre-guard, straight off the chunk
    executed: np.ndarray  # (2,) post square -> fence -> leash (the spline anchor)
    replanned: bool  # this event called predict_chunk
    ticks: GuardTicks


class ActionChunkController(LeafSystem):
    """Runs a sim.interface.Policy at the policy rate, inside the diagram.

    The periodic unrestricted update (period 1/freq, offset 0) builds the
    5-D observation and, when the chunk is exhausted, calls predict_chunk,
    guards every waypoint (chained: the first against the measured pusher, each
    next against its guarded predecessor, so the leash stays at 10 Hz waypoint
    granularity), and stores the resulting ChunkSpline in abstract state. The
    output port evaluates that spline at the context time and passes the point
    through the containment layers of the same guard chain (square + fence, no
    leash), so the commanded signal is a continuous C^1 trajectory that attains
    each guarded anchor exactly one period after its event: the staircase's
    attainment timing without its stop-go execution. Drake's event scheduler
    remains the sole authority on when the plan changes; there is no
    ZeroOrderHold.

    Reset and seeding use a mailbox consumed at the start of the next event
    (upstream's lazy _received_reset_signal pattern): reset/begin_episode only
    write Python attributes while the simulator is paused, so every context
    mutation happens inside event dispatch. Note the Drake boundary rule: an
    event scheduled exactly at an AdvanceTo target fires at the start of the
    next call. The rollout's grid stepping and the resetter's settle both lean
    on this (one command event per grid step; the mailbox is consumed at the
    first settle slice while hold_input suppresses prediction).

    Telemetry (commands, tracking errors) is write-only: drained by the
    rollout at episode end, never fed back into the output.
    """

    def __init__(
        self,
        initial_xy: np.ndarray,
        *,
        square,
        workspace,
        policy=None,
        freq: float = 10.0,
        leash_m: float = LEASH_M,
        execution_horizon: int | None = None,
        execution_mode: str = "bspline",
    ) -> None:
        super().__init__()
        # execution_horizon is the commitment: how many of the predicted
        # waypoints actually execute before the policy is re-queried. It is
        # independent of the lookahead, which is always the whole predicted
        # chunk and only shapes the spline. None == open loop: commit
        # everything. A policy that declares its own execution_horizon owns
        # this value; the constructor argument exists for scripted test
        # policies.
        if execution_horizon is not None and execution_horizon < 1:
            raise ValueError(f"execution_horizon must be >= 1, got {execution_horizon}")
        if execution_mode not in EXECUTION_MODES:
            raise ValueError(
                f"execution_mode must be one of {EXECUTION_MODES}, got {execution_mode!r}"
            )
        self._square = square
        self._workspace = workspace
        self._freq = float(freq)
        self._leash_m = float(leash_m)
        self._execution_horizon = execution_horizon
        self._policy_execution_horizon: int | None = None
        self._execution_mode = execution_mode
        self._policy = policy
        self._seeded = False
        self._held = False
        self._pending_reset_xy: np.ndarray | None = None
        self._pending_seed: int | None = None
        # (requested, executed_anchor, replanned, ticks) per pending event
        self._queue: deque[tuple[np.ndarray, np.ndarray, bool, GuardTicks]] = deque()
        self._commands: list[CommandRecord] = []
        self._tracking_errors: list[float] = []
        self._issued_any = False  # a policy command has completed >= 0 periods this episode
        # Containment-clamp diagnostics: counts context times at which the
        # evaluated spline point had to be clamped back into square-and-fence.
        # Deduped by time because a cached output calc can run more than once
        # per instant (initialization, publishes). Expected 0 in-distribution.
        self._spline_clip_ticks = 0
        self._last_clip_time: float | None = None

        self._pusher_pose_input = self.DeclareAbstractInputPort(
            "pusher_pose_measured", AbstractValue.Make(RigidTransform())
        )
        self._slider_pose_input = self.DeclareAbstractInputPort(
            "slider_pose_measured", AbstractValue.Make(RigidTransform())
        )
        initial = np.asarray(initial_xy, dtype=float).reshape(2)
        self._command_state = self.DeclareAbstractState(
            AbstractValue.Make(ChunkSpline.constant(initial, t_origin=0.0, data_freq=self._freq))
        )
        # The active 10 Hz anchor (the target attained one period after its
        # event). All modes keep it: staircase outputs it, cartesian_pd
        # attracts toward it, and the tracking sample measures against it
        # (for bspline that equals spline.eval at event times bit-exactly).
        self._anchor_state = self.DeclareAbstractState(AbstractValue.Make(initial.copy()))
        self.DeclarePeriodicUnrestrictedUpdateEvent(1.0 / freq, 0.0, self._update)
        if execution_mode == "cartesian_pd":
            # (v_ref_x, v_ref_y, p_ref_x, p_ref_y): the virtual pymunk agent.
            self._servo_state = self.DeclareDiscreteState(np.concatenate([np.zeros(2), initial]))
            self.DeclarePeriodicDiscreteUpdateEvent(_SERVO_DT, 0.0, self._servo_update)
        # Output dependencies, mode-exact, so caching recomputes exactly when
        # the command can change (and the port is not conservatively marked
        # feedthrough from the pose inputs).
        if execution_mode == "bspline":
            prerequisites = {self.abstract_state_ticket(self._command_state), self.time_ticket()}
        elif execution_mode == "staircase":
            prerequisites = {self.abstract_state_ticket(self._anchor_state)}
        else:
            prerequisites = {self.xd_ticket()}
        self.DeclareVectorOutputPort(
            "planar_position_command",
            2,
            self._calc_command,
            prerequisites_of_calc=prerequisites,
        )

    # -- Drake callbacks ----------------------------------------------------

    def _calc_command(self, context, output) -> None:
        if self._execution_mode == "staircase":
            # Historical semantics, exactly: the guarded anchor, held raw for
            # its period (no containment clamp; anchors are already guarded).
            output.SetFromVector(context.get_abstract_state(int(self._anchor_state)).get_value())
            return
        if self._execution_mode == "cartesian_pd":
            xy = context.get_discrete_state(int(self._servo_state)).get_value()[2:4]
        else:  # bspline
            spline = context.get_abstract_state(int(self._command_state)).get_value()
            xy = spline.eval(context.get_time())
        # Final containment: the same guard chain minus the leash. The anchors
        # are guarded; a cubic between them (or a servo step from the measured
        # pose) can leave the square only by millimeters, and this clamp turns
        # that into a certified guarantee. Expected to be the identity always.
        safe, ticks = guard_command(
            xy, square=self._square, workspace=self._workspace, leash_m=None
        )
        if ticks.square or ticks.fence:
            t = context.get_time()
            if t != self._last_clip_time:
                self._spline_clip_ticks += 1
                self._last_clip_time = t
        output.SetFromVector(safe)

    def _servo_update(self, context, discrete_state) -> None:
        """The canonical gym_pusht agent, simulated in reference space.

        a = KP*(anchor - p_ref) - KD*v_ref; v_ref += a*dt; p_ref += v_ref*dt,
        the exact update the canonical env applies to its physical body, with
        v_ref clamped to the DiffIK demand cap so the reference stays
        reachable. The arm tracks p_ref (sub-mm, unsaturated); the reference is
        autonomous and stable, so a blocked pusher cannot wind it up beyond the
        anchor itself.
        """
        servo = discrete_state.get_mutable_vector(int(self._servo_state))
        if self._held or not self._seeded or self._policy is None:
            # Freeze at the parked reference (the mailbox parks p_ref on the
            # episode start at the first settle event, exactly like the other
            # modes' parked anchor). A measured-tracking hold was tried and
            # rejected: it turns the command into a follower, and the IDC's
            # episode-accumulated integral state then walks the free-floating
            # arm ~0.2 m/s during settling (measured; tripped the 3 cm
            # excursion guard on every second-and-later episode per rig).
            return
        anchor = context.get_abstract_state(int(self._anchor_state)).get_value()
        state = servo.get_value()
        v_ref = np.array(state[0:2])
        p_ref = np.array(state[2:4])
        a_ref = CARTESIAN_PD_KP * (anchor - p_ref) - CARTESIAN_PD_KD * v_ref
        v_ref = v_ref + a_ref * _SERVO_DT
        speed = float(np.linalg.norm(v_ref))
        if speed > _SERVO_V_CAP:
            v_ref = v_ref * (_SERVO_V_CAP / speed)
        p_ref = p_ref + v_ref * _SERVO_DT
        servo.set_value(np.concatenate([v_ref, p_ref]))

    def _update(self, context, state) -> None:
        command_state = state.get_mutable_abstract_state(int(self._command_state))
        anchor_state = state.get_mutable_abstract_state(int(self._anchor_state))

        # 1. Mailbox first, even while held: the resetter parks the command on
        # the episode's start pose during settling, as a degenerate constant
        # spline, so the finished episode's plan stops playing.
        if self._pending_reset_xy is not None:
            parked, _ = guard_command(
                self._pending_reset_xy, square=self._square, workspace=self._workspace, leash_m=None
            )
            command_state.set_value(
                ChunkSpline.constant(
                    parked, t_origin=float(context.get_time()), data_freq=self._freq
                )
            )
            anchor_state.set_value(parked.copy())
            if self._execution_mode == "cartesian_pd":
                state.get_mutable_discrete_state(int(self._servo_state)).set_value(
                    np.concatenate([np.zeros(2), parked])
                )
            self._queue.clear()
            self._issued_any = False
            self._pending_reset_xy = None
            # The finished episode's plan has stopped playing, so a warm-start
            # policy's cached forecast now refers to a trajectory that is gone.
            invalidate = getattr(self._policy, "invalidate_warm_cache", None)
            if invalidate is not None:
                invalidate()
        if self._pending_seed is not None and self._policy is not None:
            self._policy.reset(self._pending_seed)
            self._seeded = True
            self._pending_seed = None

        # 2. Hold: output the parked/last command, predict nothing. Unseeded
        # (fresh rig, no begin_episode yet) behaves like FixedPositionSource.
        if self._held or not self._seeded or self._policy is None:
            return

        # 3. Observe at the event time. The tracking sample measures the
        # measured pusher against the anchor that was due now, uniformly across
        # execution modes (for bspline it equals spline.eval at event times
        # bit-exactly; for cartesian_pd it is the attractor error, the
        # canonical servo's own lag, reported with that meaning). The episode's
        # first event has no completed policy command to measure.
        pusher_xy = np.array(self._pusher_pose_input.Eval(context).translation()[:2])
        slider_planar = PlanarPose.from_pose(self._slider_pose_input.Eval(context)).vector()
        if self._issued_any:
            current = anchor_state.get_value()
            self._tracking_errors.append(float(np.linalg.norm(pusher_xy - current)))

        # 4. Refill on exhaustion: predict, guard the whole chunk (chained:
        # first anchor against the measured pusher, each next against its
        # guarded predecessor, so the leash stays at 10 Hz waypoint
        # granularity), and replace the executing spline.
        #
        # Lookahead vs commitment. Every predicted waypoint shapes the spline;
        # only the first commit of them are enqueued and therefore ever
        # executed. The queue drains in commit events, so this block is also
        # the replan clock. Under open loop the two coincide.
        #
        # The new plan starts at the outgoing spline's position and velocity at
        # the splice node, the node just reached, phase commit of the outgoing
        # plan, so the commanded signal is C^1 across a mid-plan cut and not
        # only across an exhausted one.
        if not self._queue:
            observation = build_observation(pusher_xy, slider_planar)
            chunk = np.asarray(self._policy.predict_chunk(observation), dtype=np.float64)
            if chunk.ndim != 2 or chunk.shape[1] != 2 or len(chunk) == 0:
                raise ValueError(f"policy returned chunk of shape {chunk.shape}; expected (H, 2)")
            lookahead = len(chunk)
            commit = self.commit_horizon
            commit = lookahead if commit is None else commit
            if commit > lookahead:
                # Truncating here would let a checkpoint trained to be
                # re-queried every commit steps run at a slower cadence,
                # de-synchronizing its own forecast cache. Raise instead.
                raise ValueError(
                    f"execution_horizon {commit} exceeds the predicted chunk length "
                    f"{lookahead}; the policy cannot commit more waypoints than it planned."
                )
            anchors: list[np.ndarray] = []
            per_anchor: list[tuple[np.ndarray, np.ndarray, bool, GuardTicks]] = []
            chain_anchor = pusher_xy
            for index, row in enumerate(chunk):
                requested = np.array(row, dtype=float)
                executed, ticks = guard_command(
                    requested,
                    chain_anchor,
                    square=self._square,
                    workspace=self._workspace,
                    leash_m=self._leash_m,
                )
                anchors.append(executed)
                per_anchor.append((requested, executed, index == 0, ticks))
                chain_anchor = executed
            if self._execution_mode == "bspline":
                now = float(context.get_time())
                outgoing = command_state.get_value()
                if commit == lookahead:
                    # Open loop: the outgoing plan was queued whole and drained
                    # whole, so it is exhausted and its end is the splice point.
                    # Take it from the node rather than from the clock: a parked
                    # hold is only constant to within an ULP (measured: 30% of
                    # holds differ by 2.8e-17 m between phase 0 and phase
                    # n_spans, because de Boor's convex combinations do not
                    # reproduce equal control points exactly), and one ULP in
                    # this boundary condition changes episode length and
                    # termination through contact.
                    p_start, v_start = outgoing.end_state()
                else:
                    p_start, v_start = outgoing.splice_state(now)
                command_state.set_value(
                    build_chunk_spline(
                        p_start,
                        v_start,
                        np.array(anchors),
                        t_origin=now,
                        data_freq=self._freq,
                        n_commit=commit,
                    )
                )
            self._queue.extend(per_anchor[:commit])
            # The consumed count is pushed to the policy, never inferred by it:
            # a warm-start policy re-anchors its cached forecast by exactly
            # this many steps, and no geometric check downstream can detect an
            # off-by-one in it (waypoint spacing is smaller than the arm's own
            # tracking error 16% of the time).
            notify = getattr(self._policy, "notify_committed", None)
            if notify is not None:
                notify(commit)

        # 5. One telemetry record per event, stamped at pop time from the
        # context (precomputing t_refill + j/freq would not bit-match the
        # grid). The state was written at the replan event; this pop only
        # attributes the anchor scheduled now.
        requested, executed, replanned, ticks = self._queue.popleft()
        anchor_state.set_value(np.array(executed, dtype=float))
        self._issued_any = True
        self._commands.append(
            CommandRecord(
                t=float(context.get_time()),
                requested=requested,
                executed=executed,
                replanned=replanned,
                ticks=ticks,
            )
        )

    # -- episode control (simulator paused; consumed by the next event) -----

    @property
    def commit_horizon(self) -> int | None:
        """Waypoints committed per replan; None means open loop (commit all).

        A policy that declares execution_horizon is authoritative, because the
        value is a property of the trained artifact: a warm-start checkpoint
        anchors its cached forecast on being re-queried at exactly this cadence.
        The constructor argument is for scripted policies that declare nothing.
        """
        if self._policy_execution_horizon is not None:
            return self._policy_execution_horizon
        return self._execution_horizon

    def set_policy(self, policy) -> None:
        """Install the policy for subsequent episodes; it must be re-seeded."""
        if policy is None:
            raise ValueError("set_policy requires a policy; construct with policy=None to defer")
        declared = getattr(policy, "execution_horizon", None)
        if declared is not None:
            declared = int(declared)
            if declared < 1:
                raise ValueError(f"policy declares execution_horizon={declared}; must be >= 1")
            if self._execution_horizon is not None and declared != self._execution_horizon:
                raise ValueError(
                    f"policy declares execution_horizon={declared} but this source was "
                    f"built with {self._execution_horizon}. A disagreement here mis-times "
                    "the policy's own forecast cache; refusing to guess."
                )
        self._policy_execution_horizon = declared
        self._policy = policy
        self._seeded = False
        self._queue.clear()

    def begin_episode(self, seed: int) -> None:
        """Arm one episode: seed via the mailbox, drop stale queue + telemetry."""
        if self._policy is None:
            raise RuntimeError("set_policy must be called before begin_episode")
        self._pending_seed = int(seed)
        self._queue.clear()
        self._commands.clear()
        self._tracking_errors.clear()
        self._issued_any = False
        self._spline_clip_ticks = 0
        self._last_clip_time = None

    def reset(self, reset_xy=None) -> None:
        if reset_xy is not None:
            self._pending_reset_xy = np.asarray(reset_xy, dtype=float).reshape(2)

    def hold_input(self, held: bool) -> None:
        """Freeze the command and skip prediction (owned by the resetter)."""
        self._held = bool(held)

    def drain_telemetry(self) -> tuple[list[CommandRecord], list[float]]:
        commands, tracking = self._commands, self._tracking_errors
        self._commands, self._tracking_errors = [], []
        return commands, tracking

    def take_spline_clip_ticks(self) -> int:
        """Distinct context times at which the containment clamp moved the
        evaluated spline point. Expected 0 in-distribution; nonzero means the
        spline arced outside square-and-fence between guarded anchors."""
        ticks = self._spline_clip_ticks
        self._spline_clip_ticks = 0
        self._last_clip_time = None
        return ticks


class ActionChunkPolicySource(Diagram):
    """Diagram wrapper wiring ActionChunkController into the station.

    Declares desired_position_source_type = "diffusion_policy". The station
    reads that attribute to decide which measured-pose converters to build
    (pusher and slider) and connects them to the two exported input ports.
    """

    desired_position_source_type = "diffusion_policy"

    def __init__(
        self,
        initial_xy: np.ndarray,
        *,
        square,
        workspace,
        policy=None,
        freq: float = 10.0,
        leash_m: float = LEASH_M,
        execution_horizon: int | None = None,
        execution_mode: str = "bspline",
    ) -> None:
        super().__init__()
        builder = DiagramBuilder()
        self._controller = builder.AddNamedSystem(
            "ActionChunkController",
            ActionChunkController(
                initial_xy,
                square=square,
                workspace=workspace,
                policy=policy,
                freq=freq,
                leash_m=leash_m,
                execution_horizon=execution_horizon,
                execution_mode=execution_mode,
            ),
        )
        builder.ExportInput(
            self._controller.GetInputPort("pusher_pose_measured"), "pusher_pose_measured"
        )
        builder.ExportInput(
            self._controller.GetInputPort("slider_pose_measured"), "slider_pose_measured"
        )
        builder.ExportOutput(
            self._controller.GetOutputPort("planar_position_command"), "planar_position_command"
        )
        builder.BuildInto(self)

    @property
    def controller(self) -> ActionChunkController:
        return self._controller

    @property
    def commit_horizon(self) -> int | None:
        return self._controller.commit_horizon

    def set_policy(self, policy) -> None:
        self._controller.set_policy(policy)

    def begin_episode(self, seed: int) -> None:
        self._controller.begin_episode(seed)

    def reset(self, reset_xy=None) -> None:
        self._controller.reset(reset_xy)

    def hold_input(self, held: bool) -> None:
        self._controller.hold_input(held)

    def drain_telemetry(self) -> tuple[list[CommandRecord], list[float]]:
        return self._controller.drain_telemetry()

    def take_spline_clip_ticks(self) -> int:
        return self._controller.take_spline_clip_ticks()
