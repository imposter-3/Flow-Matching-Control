"""The LeafSystems the diagram needs, outside DiffIK and the policy source.

Four survive from the eleven the upstream station and environment carried:

    PlanarCommandToPose         lift the commanded (x, y) to a tip pose
    JointVelocityClamp          rate-limit the commanded joint positions
    RobotStateToPusherPose      arm state -> pusher pose, for the policy source
    SliderStateToPose           T state -> T pose, for the policy source

The seven that are gone were never load-bearing here:

    RobotStateToPlanarVelocity          built but never connected: no source in
                                        this repo declares the
                                        ``pusher_velocity_measured`` port it fed
    GeneralizedCoordsToPlanarPose       fed only a 100 Hz VectorLogSink that
    RigidTransformToPlanarPoseVector    nothing in evaluation, teleop or replay
                                        ever read
    IiwaPlanner                         a two-mode state machine whose HOLD mode
                                        the reset skips before the first
                                        AdvanceTo; its IK solve is now
                                        pusht_drake.sim.ik
    PortSwitch                          selected between the planner's hold
                                        output and DiffIK; with one reachable
                                        mode it is a wire
    RunFlagSystem                       23 lines comparing a port index to a
                                        constant, for that same one-mode machine
    CombineSpatialForces                the disturbance path, never built

``JointVelocityClamp`` is reproduced with upstream's evaluation timing
deliberately unchanged; see its docstring.
"""

from __future__ import annotations

import logging

from pydrake.all import AbstractValue, LeafSystem, MultibodyPlant, RigidTransform

from pusht_drake.sim.planar_pose import PlanarPose

logger = logging.getLogger(__name__)

#: First-call sentinel for the clamp's remembered command. A clamp that has
#: never seen a command must pass the first one through rather than rate-limit
#: it against zero. After a teleport the remembered command describes an arm
#: that no longer exists, so the reset must put this sentinel back
#: (:meth:`pusht_drake.sim.reset.DirectResetter._reset_clamp_state`).
NO_COMMAND_YET = -999


class PlanarCommandToPose(LeafSystem):
    """``[x, y]`` in, the pusher pose differential IK should track out.

    The height and the downward z-axis are the convention; only x and y are
    commanded. See :mod:`pusht_drake.sim.planar_pose`.
    """

    def __init__(self, z_dist: float = 0.02) -> None:
        super().__init__()
        self._z_dist = z_dist
        self.DeclareVectorInputPort("planar_position_input", 2)
        self.DeclareAbstractOutputPort(
            "rigid_transform_output",
            lambda: AbstractValue.Make(RigidTransform()),
            self.DoCalcOutput,
        )

    def DoCalcOutput(self, context, output) -> None:
        planar_translation = self.EvalVectorInput(context, 0).get_value()
        planar_pose = PlanarPose(planar_translation[0], planar_translation[1], 0.0)
        output.set_value(planar_pose.to_pose(z_value=self._z_dist))


class JointVelocityClamp(LeafSystem):
    """Rate-limit the commanded joint positions, per joint.

    A second safety layer under differential IK: the QP already respects the
    velocity limits, but this bounds what actually reaches the driver, and it is
    what stops a teleport from being followed by the arm sweeping across the
    table at the limit to catch up with a stale command.

    THE EVALUATION TIMING IS THE BEHAVIOUR. The clamp measures its own timestep
    as ``context.get_time() - last_time`` and writes all three discrete states
    from inside the output-port calc callback, so what it produces depends on
    *when and how often* the port is evaluated, not only on the command. That is
    not the split Drake asks for -- the framework-shaped version declares a
    periodic update that advances the state and keeps the output a pure function
    of ``(state, input)`` -- but changing it changes the trajectory, so this
    layer reproduces the upstream callback line for line and the rewrite is
    deferred to its own layer where it can be measured on its own.

    Two consequences of the shape worth naming, because both are load-bearing
    and neither is obvious:

    - The manual writes survive the framework's own discrete update. Drake
      applies an update buffer only to subsystems that declared events, and this
      one declares none, so its context state is never overwritten from a stale
      copy.
    - The output is cached per evaluation, so during a 1 ms plant step the
      driver's several reads of it produce one calc, one ``last_time`` write,
      and therefore one timestep of exactly the plant period. Publish events at
      1/64 s (the visualizers) never touch this port, so they cannot corrupt
      that measurement.

    The upstream acceleration limit (0.1x the velocity limit) is not reproduced:
    it was commented out upstream and has never run.
    """

    def __init__(self, num_positions, joint_velocity_limits) -> None:
        LeafSystem.__init__(self)
        self._num_positions = num_positions
        self._joint_velocity_limits = joint_velocity_limits
        self._joint_positions_commanded = self.DeclareVectorInputPort(
            "joint_positions_commanded", num_positions
        )
        self.DeclareVectorOutputPort("joint_positions_clamped", num_positions, self.DoCalcOutput)

        self._last_velocity = self.DeclareDiscreteState([0.0] * num_positions)
        self._last_command = self.DeclareDiscreteState([NO_COMMAND_YET] * num_positions)
        self._last_time = self.DeclareDiscreteState([0.0])

    def reset(self, context) -> None:
        """Forget the remembered command. Call this after teleporting the arm.

        Without it the clamp walks the command from the old arm pose toward the
        new one at the velocity limit while the plant is already elsewhere, and
        the driver drags the arm across the table, over whatever was just placed
        there. ``last_time`` is deliberately NOT reset: the sentinel already
        makes the next call a passthrough, and upstream's reset left it alone.
        """
        n = self._num_positions
        context.get_mutable_discrete_state(self._last_command).set_value(
            [float(NO_COMMAND_YET)] * n
        )
        context.get_mutable_discrete_state(self._last_velocity).set_value([0.0] * n)

    def DoCalcOutput(self, context, output) -> None:
        joint_positions_commanded = self._joint_positions_commanded.Eval(context)
        last_velocity_state = context.get_discrete_state(self._last_velocity)
        last_command_state = context.get_discrete_state(self._last_command)
        last_time_state = context.get_discrete_state(self._last_time)
        time_step = context.get_time() - last_time_state.get_value()[0]

        last_command = last_command_state.get_value()
        if time_step == 0 or (last_command == [NO_COMMAND_YET] * self._num_positions).all():
            joint_positions_clamped = joint_positions_commanded
            velocity_clamped = [0.0] * self._num_positions
        else:
            joint_positions_clamped = [0.0] * self._num_positions
            velocity_clamped = [0.0] * self._num_positions
            for i in range(self._num_positions):
                vel_sign = 1.0 if joint_positions_commanded[i] > last_command[i] else -1.0
                velocity = (joint_positions_commanded[i] - last_command[i]) / time_step
                speed = abs(velocity)
                if speed > self._joint_velocity_limits[i]:
                    # Above speed limit
                    joint_positions_clamped[i] = (
                        last_command[i] + self._joint_velocity_limits[i] * time_step * vel_sign
                    )
                    velocity_clamped[i] = self._joint_velocity_limits[i] * vel_sign
                else:
                    joint_positions_clamped[i] = joint_positions_commanded[i]
                    velocity_clamped[i] = speed * vel_sign
            if (joint_positions_clamped != joint_positions_commanded).any():
                logger.warning(
                    "(%s) clamped joint_positions_commanded deltas %s",
                    context.get_time(),
                    abs((joint_positions_commanded - last_command) / time_step),
                )

        output.set_value(joint_positions_clamped)
        last_command_state.set_value(joint_positions_clamped)
        last_time_state.set_value([context.get_time()])
        last_velocity_state.set_value(velocity_clamped)


class RobotStateToPusherPose(LeafSystem):
    """Arm state ``[q, v]`` in, the pusher body's world pose out.

    Forward kinematics against a private context of the full plant. Only the
    arm's positions are written into it, and the pusher's pose depends on
    nothing else, so the T sitting at its default pose in that context is
    immaterial.
    """

    def __init__(self, plant: MultibodyPlant, robot_model_name: str) -> None:
        super().__init__()
        self._plant = plant
        self._plant_context = self._plant.CreateDefaultContext()
        self._robot_model_instance_index = plant.GetModelInstanceByName(robot_model_name)
        self._num_positions = self._plant.num_positions(self._robot_model_instance_index)
        self._num_velocities = self._plant.num_velocities(self._robot_model_instance_index)
        self._pusher_body = self._plant.GetBodyByName("pusher")

        self.DeclareVectorInputPort("state", self._num_positions + self._num_velocities)
        self.DeclareAbstractOutputPort(
            "pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self.DoCalcOutput,
        )

    def DoCalcOutput(self, context, output) -> None:
        robot_state = self.EvalVectorInput(context, 0).get_value()
        q = robot_state[: self._num_positions]
        self._plant.SetPositions(self._plant_context, self._robot_model_instance_index, q)
        output.set_value(self._plant.EvalBodyPoseInWorld(self._plant_context, self._pusher_body))


class SliderStateToPose(LeafSystem):
    """T state ``[q, v]`` in, the T body's world pose out.

    ``q`` is the free body's ``[quaternion, translation]``, straight off the
    plant's per-instance state port, so it carries whatever quaternion drift the
    integration has accumulated; ``RigidTransform`` normalizes on construction.
    """

    def __init__(self, plant: MultibodyPlant, object_model_name: str) -> None:
        super().__init__()
        self._plant = plant
        self._plant_context = self._plant.CreateDefaultContext()
        self._object_model_instance_index = plant.GetModelInstanceByName(object_model_name)
        self._object_body = self._plant.GetBodyByName(object_model_name)
        self._num_positions = self._plant.num_positions(self._object_model_instance_index)
        self._num_velocities = self._plant.num_velocities(self._object_model_instance_index)

        self.DeclareVectorInputPort("state", self._num_positions + self._num_velocities)
        self.DeclareAbstractOutputPort(
            "pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self.DoCalcOutput,
        )

    def DoCalcOutput(self, context, output) -> None:
        object_state = self.EvalVectorInput(context, 0).get_value()
        q = object_state[: self._num_positions]
        self._plant.SetPositions(self._plant_context, self._object_model_instance_index, q)
        output.set_value(self._plant.EvalBodyPoseInWorld(self._plant_context, self._object_body))
