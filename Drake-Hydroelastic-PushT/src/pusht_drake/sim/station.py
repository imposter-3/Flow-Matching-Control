"""The diagram: the world, the controller that drives it, and the seam between.

The upstream stack split this across two nested layers,
``IiwaHardwareStation`` (572 lines) and ``SimulatedRealTableEnvironment`` (451),
because upstream had a second robot station and five kinds of command source to
wire differently. Here there is one of each, so there is one layer.

The control chain, in order::

    ActionChunkPolicySource   "planar_position_command"   policy, inside the diagram
      -> PlanarCommandToPose      lift (x, y) to a tip pose, z-axis down
      -> YawRelaxedDiffIk         1 kHz QP, 5-D task: tilt + xyz, yaw free
      -> JointVelocityClamp       per-joint rate limit on the command
      -> SimIiwaDriver            interpolate, inverse dynamics, saturate
      -> MultibodyPlant           1 kHz hydroelastic SAP

and the measurement chain back to the policy::

    driver position_measured + velocity_estimated -> Multiplexer
      -> RobotStateToPusherPose   -> source "pusher_pose_measured"
    plant <slider>_state
      -> SliderStateToPose        -> source "slider_pose_measured"

Nothing else drives the plant: no port switch selecting between a planner's
hold output and this chain, no disturbance injection, no in-diagram pose
logging.

Three upstream behaviours are reproduced deliberately rather than improved,
because each is a physical change that belongs to a later layer where it can be
measured on its own:

1. Differential IK reads the arm state through the DRIVER's measured ports and
   a Multiplexer, not from ``plant.get_state_output_port(iiwa)``. The driver's
   ``position_measured`` is a pass-through of the plant state but
   ``velocity_estimated`` runs through the driver's low-pass filter, so the two
   are not the same signal.
2. The rate limiter keeps upstream's ``JointVelocityClamp`` evaluation timing
   (see :mod:`pusht_drake.sim.systems`).
3. A teleport does NOT call ``set_initial_position`` on the driver's
   ``StateInterpolatorWithDiscreteDerivative``. Drake ships that method for
   exactly this case, and without it the interpolator finite-differences the old
   posture against the new one on the first tick after a reset and hands inverse
   dynamics a desired velocity of hundreds of rad/s, which the settle then
   spends part of its budget absorbing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydrake.all import Context, Diagram, DiagramBuilder, Multiplexer, Simulator

from pusht_drake.sim.env_config import EnvConfig
from pusht_drake.sim.planar_pose import PlanarPose
from pusht_drake.sim.scene import Scene, build_scene
from pusht_drake.sim.systems import (
    JointVelocityClamp,
    PlanarCommandToPose,
    RobotStateToPusherPose,
    SliderStateToPose,
)

#: The frame differential IK tracks: the pusher tip, not the cylinder's origin.
END_EFFECTOR_FRAME = "pusher_end"

#: Clearance added when placing the T, so it is never spawned interpenetrating
#: the table. Half a millimetre, which it falls through in a few steps.
SLIDER_SPAWN_CLEARANCE_M = 5e-4


@dataclass
class Station:
    """A built diagram plus everything needed to drive and inspect it."""

    diagram: Diagram
    simulator: Simulator
    scene: Scene
    source: object
    diff_ik: object
    clamp: JointVelocityClamp
    config: EnvConfig
    #: The process Meshcat, or None for a station with no visualization.
    meshcat: object | None
    #: The translucent T at the goal pose. None when there is no Meshcat.
    goal_ghost: object = None
    #: Half the T's thickness, read from its collision shapes rather than from
    #: the config, so the spawn height and the simulated body cannot disagree.
    slider_half_height: float = 0.0
    _pusher_body: object = field(default=None, repr=False)
    _slider_body: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        plant = self.scene.plant
        self._pusher_body = plant.GetBodyByName("pusher")
        self._slider_body = plant.GetBodyByName(self.config.slider.name)
        inspector = self.scene.scene_graph.model_inspector()
        heights = [
            inspector.GetShape(gid).height()
            for gid in plant.GetCollisionGeometriesForBody(self._slider_body)
        ]
        self.slider_half_height = min(heights) / 2

    # -- contexts ---------------------------------------------------------

    @property
    def context(self) -> Context:
        return self.simulator.get_mutable_context()

    @property
    def plant_context(self) -> Context:
        return self.scene.plant.GetMyContextFromRoot(self.context)

    @property
    def time(self) -> float:
        return float(self.context.get_time())

    # -- probes -----------------------------------------------------------

    def pusher_xy(self) -> np.ndarray:
        """Measured pusher position: the body origin, as the dataset records it."""
        pose = self.scene.plant.EvalBodyPoseInWorld(self.plant_context, self._pusher_body)
        return pose.translation()[:2].copy()

    def slider_pose(self) -> np.ndarray:
        """``[x, y, theta]`` of the T, about its area centroid."""
        plant = self.scene.plant
        q = plant.GetPositions(self.plant_context, self.scene.slider)
        return PlanarPose.from_generalized_coords(q).vector().astype(np.float64).ravel()

    def slider_quaternion_norm(self) -> float:
        q = self.scene.plant.GetPositions(self.plant_context, self.scene.slider)
        return float(np.linalg.norm(q[0:4]))

    def speeds(self) -> tuple[float, float]:
        """``(slider, arm)`` peak absolute generalized velocity. The settle test."""
        plant, context = self.scene.plant, self.plant_context
        slider = float(np.max(np.abs(plant.GetVelocities(context, self.scene.slider))))
        arm = float(np.max(np.abs(plant.GetVelocities(context, self.scene.iiwa))))
        return slider, arm

    def state_finite(self) -> bool:
        plant, context = self.scene.plant, self.plant_context
        return bool(
            np.all(np.isfinite(plant.GetPositions(context)))
            and np.all(np.isfinite(plant.GetVelocities(context)))
        )

    # -- the teleport -----------------------------------------------------

    def set_robot_position(self, q: np.ndarray) -> None:
        """Place the arm, zero its velocities, and resynchronise differential IK.

        Differential IK integrates its own copy of ``q``; left stale it would
        drive the arm back toward where it thinks it is.
        """
        plant, context = self.scene.plant, self.plant_context
        v = np.zeros(plant.num_velocities(self.scene.iiwa))
        plant.SetPositions(context, self.scene.iiwa, q)
        plant.SetVelocities(context, self.scene.iiwa, v)
        self.diff_ik.SetPositions(self.diff_ik.GetMyMutableContextFromRoot(self.context), q)

    def set_slider_planar_pose(self, pose: PlanarPose) -> None:
        """Place the T flat on the table at ``pose``, at rest."""
        plant, context = self.scene.plant, self.plant_context
        z_value = self.slider_half_height + SLIDER_SPAWN_CLEARANCE_M
        q = pose.to_generalized_coords(z_value, z_axis_is_positive=True)
        plant.SetPositions(context, self.scene.slider, q)
        plant.SetVelocities(context, self.scene.slider, np.zeros(6))

    def teleport(self, q_iiwa, slider_pose: PlanarPose, pusher_pose: PlanarPose) -> None:
        """Place the arm and the T, and park the command source at the pusher.

        The order is upstream's and is load-bearing: the arm first (which
        resynchronises differential IK), then the T, then the command source, so
        that releasing the input hold cannot hand differential IK a stale target.
        """
        if q_iiwa is not None:
            self.set_robot_position(np.asarray(q_iiwa, dtype=float).ravel())
        if slider_pose is not None:
            self.set_slider_planar_pose(slider_pose)
        if pusher_pose is not None:
            self.source.reset(np.array([pusher_pose.x, pusher_pose.y]))

    # -- overlays ---------------------------------------------------------

    def publish_goal(self, time_in_recording: float = 0.0) -> None:
        """(Re-)draw the goal ghost. A no-op on a station with no Meshcat."""
        if self.goal_ghost is not None:
            self.goal_ghost.publish(time_in_recording)


def build_station(
    config: EnvConfig,
    source,
    *,
    meshcat=None,
    realtime_rate: float = 0.0,
) -> Station:
    """Assemble the diagram and wrap it in a ``Simulator``.

    Expensive (seconds, mostly parsing and finalizing two plants), so build one
    per process and reuse it across episodes. ``source`` is any Diagram
    exporting ``planar_position_command`` and, optionally, the two measured-pose
    input ports; it is added to the diagram, not merely read.
    """
    from pusht_drake.sim.diffik import (
        YawRelaxedDiffIk,
        arm_velocity_limits,
        make_yaw_relaxed_params,
    )
    from pusht_drake.sim.workspace import nominal_schedule

    builder = DiagramBuilder()
    scene = build_scene(builder, config, meshcat=meshcat)
    robot = scene.controller_plant
    num_positions = robot.num_positions()

    if meshcat is not None:
        from pusht_drake.sim.viz import set_meshcat_camera

        set_meshcat_camera(meshcat, config.slider_goal_pose)

    builder.AddNamedSystem("DesiredPlanarPositionSource", source)
    to_pose = builder.AddSystem(PlanarCommandToPose(z_dist=config.pusher_z_offset))

    velocity_limit_factor = config.joint_velocity_limit_factor
    ik_params = make_yaw_relaxed_params(
        robot,
        config.default_joint_positions,
        velocity_limit_factor=velocity_limit_factor,
    )
    ik_params.set_time_step(config.time_step)
    # Radius-scheduled joint-centering nominal, from the profile's certified
    # map when it carries one. The schedule keeps the QP's null-space pull
    # pointed at a certified collision-free posture for the target's radius; a
    # constant nominal measurably drags the wrist across branches at the far
    # wall.
    try:
        schedule = nominal_schedule(config.robot_base_profile)
    except FileNotFoundError:
        schedule = None
    # secondary=None is the DELIBERATE production default, not an omission. The
    # null-space mechanism is kept (it costs nothing disabled, and it is where
    # collision avoidance would go), but the Level-4 ablation measured no benefit
    # on this task: +0.2% median sigma_min, -0.42% at the minimum, and a
    # bit-identical 11.93 deg minimum joint margin. The joint-limit objective is
    # provably unable to help -- the binding joint has exactly zero null-space
    # component. Enabling it also stops the arm ever coming to rest, which breaks
    # the reset settle criterion.
    diff_ik = builder.AddNamedSystem(
        "DiffIk",
        YawRelaxedDiffIk(
            robot,
            END_EFFECTOR_FRAME,
            config.time_step,
            ik_params,
            secondary=None,
            nominal_schedule=schedule,
        ),
    )

    # The arm state differential IK sees: position straight off the plant,
    # velocity through the driver's estimator. See the module docstring.
    robot_state = builder.AddSystem(
        Multiplexer(input_sizes=[num_positions, robot.num_velocities()])
    )
    # Velocity limits read from the loaded plant instead of a hardcoded iiwa7
    # table, so the clamp and the DiffIK QP always match the arm the profile
    # actually loads (iiwa14's J4 limit is 40% below iiwa7's).
    clamp = builder.AddNamedSystem(
        "JointVelocityClamp",
        JointVelocityClamp(
            num_positions=num_positions,
            joint_velocity_limits=velocity_limit_factor * arm_velocity_limits(robot),
        ),
    )
    pusher_pose = builder.AddNamedSystem(
        "RobotStateToPusherPose", RobotStateToPusherPose(scene.plant, "iiwa")
    )
    slider_pose = builder.AddNamedSystem(
        "SliderStateToPose", SliderStateToPose(scene.plant, config.slider.name)
    )

    # -- the command chain ------------------------------------------------
    builder.Connect(source.GetOutputPort("planar_position_command"), to_pose.get_input_port())
    builder.Connect(to_pose.get_output_port(), diff_ik.GetInputPort("X_WE_desired"))
    builder.Connect(diff_ik.get_output_port(), clamp.get_input_port())
    builder.Connect(clamp.get_output_port(), scene.driver.GetInputPort("position"))
    # diff_ik's "use_robot_state" is left unconnected, so it integrates its own
    # state. The upstream station drove it from the planner's mode, which was
    # False in the only reachable mode; driving it True was measured upstream to
    # accumulate persistent drift.

    # -- the measurement chain --------------------------------------------
    builder.Connect(scene.driver.GetOutputPort("position_measured"), robot_state.get_input_port(0))
    builder.Connect(scene.driver.GetOutputPort("velocity_estimated"), robot_state.get_input_port(1))
    builder.Connect(robot_state.get_output_port(), diff_ik.GetInputPort("robot_state"))
    builder.Connect(robot_state.get_output_port(), pusher_pose.GetInputPort("state"))
    builder.Connect(
        scene.plant.get_state_output_port(scene.slider), slider_pose.GetInputPort("state")
    )
    if source.HasInputPort("pusher_pose_measured"):
        builder.Connect(
            pusher_pose.GetOutputPort("pose"), source.GetInputPort("pusher_pose_measured")
        )
    if source.HasInputPort("slider_pose_measured"):
        builder.Connect(
            slider_pose.GetOutputPort("pose"), source.GetInputPort("slider_pose_measured")
        )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(realtime_rate)

    station = Station(
        diagram=diagram,
        simulator=simulator,
        scene=scene,
        source=source,
        diff_ik=diff_ik,
        clamp=clamp,
        config=config,
        meshcat=meshcat,
    )
    if meshcat is not None:
        from pusht_drake.sim.action_square import ActionSquare
        from pusht_drake.sim.viz import GoalGhost, add_action_square

        add_action_square(meshcat, ActionSquare.from_config(config.raw))
        station.goal_ghost = GoalGhost(
            scene.plant,
            scene.scene_graph,
            meshcat,
            station._slider_body,
            config.slider_goal_pose,
        )
        station.publish_goal()
    return station
