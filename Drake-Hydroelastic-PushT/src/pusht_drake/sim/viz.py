"""Meshcat overlays: the camera, the action square, and the goal ghost.

Visualization only. Every overlay is drawn from the same object the simulator
or the guard uses -- the T's own collision shapes, the ``ActionSquare`` the
command chain clamps against -- so the visible marker and the enforced or
simulated thing cannot disagree. That is the only reason these helpers exist
instead of literal coordinates at the call site.

The goal ghost is also registered into the SceneGraph as anchored geometry with
an illustration role and no proximity role, which is what the upstream
``create_goal_geometries`` did. It adds no contact geometry and no plant
presence; it exists so the goal appears in anything that renders from the scene
graph rather than from Meshcat alone.
"""

from __future__ import annotations

import numpy as np
from pydrake.all import (
    GeometryInstance,
    MakePhongIllustrationProperties,
    Rgba,
)

from pusht_drake.sim.action_square import ActionSquare, add_action_square_visualization
from pusht_drake.sim.planar_pose import PlanarPose

#: The goal ghost's colour: upstream's palette's "emeraldgreen"
#: (RGB 0, 201, 87, normalized by 256) at alpha 0.3.
GOAL_GHOST_RGBA = (0.0, 0.78515625, 0.33984375, 0.3)

#: How far back the canonical view sits from the goal. Larger is closer.
CAMERA_ZOOM = 1.8


def set_meshcat_camera(meshcat, goal_pose: PlanarPose) -> None:
    """Point a live Meshcat at the canonical view: just in front of the goal.

    Slightly above the table, looking at the goal T. The live tab, HTML replays
    and any video export all anchor to this one frame, so a recording looks like
    the session it came from.
    """
    camera_in_world = [
        goal_pose.x,
        (goal_pose.y - 1) / CAMERA_ZOOM,
        1.5 / CAMERA_ZOOM,
    ]
    target_in_world = [goal_pose.x, goal_pose.y, 0]
    meshcat.SetCameraPose(camera_in_world, target_in_world)


def add_action_square(meshcat, square: ActionSquare) -> None:
    """Draw the action square: thin light-green boundary, no fill.

    The action square is the only region a normal session shows; the certified
    fence underneath it is a safety net, not an aiming target.
    """
    add_action_square_visualization(meshcat, square)


class GoalGhost:
    """A translucent T at the frozen goal pose, in Meshcat and in the SceneGraph.

    Built from the T's own collision shapes and their poses in its body frame,
    read out of the scene graph's model inspector, so the drawn ghost and the
    scored geometry cannot disagree.

    Only the slider goal is drawn. The upstream station also drew a pusher-goal
    cylinder at ``pusher_start_pose``, which on this profile is (0.500, 0.000),
    the action square's centre -- exactly on top of the goal T, occluding it,
    and marking a pose no episode starts from since every episode's pusher start
    is sampled.
    """

    def __init__(self, plant, scene_graph, meshcat, slider_body, goal_pose: PlanarPose) -> None:
        self._meshcat = meshcat
        inspector = scene_graph.model_inspector()
        geometry_ids = plant.GetCollisionGeometriesForBody(slider_body)
        self._shapes = [inspector.GetShape(gid) for gid in geometry_ids]
        self._poses = [inspector.GetPoseInFrame(gid) for gid in geometry_ids]
        # The T lies flat on the table, so half its (uniform) thickness puts the
        # ghost's body frame where the simulated T's would be.
        self._height = min(shape.height() for shape in self._shapes)
        self._goal_pose = goal_pose

        rgba = np.array(GOAL_GHOST_RGBA)
        desired_pose = self._desired_pose()
        source_id = scene_graph.RegisterSource()
        self._names = []
        for index, (shape, pose) in enumerate(zip(self._shapes, self._poses)):
            geometry_id = scene_graph.RegisterAnchoredGeometry(
                source_id,
                GeometryInstance(desired_pose.multiply(pose), shape, f"shape_{index}"),
            )
            scene_graph.AssignRole(source_id, geometry_id, MakePhongIllustrationProperties(rgba))
            name = f"goal_shape_{index}"
            self._names.append(name)
            meshcat.SetObject(name, shape, rgba=Rgba(*rgba))

    def _desired_pose(self):
        return self._goal_pose.to_pose(self._height / 2, z_axis_is_positive=True)

    def publish(self, time_in_recording: float = 0.0) -> None:
        """(Re-)emit the ghost's transforms.

        Called again by the recorder: ``StaticHtml`` only serializes commands
        issued after ``StartRecording``, so a ghost drawn earlier would be
        missing from a saved replay, and the transforms have to carry the
        recording time to land inside the animation timeline.
        """
        desired_pose = self._desired_pose()
        for pose, name in zip(self._poses, self._names):
            self._meshcat.SetTransform(name, desired_pose.multiply(pose), time_in_recording)
