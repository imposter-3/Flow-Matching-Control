"""Build the plant: the arm, the table, the pedestal, and the T it pushes.

This replaces the upstream copy of the ``manipulation`` package's
``MakeHardwareStation`` (Michaelszeng/diffusion-policy-drake; see NOTICE.md):
1,492 lines written to serve a teaching course's needs, with YAML scenario
schemas, LCM buses, hardware/simulation symmetry, Schunk grippers and point
clouds. About 150 of those lines were reachable here, and what they did is
below.

Four things that copy provided are not reproduced:

- A generic controller-plant builder. It walked a directives tree to find the
  arm's welded children and froze their joints, so that an arbitrary scene could
  yield an arm-only model for inverse dynamics. That generality is unnecessary
  when the arm's model is a file you wrote: ``arm.yaml`` IS the controller
  plant, and the full scene includes it by reference. The upstream station then
  built the arm-only plant three times over -- once inside the driver setup,
  once for the planner's IK, once for differential IK -- from two different code
  paths; there is one here.
- LCM buses. The scenario declared one with an empty URL, and a second null bus
  was then injected to opt out of it. Simulation needs neither: the default
  visualizers open their own LCM.
- The two RgbdSensors. The config still describes an overhead and a wrist
  camera, and the upstream loader assembled ~100 lines of ``CameraConfig`` for
  them, including two VTK render engines, on every rig build. Nothing in
  evaluation, replay or the check suite reads an image; they rendered 640x480
  twice per 100 ms of simulation and were published to a socket nobody listened
  on. Rendering never touches plant state, so this changes no trajectory.
- The disturbance systems and the success checker, guarded behind config keys
  that are zero in every config in this repository.

The three MultibodyPlants this project ends up with are worth naming, because
reaching for the wrong one is a bug nothing reports (its ``nv`` includes the T's
degrees of freedom):

    plant             the world: arm + pusher + table + pedestal + T
    controller_plant  arm + pusher only, for inverse dynamics, differential IK
                      and the reset's IK solve
    (Drake internally builds a third inside the driver)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    ApplyMultibodyPlantConfig,
    ApplyVisualizationConfig,
    DiagramBuilder,
    LoadModelDirectives,
    ModelInstanceIndex,
    MultibodyPlant,
    MultibodyPlantConfig,
    Parser,
    ProcessModelDirectives,
    SceneGraph,
    VisualizationConfig,
)

from pusht_drake.sim.env_config import EnvConfig
from pusht_drake.sim.slider_sdf import ensure_slider_sdf

MODELS_DIR = Path(__file__).resolve().parents[1] / "assets" / "models"
PACKAGE_NAME = "pusht_drake"

#: The arm's model instance name. Ports and lookups key on it.
IIWA_MODEL_NAME = "iiwa"

#: Contact settings, exactly the upstream scenario's "workspace-optimized"
#: plant_config. Strict ``"hydroelastic"``, not ``hydroelastic_with_fallback``:
#: a geometry pair lacking proximity properties must raise rather than fall back
#: to point contact with no error. Everything else -- ``stiction_tolerance``
#: (1e-4 m/s), ``penetration_allowance``, ``sap_near_rigid_threshold``,
#: ``contact_surface_representation`` -- is left at Drake's default, which is
#: what the scenario YAML did by omission.
CONTACT_MODEL = "hydroelastic"
CONTACT_APPROXIMATION = "sap"


@dataclass(frozen=True)
class Scene:
    """Handles into the built plant. Holding one keeps the controller plant alive.

    That is not incidental: ``SimIiwaDriver`` keeps an internal reference to
    ``controller_plant`` and Drake requires it to outlive the diagram, so this
    object must outlive the diagram too. The upstream station bought the same
    guarantee with a ``SharedPointerSystem`` wired into the diagram.
    """

    plant: MultibodyPlant
    scene_graph: SceneGraph
    controller_plant: MultibodyPlant
    iiwa: ModelInstanceIndex
    slider: ModelInstanceIndex
    driver: object

    @property
    def num_iiwa_positions(self) -> int:
        return self.controller_plant.num_positions()


def _configure_parser(parser: Parser) -> None:
    """Teach the parser where ``package://pusht_drake/...`` lives."""
    parser.package_map().Add(PACKAGE_NAME, str(MODELS_DIR))


def build_controller_plant(config: EnvConfig) -> MultibodyPlant:
    """The arm's model of itself: iiwa + welded pusher, no world, no T.

    The driver's inverse dynamics, differential IK and the reset's IK all solve
    against this one plant. It is the same file the full scene includes, so the
    two cannot drift.
    """
    plant = MultibodyPlant(config.time_step)
    parser = Parser(plant)
    _configure_parser(parser)
    ProcessModelDirectives(LoadModelDirectives(str(MODELS_DIR / "arm.yaml")), plant, parser)
    plant.Finalize()
    return plant


def parse_world(
    builder: DiagramBuilder,
    config: EnvConfig,
) -> tuple[MultibodyPlant, SceneGraph, ModelInstanceIndex, ModelInstanceIndex]:
    """The physical world alone: arm, table, pedestal and T, finalized.

    Returns ``(plant, scene_graph, iiwa, slider)``. The model instances are
    added in the order ``scene.yaml`` declares them and the T last, which is the
    order the upstream scenario produced and the order the contact solver sees.
    """
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=config.time_step)
    ApplyMultibodyPlantConfig(
        MultibodyPlantConfig(
            time_step=config.time_step,
            contact_model=CONTACT_MODEL,
            discrete_contact_approximation=CONTACT_APPROXIMATION,
        ),
        plant,
    )

    parser = Parser(plant)
    _configure_parser(parser)
    ProcessModelDirectives(LoadModelDirectives(str(MODELS_DIR / "scene.yaml")), plant, parser)
    # The T is generated rather than shipped: its inertia follows from the
    # configured mass, and its contact properties are configuration.
    slider_models = parser.AddModels(str(ensure_slider_sdf(config.slider)))
    plant.Finalize()
    return plant, scene_graph, plant.GetModelInstanceByName(IIWA_MODEL_NAME), slider_models[0]


def build_scene(
    builder: DiagramBuilder,
    config: EnvConfig,
    *,
    meshcat=None,
) -> Scene:
    """Add the whole physical world, its driver and its visualizer to ``builder``."""
    from pydrake.manipulation import IiwaDriver, SimIiwaDriver

    plant, scene_graph, iiwa, slider = parse_world(builder, config)
    controller_plant = build_controller_plant(config)
    # position_only and an empty hand: the scenario's own driver settings. The
    # driver interpolates the commanded position into a desired state, runs
    # inverse dynamics against controller_plant, and saturates the torque.
    driver = SimIiwaDriver.AddToBuilder(
        builder=builder,
        plant=plant,
        iiwa_instance=iiwa,
        driver_config=IiwaDriver(hand_model_name="", control_mode="position_only"),
        controller_plant=controller_plant,
    )

    if meshcat is not None:
        # Drake's defaults, which is what the upstream scenario used: meshcat
        # illustration, proximity and inertia visualizers plus contact arrows,
        # and the LCM DrakeVisualizer publishers, all at 1/64 s. Those publish
        # events sit off the 1 ms plant grid, which is safe only because no
        # publish path ever evaluates the joint-velocity clamp (see
        # pusht_drake.sim.systems.JointVelocityClamp).
        ApplyVisualizationConfig(VisualizationConfig(), builder, meshcat=meshcat)

    return Scene(
        plant=plant,
        scene_graph=scene_graph,
        controller_plant=controller_plant,
        iiwa=iiwa,
        slider=slider,
        driver=driver,
    )
