"""The environment config: one YAML in, one frozen dataclass out.

Pure parse. This module reads a file and returns objects; it writes nothing,
executes nothing named in the config, and resolves no ``_target_`` strings.
It replaces ``PlanarPushingSimConfig.from_yaml``, which used hydra to
instantiate classes named in the file, ``eval()``'d the contact model, and --
as a side effect of *loading a config* -- regenerated the slider SDF and
rewrote the table URDF in place under the repository root. That side effect is
why every entry point needed an ``os.chdir`` and an ``fcntl`` lock; asset
generation now lives in :mod:`pusht_drake.sim.slider_sdf`, is content-addressed
and is idempotent.

The raw mapping is kept alongside the typed view: :mod:`pusht_drake.sim.spawn`
and :mod:`pusht_drake.sim.action_square` read the config by key, and the
evaluation harness passes it straight through to them.

Fields the upstream dataclass carried and nothing reads are not reproduced:
the five disturbance blocks (all zero, and their systems were never built),
the multi-run/success-criteria block (evaluation scores by coverage), the
gamepad block (teleoperation lives in the sibling suite), and the camera
configs (~100 lines of CameraConfig assembly for two RgbdSensors whose images
this evaluation never reads).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pusht_drake.sim.planar_pose import PlanarPose
from pusht_drake.sim.tblock import PUSHT_T, TBlock


class ConfigError(ValueError):
    """The config file is missing a key, or carries one that means nothing."""


@dataclass(frozen=True)
class SliderConfig:
    """The T's physical properties. Its *geometry* comes from :data:`PUSHT_T`."""

    #: Model instance and link name in the plant. Port names derive from it
    #: (``arbitrary_state``), so it is part of the diagram's wiring contract.
    name: str
    mass: float
    inertia: np.ndarray  # 3x3 about the body origin, which is the area centroid
    hydroelastic_modulus: float
    mu_dynamic: float
    mu_static: float
    mesh_resolution_hint: float
    rgba: tuple[float, float, float, float]

    @property
    def block(self) -> TBlock:
        return PUSHT_T


@dataclass(frozen=True)
class EnvConfig:
    """Everything the simulator needs, resolved and validated."""

    slider: SliderConfig
    pusher_radius: float
    #: Height of the pusher tip above the tabletop (world z = 0).
    pusher_z_offset: float
    #: Nominal pusher start. Per-episode starts are sampled; this anchors
    #: ``default_joint_positions`` and the certified nominal-posture branch.
    pusher_start_pose: PlanarPose
    slider_goal_pose: PlanarPose
    time_step: float
    default_joint_positions: np.ndarray
    joint_velocity_limit_factor: float
    #: The frozen base-placement profile: selects the certified fence map and
    #: the nominal-posture schedule.
    robot_base_profile: str
    policy_freq: float
    source_path: Path
    #: The parsed YAML itself. spawn and action_square read it by key.
    raw: dict[str, Any]

    @property
    def control_period_s(self) -> float:
        return 1.0 / self.policy_freq


def _require(mapping: dict, key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{where}: missing required key {key!r}")
    return mapping[key]


def _planar_pose(mapping: dict, where: str) -> PlanarPose:
    return PlanarPose(
        float(_require(mapping, "x", where)),
        float(_require(mapping, "y", where)),
        float(mapping.get("theta", 0.0)),
    )


def load_env_config(path: Path | str) -> EnvConfig:
    """Read and validate the environment config."""
    import yaml

    path = Path(path)
    if not path.exists():
        raise ConfigError(f"no config at {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    where = str(path)

    slider_type = str(_require(raw, "slider_type", where))
    if slider_type != "arbitrary":
        raise ConfigError(
            f"{where}: only slider_type 'arbitrary' is built here, got {slider_type!r}"
        )
    contact_model = str(_require(raw, "contact_model", where))
    if contact_model != "ContactModel.kHydroelastic":
        raise ConfigError(
            f"{where}: only hydroelastic contact is built here, got {contact_model!r}"
        )

    physical = _require(raw, "physical_properties", where)
    inertia = np.asarray(_require(physical, "inertia", "physical_properties"), dtype=float)
    if inertia.shape != (3, 3):
        raise ConfigError(f"physical_properties.inertia must be 3x3, got {inertia.shape}")

    return EnvConfig(
        slider=SliderConfig(
            name=slider_type,
            mass=float(_require(physical, "mass", "physical_properties")),
            inertia=inertia,
            hydroelastic_modulus=float(
                _require(physical, "hydroelastic_modulus", "physical_properties")
            ),
            mu_dynamic=float(_require(physical, "mu_dynamic", "physical_properties")),
            mu_static=float(_require(physical, "mu_static", "physical_properties")),
            mesh_resolution_hint=float(
                _require(physical, "mesh_resolution_hint", "physical_properties")
            ),
            rgba=tuple(float(v) for v in _require(raw, "arbitrary_shape_rgba", where)),
        ),
        pusher_radius=float(_require(raw, "pusher_radius", where)),
        pusher_z_offset=float(_require(raw, "pusher_z_offset", where)),
        pusher_start_pose=_planar_pose(
            _require(raw, "pusher_start_pose", where), "pusher_start_pose"
        ),
        slider_goal_pose=_planar_pose(_require(raw, "slider_goal_pose", where), "slider_goal_pose"),
        time_step=float(_require(raw, "time_step", where)),
        default_joint_positions=np.asarray(
            _require(raw, "default_joint_positions", where), dtype=float
        ),
        joint_velocity_limit_factor=float(raw.get("joint_velocity_limit_factor", 1.0)),
        robot_base_profile=str(_require(raw, "robot_base_profile", where)),
        policy_freq=float(
            _require(
                _require(raw, "data_collection_config", where),
                "policy_freq",
                "data_collection_config",
            )
        ),
        source_path=path,
        raw=raw,
    )
