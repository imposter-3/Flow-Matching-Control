"""The headless Drake evaluation rig: one station, built once, reused per episode.

The canonical construction recipe, packaged: parsed env config + station +
position source + branch-seeded resetter, realtime unlocked. "Headless" here
means what it means everywhere in this repo: StartMeshcat() (a websocket server
nobody opens) plus set_target_realtime_rate(0.0). The Meshcat instance is not a
requirement of the station any more (build_station accepts meshcat=None), but
the evaluation keeps one so that a run can be watched, recorded or replayed
through the same code path that produced the numbers.

Two source flavors, selected at build time:

- source="policy" (evaluation): an ActionChunkPolicySource. The
  policy runs inside the diagram, commands live in Drake state, and every
  command passes the demonstration guard chain (square -> fence -> leash).
- source="fixed" (arm-dynamics tests): the classic FixedPositionSource
  with set_target, for driving the chain below the policy seam.

Two things the upstream construction needed and this one does not: an
``os.chdir`` to the repository root and an ``fcntl`` lock around config
loading. Both existed because the upstream config loader regenerated the
slider SDF and rewrote the table URDF, under relative paths, as a side effect
of parsing a config. Asset generation is now content-addressed
(pusht_drake.sim.slider_sdf) and the table is a static file, so loading a
config reads and returns.

This module may import pydrake (evaluator layer) but must not import torch:
policies arrive as Policy objects (sim.interface) through the harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EvalRig:
    """Everything one evaluation process holds. Build once; reset per episode."""

    #: The parsed YAML mapping. sample_task_init and ActionSquare read it.
    cfg: dict
    config: Any  # EnvConfig: the typed view of the same file
    action_square: Any  # ActionSquare: layer 1 of the command guard
    workspace: Any  # SafeWorkspace: layer 2, the certified fence
    station: Any  # Station: the diagram, the simulator and the probes
    source: Any  # ActionChunkPolicySource (eval) or FixedPositionSource (tests)
    resetter: Any
    push_start: Any  # PushStartConfiguration: solved once, reported by checks
    goal_pose: np.ndarray  # (3,) frozen slider goal
    control_period_s: float
    #: Recorded on every episode record so two backends' artifacts cannot be
    #: confused on disk.
    backend: str = "drake"
    engine_variant: str = "hydroelastic_sap"
    _k0: int = 0

    # -- handles ------------------------------------------------------------

    @property
    def plant(self):
        return self.station.scene.plant

    def root_context(self):
        return self.station.context

    # -- state probes (measured, from the plant) ----------------------------

    def pusher_xy(self) -> np.ndarray:
        return self.station.pusher_xy()

    def slider_pose(self) -> np.ndarray:
        """Measured T planar pose (x, y, theta) about its area centroid."""
        return self.station.slider_pose()

    def state_finite(self) -> bool:
        return self.station.state_finite()

    def sim_time(self) -> float:
        return float(self.station.simulator.get_context().get_time())

    # -- the episode --------------------------------------------------------

    def begin_episode(self, policy, init, action_seed):
        """Everything engine-specific about starting an episode, in one place.

        Order is load-bearing: set_policy -> resetter.reset -> begin_episode.
        The resetter's finally calls source.reset(start_xy), which arms the
        executor's pending-reset mailbox and, at the next update,
        policy.invalidate_warm_cache().
        """
        from pusht_drake.sim.planar_pose import PlanarPose

        self.source.set_policy(policy)
        report = self.resetter.reset(PlanarPose(*init.slider_pose), pusher_xy=init.pusher_xy)
        self.source.begin_episode(action_seed)

        # Episode grid base: the first command event that is provably
        # pending-or-future. round() lands on the nearest grid index; the
        # while-bump skips an event the settle already consumed when its time
        # sits ULPs below now.
        simulator = self.station.simulator
        period = self.control_period_s
        now = simulator.get_context().get_time()
        k0 = int(round(now / period))
        while k0 * period < now:
            k0 += 1
        if k0 * period > now:
            simulator.AdvanceTo(k0 * period)
        self._k0 = k0
        return report

    def advance_control_step(self, index: int) -> None:
        """Advance to the end of control step index.

        One multiplication against the grid base, never an accumulation:
        the check suite asserts record.t == (k0 + n) * period
        bitwise, and accumulating would also resurrect the bimodal-tracking
        defect that test exists to prevent.
        """
        self.station.simulator.AdvanceTo((self._k0 + index + 1) * self.control_period_s)


#: One Meshcat server per process, not per rig.
#:
#: The harness rebuilds the rig after every sim_error episode, and a Drake
#: diagram is dense with reference cycles, so the previous rig's server is freed
#: only whenever the collector gets round to it. A worker that hits a run of sim
#: errors therefore leaks websocket servers until the port range is exhausted, at
#: which point StartMeshcat throws from inside the rebuild. That is not a
#: BrokenProcessPool, so it escapes the harness's handler and destroys the whole
#: invocation rather than one episode.
#:
#: Reusing one server is safe because everything drawn onto it (the action square,
#: the goal pose) is an idempotent set_object keyed by path.
_MESHCAT = None


def _process_meshcat(start):
    global _MESHCAT
    if _MESHCAT is None:
        _MESHCAT = start()
    return _MESHCAT


def build_rig(
    env_config: str | Path,
    source: str = "policy",
    execution_mode: str = "bspline",
):
    """Construct the full evaluation stack for one process.

    execution_mode selects the 10 Hz-waypoint execution strategy of the
    policy source; the campaign runs "bspline". Everything else (policy,
    observation, guard, DiffIK, driver, plant) is shared.
    """

    from pydrake.all import StartMeshcat

    from pusht_drake.sim.action_square import ActionSquare
    from pusht_drake.sim.env_config import load_env_config
    from pusht_drake.sim.guard import LEASH_M
    from pusht_drake.sim.policy_source import ActionChunkPolicySource, FixedPositionSource
    from pusht_drake.sim.push_start import solve_push_start_configuration
    from pusht_drake.sim.reset import DirectResetter
    from pusht_drake.sim.scene import build_controller_plant
    from pusht_drake.sim.station import build_station
    from pusht_drake.sim.workspace import SafeWorkspace

    config = load_env_config(Path(env_config))
    square = ActionSquare.from_config(config.raw)
    workspace = SafeWorkspace.load(profile=config.robot_base_profile)
    initial_xy = np.array([config.pusher_start_pose.x, config.pusher_start_pose.y])
    if source == "policy":
        position_source: Any = ActionChunkPolicySource(
            initial_xy,
            square=square,
            workspace=workspace,
            freq=config.policy_freq,
            leash_m=LEASH_M,
            execution_mode=execution_mode,
        )
    elif source == "fixed":
        position_source = FixedPositionSource(initial_xy)
    else:
        raise ValueError(f'source must be "policy" or "fixed", got {source!r}')

    meshcat = _process_meshcat(StartMeshcat)
    station = build_station(config, position_source, meshcat=meshcat, realtime_rate=0.0)

    # The push start is solved against a controller plant of its own, so the IK
    # cannot touch the one the running diagram holds.
    push_start = solve_push_start_configuration(build_controller_plant(config), config)
    resetter = DirectResetter(station, config, push_start.q)

    goal = config.slider_goal_pose
    return EvalRig(
        cfg=config.raw,
        config=config,
        action_square=square,
        workspace=workspace,
        station=station,
        source=position_source,
        resetter=resetter,
        push_start=push_start,
        goal_pose=np.array([goal.x, goal.y, goal.theta], dtype=np.float64),
        control_period_s=config.control_period_s,
    )
