"""One closed-loop episode: the policy drives the Drake pusher, coverage scores it.

Episode semantics follow gym-pusht's rollout loop; the simulator underneath is
this package's Drake station.

- Open loop: the whole predicted chunk executes before the next replan
  (H_e == H_p, the baseline control regime).
- Coverage is sampled once per control step (10 Hz, the gym-pusht reward
  cadence); reward = clip(coverage / score_tau, 0, 1).
- Termination at coverage > terminate_coverage (0.95, the env-native value);
  truncation at max_steps control steps (300 = 30 s).
- Episode success = max_coverage > score_tau, an episode-level judgment.

The policy itself runs inside the diagram (ActionChunkPolicySource): a periodic
unrestricted update event predicts, guards and stores each command in Drake
state, so this loop is only the experiment driver. It schedules grid steps,
samples coverage, checks termination, and drains the controller's telemetry at
the end. Two scheduling facts it leans on, verified against Drake 1.48:

- Drake schedules the k-th periodic event at k * period computed by a single
  multiplication, so the loop's absolute targets bit-match the event schedule
  and every AdvanceTo fires exactly one command event: the one deferred from
  its own target time, because an event landing exactly on an AdvanceTo
  boundary runs at the start of the next call. After iteration n the plant
  state reflects command n having acted for one full period.
- The resetter's settle loop advances relatively, so the settle can end ULPs
  past a grid point whose event already fired (held) during settling; the
  episode bases itself on the first event that is provably pending-or-future.

Diagnostics the pymunk world had no analog for: the commanded jump per step
(chunk-boundary vs within-chunk), the tracking error between the measured
pusher and the command that just finished its period, and the guard tick rates
(square / fence / leash), which say how often the demonstration-time guard
chain had to move what the policy asked for.

The final slider tolerance error (15 mm / 3.5 deg, slider only) is also
reported, for continuity with the teleop-era statistics. The upstream checker's
extra pusher-near-start criterion is a startup artifact this evaluation drops.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from pusht_drake.sim.coverage import coverage
from pusht_drake.sim.interface import Policy, RolloutLimits
from pusht_drake.sim.spawn import TaskInit


@dataclass(frozen=True)
class EpisodeResult:
    """Per-episode outcome and behaviour diagnostics."""

    env_seed: int
    action_seed: int
    length: int  # control steps executed
    num_replans: int
    max_coverage: float
    final_coverage: float
    max_reward: float
    final_reward: float
    success: bool  # max_coverage > score_tau
    terminated: bool  # hit terminate_coverage
    truncated: bool  # hit max_steps
    tolerance_success: bool  # final slider within 15 mm / 3.5 deg
    final_trans_err_m: float
    final_rot_err_rad: float
    clip_rate: float
    # Replans that fell back to a cold source. Exactly 1 for a warm policy
    # (every episode's first replan is genuinely cold) and 0 for a vanilla
    # one; anything else means the forecast cache broke mid-episode.
    cold_fallbacks: int
    boundary_jump_mean_m: float
    boundary_jump_max_m: float
    within_chunk_step_mean_m: float
    tracking_err_mean_m: float
    tracking_err_max_m: float
    smoothness_m: float  # mean |second difference| of executed anchors
    spline_clip_rate: float  # containment clips per control step (expected 0)
    square_tick_rate: float  # fraction of commands the square clamp moved
    fence_tick_rate: float  # fraction the certified fence moved (should be ~0)
    leash_tick_rate: float  # fraction the 5 cm leash rate-limited
    settle_time_s: float
    wall_time_s: float
    realtime_factor: float
    termination_reason: str = "timeout"  # goal_coverage | timeout | sim_error
    #: Which simulator produced this record. Without it two backends' artifacts
    #: are indistinguishable on disk, which is how a surrogate number ends up
    #: quoted as a Drake one. Defaulted so older artifacts still parse.
    backend: str = "drake"
    engine_variant: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


TRANS_TOL_M = 0.015
ROT_TOL_RAD = np.deg2rad(3.5)


def _wrapped_angle_error(theta: float, goal_theta: float) -> float:
    delta = (theta - goal_theta + np.pi) % (2.0 * np.pi) - np.pi
    return abs(float(delta))


def rollout_episode(
    rig,
    policy: Policy,
    init: TaskInit,
    *,
    env_seed: int,
    action_seed: int,
    limits: RolloutLimits,
) -> EpisodeResult:
    """Run one episode from a frozen-distribution init. Raises on sim instability.

    rig is duck-typed rather than annotated: the six methods below are the whole
    contract, so another simulator can drive this same loop without a Backend
    base class.

        begin_episode(policy, init, action_seed) -> ResetReport
        advance_control_step(n) -> None      n is the 0-based step index
        sim_time() -> float
        slider_pose() / pusher_xy() / state_finite()

    plus goal_pose, control_period_s, backend, engine_variant, and a
    source with the seven episode-control methods ChunkExecutor provides.
    """

    wall_start = time.perf_counter()

    # Everything engine-specific about starting an episode (clearing the pose
    # logs, wrapping the init for the resetter, aligning onto the command grid)
    # lives behind this one call, so the loop below is engine-neutral.
    report = rig.begin_episode(policy, init, action_seed)

    goal = rig.goal_pose
    score_tau = limits.score_tau
    slider = rig.slider_pose()
    running_coverage = coverage(slider, goal)
    max_coverage = running_coverage

    steps = 0
    terminated = truncated = False
    sim_time_start = rig.sim_time()

    while not (terminated or truncated):
        rig.advance_control_step(steps)
        if not rig.state_finite():
            raise RuntimeError(f"non-finite plant state at control step {steps}")

        slider = rig.slider_pose()
        running_coverage = coverage(slider, goal)
        max_coverage = max(max_coverage, running_coverage)
        steps += 1
        if running_coverage > limits.terminate_coverage:
            terminated = True
        elif steps >= limits.max_steps:
            truncated = True

    commands, tracking_errors = rig.source.drain_telemetry()
    spline_clip_ticks = rig.source.take_spline_clip_ticks()
    if len(commands) != steps:
        raise RuntimeError(
            f"command scheduling drifted: {len(commands)} command events for "
            f"{steps} control steps (the one-event-per-AdvanceTo invariant broke)"
        )
    # The controller samples tracking at event k+1 against command k, so the
    # final command's full period has no in-diagram sample; take it here, from
    # exactly the state the loop terminated on.
    tracking_errors = list(tracking_errors)
    tracking_errors.append(float(np.linalg.norm(rig.pusher_xy() - commands[-1].executed)))

    executed = np.array([c.executed for c in commands])
    replanned = np.array([c.replanned for c in commands], dtype=bool)
    num_replans = int(replanned.sum())
    # Jump into command i+1 is a chunk boundary iff command i+1 opened a new
    # chunk; the first command of the episode has no predecessor and is
    # excluded.
    jumps = np.linalg.norm(np.diff(executed, axis=0), axis=1)
    boundary_jumps = jumps[replanned[1:]]
    within_chunk_steps = jumps[~replanned[1:]]

    smoothness = 0.0
    if len(executed) >= 3:
        smoothness = float(
            np.linalg.norm(executed[2:] - 2 * executed[1:-1] + executed[:-2], axis=1).mean()
        )

    final_coverage = running_coverage
    trans_err = float(np.linalg.norm(slider[:2] - goal[:2]))
    rot_err = _wrapped_angle_error(slider[2], goal[2])
    wall = time.perf_counter() - wall_start
    sim_elapsed = rig.sim_time() - sim_time_start

    return EpisodeResult(
        env_seed=env_seed,
        action_seed=action_seed,
        length=steps,
        num_replans=num_replans,
        max_coverage=float(max_coverage),
        final_coverage=float(final_coverage),
        max_reward=float(np.clip(max_coverage / score_tau, 0.0, 1.0)),
        final_reward=float(np.clip(final_coverage / score_tau, 0.0, 1.0)),
        success=bool(max_coverage > score_tau),
        terminated=terminated,
        truncated=truncated,
        tolerance_success=bool(trans_err <= TRANS_TOL_M and rot_err <= ROT_TOL_RAD),
        final_trans_err_m=trans_err,
        final_rot_err_rad=rot_err,
        clip_rate=float(getattr(policy, "clip_rate", 0.0)),
        cold_fallbacks=int(getattr(policy, "cold_fallbacks", 0)),
        boundary_jump_mean_m=float(np.mean(boundary_jumps)) if len(boundary_jumps) else 0.0,
        boundary_jump_max_m=float(np.max(boundary_jumps)) if len(boundary_jumps) else 0.0,
        within_chunk_step_mean_m=(
            float(np.mean(within_chunk_steps)) if len(within_chunk_steps) else 0.0
        ),
        tracking_err_mean_m=float(np.mean(tracking_errors)),
        tracking_err_max_m=float(np.max(tracking_errors)),
        smoothness_m=smoothness,
        spline_clip_rate=float(spline_clip_ticks) / float(steps),
        square_tick_rate=float(np.mean([c.ticks.square for c in commands])),
        fence_tick_rate=float(np.mean([c.ticks.fence for c in commands])),
        leash_tick_rate=float(np.mean([c.ticks.leash for c in commands])),
        settle_time_s=float(report.settle_time_s),
        wall_time_s=float(wall),
        realtime_factor=float(sim_elapsed / wall) if wall > 0 else 0.0,
        termination_reason="goal_coverage" if terminated else "timeout",
        backend=getattr(rig, "backend", "drake"),
        engine_variant=getattr(rig, "engine_variant", ""),
    )
