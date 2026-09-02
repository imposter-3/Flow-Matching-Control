"""Evaluate a policy over seeded episodes, serially or across worker processes.

Episode i draws its initial state from the frozen evaluation distribution,
sample_task_init(cfg, default_rng(SeedSequence([env_seed, i]))), the same
sampling teleop resets use, and its action stream from action_seed + i.
Workers stripe the episode indices; per-episode seeding keeps each episode's
initial state and action stream a pure function of its index.

Determinism contract. The same worker configuration reproduces itself bitwise
(fresh rigs per call). Across worker counts, task-level outcomes agree to the
reset's settle tolerances but not bitwise: a persistent rig hands episode k the
microscopic plant state episode k-1 left behind, and the reset erases it only
to tolerance. The check suite pins both strengths.

Policies cross the process boundary as a PolicySpec (module, factory, kwargs):
the worker imports the module and calls the factory. That keeps this package
torch-free, since torch enters the worker through the factory's module (e.g.
pusht_drake.fm.adapter), never through pusht_drake.sim.
"""

from __future__ import annotations

import importlib
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from pusht_drake.sim.interface import Policy, RolloutLimits


@dataclass(frozen=True)
class PolicySpec:
    """A picklable recipe for constructing a Policy inside a worker."""

    module: str
    factory: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self) -> Policy:
        target: Any = importlib.import_module(self.module)
        for part in self.factory.split("."):
            target = getattr(target, part)
        return target(**self.kwargs)


def _check_clip_box(policy, action_square) -> None:
    """Raise unless the checkpoint's baked action clip box is this env's square.

    The fm adapter clips with representation.action_low/high frozen at training
    time; the rig guards with ActionSquare.from_config on the eval env config.
    The two agree today, and they diverge with nothing to flag it the day a
    checkpoint is evaluated against a different env profile, which judges the
    policy in a box it was not trained for. Duck-typed so scripted test
    policies (no representation) pass through; float32 tolerance because the
    checkpoint stores float32.
    """

    representation = getattr(policy, "representation", None)
    low = getattr(representation, "action_low", None)
    high = getattr(representation, "action_high", None)
    if low is None or high is None:
        return
    expected_low = np.array([action_square.x_min, action_square.y_min])
    expected_high = np.array([action_square.x_max, action_square.y_max])
    if not (
        np.allclose(low, expected_low, atol=1e-6) and np.allclose(high, expected_high, atol=1e-6)
    ):
        raise ValueError(
            f"checkpoint clip box low={np.asarray(low)} high={np.asarray(high)} does not "
            f"match the eval action square low={expected_low} high={expected_high} -- "
            "this checkpoint was trained against a different env profile"
        )


def _check_commitment(policy, source) -> None:
    """Raise when the executor did not adopt the policy's declared cadence.

    A warm-start checkpoint anchors its forecast cache on being re-queried every
    H_e steps; executed at any other cadence it re-anchors a stale tail, so
    every chunk after the first is wrong while still looking like ordinary
    motion. set_policy already rejects a disagreement, so this checks that the
    value actually took, which is what a rig-construction change can break
    without producing any error.
    """

    declared = getattr(policy, "execution_horizon", None)
    if declared is None:
        return
    adopted = source.commit_horizon
    if adopted != int(declared):
        raise ValueError(
            f"policy declares execution_horizon={declared} but the executor committed "
            f"{adopted}; the policy's forecast cache would be mis-timed."
        )


def _episode_init(cfg, env_seed: int, index: int):
    from pusht_drake.sim.spawn import sample_task_init

    rng = np.random.default_rng(np.random.SeedSequence([env_seed, index]))
    return sample_task_init(cfg, rng)


def _failed_record(
    index: int,
    env_seed: int,
    action_seed: int,
    error: Exception,
    *,
    backend: str = "drake",
    engine_variant: str = "",
) -> dict[str, Any]:
    """A structurally complete record for an episode the simulator killed.

    The batch continues (one unstable episode must not destroy 49 finished
    ones), the failure is printed to the log, and it scores as a failure
    (success False, coverage 0) with the exception preserved for diagnosis.
    """

    return {
        "episode": index,
        "env_seed": env_seed,
        "action_seed": action_seed,
        "length": 0,
        "num_replans": 0,
        "max_coverage": 0.0,
        "final_coverage": 0.0,
        "max_reward": 0.0,
        "final_reward": 0.0,
        "success": False,
        "terminated": False,
        "truncated": False,
        "tolerance_success": False,
        "final_trans_err_m": float("nan"),
        "final_rot_err_rad": float("nan"),
        "clip_rate": 0.0,
        "boundary_jump_mean_m": 0.0,
        "boundary_jump_max_m": 0.0,
        "within_chunk_step_mean_m": 0.0,
        "tracking_err_mean_m": 0.0,
        "tracking_err_max_m": 0.0,
        "smoothness_m": 0.0,
        "spline_clip_rate": 0.0,
        "square_tick_rate": 0.0,
        "fence_tick_rate": 0.0,
        "leash_tick_rate": 0.0,
        "settle_time_s": 0.0,
        "wall_time_s": 0.0,
        "realtime_factor": 0.0,
        "termination_reason": "sim_error",
        "error": f"{type(error).__name__}: {error}",
        # Both keys are mandatory. aggregate_records rejects records from
        # different simulators, and a record with no backend key reads as
        # "drake", so one sim_error carrying the wrong backend makes a
        # single-simulator run look mixed and raise. A missing cold_fallbacks
        # deflates that metric with no error anywhere.
        "backend": backend,
        "engine_variant": engine_variant,
        "cold_fallbacks": 0,
    }


def _run_episodes(
    env_config: str,
    spec: PolicySpec,
    indices: list[int],
    env_seed: int,
    action_seed: int,
    limits: RolloutLimits,
    execution_mode: str = "bspline",
) -> list[dict[str, Any]]:
    """Worker body: one rig, one policy, a stripe of episode indices."""

    import os

    # Pin the math-library thread pools before the policy factory imports torch
    # (spawn gives a fresh interpreter, so this is early enough; in the
    # in-process workers=1 path it runs before spec.build() too). Without it,
    # every worker's torch/OpenMP claims all cores and busy-spins between the
    # tiny 10 Hz inference calls: measured 0.67x realtime for one worker and
    # 0.11x each for four, because the spin-wait starves Drake's own solver.
    # setdefault, so an explicit user override survives.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    from pusht_drake.sim.rig import build_rig
    from pusht_drake.sim.rollout import rollout_episode

    # Policy first: an unloadable checkpoint should fail in seconds, not
    # after a Drake plant build.
    policy = spec.build()
    rig = build_rig(env_config, execution_mode=execution_mode)
    _check_clip_box(policy, rig.action_square)
    rig.source.set_policy(policy)
    _check_commitment(policy, rig.source)
    records = []
    for position, index in enumerate(indices):
        init = _episode_init(rig.cfg, env_seed, index)
        episode_error = None
        try:
            result = rollout_episode(
                rig,
                policy,
                init,
                env_seed=env_seed,
                action_seed=action_seed + index,
                limits=limits,
            )
            record = {"episode": index, **result.to_dict()}
        except Exception as error:  # noqa: BLE001 - record the failure, continue
            print(f"  episode {index:3d}: SIM ERROR -- {error}", flush=True)
            # Carry the rig's engine_variant onto the failure record: without
            # it, a failed episode is indistinguishable on disk from one
            # produced by a different engine build.
            record = _failed_record(
                index,
                env_seed,
                action_seed + index,
                error,
                engine_variant=getattr(rig, "engine_variant", ""),
            )
            episode_error = error
        # Replaying one interesting seed needs the exact initial state in the
        # artifact, not just the seed that produced it.
        record["init_slider_pose"] = list(init.slider_pose)
        record["init_pusher_xy"] = list(init.pusher_xy)
        records.append(record)
        print(
            f"  episode {index:3d}: max_cov {record['max_coverage']:.3f}  "
            f"len {record['length']:3d}  {record['termination_reason']}",
            flush=True,
        )
        if episode_error is not None:
            # A raise mid-AdvanceTo leaves the Simulator in an undefined state
            # (interrupted event dispatch); one blown-up episode must not
            # contaminate the rest of the stripe. Rebuild costs seconds.
            try:
                rig = build_rig(env_config, execution_mode=execution_mode)
            except Exception as rebuild_error:  # noqa: BLE001 - see below
                # If the rebuild itself fails there is no rig left to run on,
                # and an exception here would escape _run_episodes entirely: it
                # is not a BrokenProcessPool, so evaluate_policy's handler does
                # not catch it and the whole invocation dies with no JSON. Score
                # the rest of this stripe as sim errors instead, so the batch
                # still lands.
                print(
                    f"  rig rebuild failed after episode {index}: "
                    f"{type(rebuild_error).__name__}: {rebuild_error} -- "
                    f"failing the remaining {len(indices) - position - 1} of this stripe",
                    flush=True,
                )
                for remaining in indices[position + 1 :]:
                    records.append(
                        _failed_record(
                            remaining,
                            env_seed,
                            action_seed + remaining,
                            rebuild_error,
                            engine_variant=getattr(rig, "engine_variant", ""),
                        )
                    )
                return records
    return records


def evaluate_policy(
    env_config: str | Path,
    spec: PolicySpec,
    *,
    episodes: int | list[int],
    workers: int = 1,
    env_seed: int = 1000,
    action_seed: int = 1000,
    limits: RolloutLimits | None = None,
    execution_mode: str = "bspline",
) -> list[dict[str, Any]]:
    """Run seeded episodes; returns per-episode records in index order.

    episodes is either a count (indices 0..n-1) or an explicit list of indices,
    so one interesting seed can be replayed by passing its index alone.
    """

    indices = list(range(episodes)) if isinstance(episodes, int) else sorted(set(episodes))
    if not indices:
        raise ValueError("no episode indices to run")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    limits = limits or RolloutLimits()
    env_config = str(Path(env_config).resolve())

    if workers == 1:
        records = _run_episodes(
            env_config,
            spec,
            indices,
            env_seed,
            action_seed,
            limits,
            execution_mode,
        )
    else:
        stripes = [indices[w::workers] for w in range(workers)]
        stripes = [s for s in stripes if s]
        # spawn, not fork: Drake and torch both hold state a forked child must
        # never inherit. ProcessPoolExecutor, not multiprocessing.Pool: when a
        # worker process dies outright (segfault, OOM kill, os._exit, as
        # opposed to a Python exception, which is fault-tolerated per episode
        # inside the worker), Pool.starmap waits forever on a result slot its
        # respawned replacement will never fill; observed live, a 6-episode
        # render hung 48 minutes until an outer timeout killed it. The executor
        # notices the death and raises instead.
        with ProcessPoolExecutor(max_workers=len(stripes), mp_context=get_context("spawn")) as pool:
            futures = [
                pool.submit(
                    _run_episodes,
                    env_config,
                    spec,
                    s,
                    env_seed,
                    action_seed,
                    limits,
                    execution_mode,
                )
                for s in stripes
            ]
            try:
                chunks = [future.result() for future in futures]
            except BrokenProcessPool as error:
                raise RuntimeError(
                    "a sim worker process died mid-batch (segfault/OOM-class "
                    "death, not a Python error); this invocation's records are "
                    "unrecoverable -- rerun the evaluation"
                ) from error
        records = [record for chunk in chunks for record in chunk]

    records.sort(key=lambda r: r["episode"])
    return records
