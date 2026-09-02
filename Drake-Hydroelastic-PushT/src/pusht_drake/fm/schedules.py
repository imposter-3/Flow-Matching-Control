"""Horizon schedules for the forecast-weighted and flow-coupled sources.

Fixed numpy functions of (H, H_e, alpha), materialized once and reused. Under
receding-horizon control the horizon position k is time-to-execution in policy
steps: entry 0 is applied immediately, the first H_e entries execute before the
next replan, and larger k are forecasts a later replan may overwrite. Flow
times follow pusht_drake.fm.path: tau=0 is the source, tau=1 the demonstration.

lambda_k, the forecast weight. A squared exponential in normalized
time-to-execution over the positions that can have a temporally aligned
predecessor, structurally zero on the last H_e::

    lambda_k = exp[-alpha * ((k+1) / (H - H_e))^2]   for 0 <= k < H - H_e
    lambda_k = 0                                      otherwise

Those trailing zeros are not a limit of the kernel: the last H_e positions
entered the horizon at this replan, so no previous forecast covers them.

tau'_k, the scheduled target flow time. Saturated over the execution window,
then a plain exponential::

    tau'_k = 1                                        for 0 <= k < H_e
    tau'_k = exp[-alpha * (k - H_e + 1) / (H - H_e)]  for H_e <= k < H

The first H_e entries are exactly one because those actions execute before the
next replan and must be fully generated. A distant position stops short at
tau'_k < 1, so no terminal iterate ever exists for it, which is why the
persistent rollout estimates its endpoint from the velocity rather than
integrating to tau=1.

tau_in_k, where an overlapping position resumes, is the target of the position
it occupied one replan earlier; the zero extension (lambda_k = tau'_k = 0 for
k >= H) makes tau_in_k = tau'_{k+H_e} hold with no separate tail case.

The two normalizations are coordinated by construction, lambda_{H-H_e-1} =
tau'_{H-1} = exp(-alpha), and validate_schedules asserts that wherever a source
or a rollout policy is built.
"""

from __future__ import annotations

import numpy as np


def forecast_weight(
    chunk_size: int,
    execution_horizon: int,
    alpha: float,
) -> np.ndarray:
    """lambda_k, shape (H,), float64.

    Returned at full horizon width, structural zeros included, so callers
    index it by absolute horizon position and never have to remember where the
    reusable region ends.
    """

    _validate_horizons(chunk_size, execution_horizon)
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}.")

    overlap = chunk_size - execution_horizon
    weights = np.zeros(chunk_size, dtype=np.float64)
    positions = np.arange(overlap, dtype=np.float64)
    normalized = (positions + 1.0) / float(overlap)
    weights[:overlap] = np.exp(-alpha * normalized * normalized)
    return weights


def target_flow_time(
    chunk_size: int,
    execution_horizon: int,
    alpha: float,
) -> np.ndarray:
    """tau'_k, shape (H,), float64.

    The scheduled target flow time (how far generation should have run for the
    action at position k), not the flow time it currently holds.
    """

    _validate_horizons(chunk_size, execution_horizon)
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}.")

    overlap = chunk_size - execution_horizon
    targets = np.ones(chunk_size, dtype=np.float64)
    tail = np.arange(execution_horizon, chunk_size, dtype=np.float64)
    exponent = (tail - execution_horizon + 1.0) / float(overlap)
    targets[execution_horizon:] = np.exp(-alpha * exponent)
    return targets


def flow_interval(
    chunk_size: int,
    execution_horizon: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(tau_in, tau_out), each shape (H,), float64.

    tau_out is the position's own target. tau_in is the target of the position
    it occupied one replan earlier, obtained by shifting tau_out left by H_e
    and padding with the zero extension, which is what a tail position needs
    since it has no predecessor and begins at flow time zero.

    Training samples over these intervals and the receding horizon produces the
    same ones at rollout; coupling the flow to the horizon is what makes them
    the same object.
    """

    targets = target_flow_time(chunk_size, execution_horizon, alpha)
    extended = np.concatenate([targets, np.zeros(execution_horizon, dtype=np.float64)])
    tau_in = extended[execution_horizon : execution_horizon + chunk_size]
    return np.ascontiguousarray(tau_in), targets


def warm_mask(chunk_size: int, execution_horizon: int, warm_count: int) -> np.ndarray:
    """Binary WarmPrior mask 1[k < warm_count], shape (H,), float64.

    The baseline against which the continuous weight is judged. warm_count may
    not exceed the reusable region: the last H_e positions have no temporally
    aligned predecessor, so warming them is undefined rather than a stronger
    baseline.
    """

    _validate_horizons(chunk_size, execution_horizon)
    overlap = chunk_size - execution_horizon
    if not 0 <= warm_count <= overlap:
        raise ValueError(
            f"warm_count must lie in [0, {overlap}] for H={chunk_size}, "
            f"H_e={execution_horizon}; got {warm_count}. Positions beyond "
            f"{overlap} entered the horizon at this replan and have no "
            "previous forecast to reuse."
        )
    mask = np.zeros(chunk_size, dtype=np.float64)
    mask[:warm_count] = 1.0
    return mask


def validate_schedules(
    chunk_size: int,
    execution_horizon: int,
    alpha: float,
) -> None:
    """Assert the structural properties both schedules are supposed to have.

    Called wherever a forecast-weighted source or a persistent rollout policy
    is built. Each check is a property the method's derivation depends on, so a
    violation means the code and the derivation have drifted apart.
    """

    overlap = chunk_size - execution_horizon
    weights = forecast_weight(chunk_size, execution_horizon, alpha)
    targets = target_flow_time(chunk_size, execution_horizon, alpha)
    tau_in, tau_out = flow_interval(chunk_size, execution_horizon, alpha)
    endpoint = float(np.exp(-alpha))

    # Tail positions carry no forecast.
    if not np.all(weights[overlap:] == 0.0):
        raise AssertionError(
            f"lambda must be structurally zero on the last H_e={execution_horizon} "
            f"positions; got {weights[overlap:]}."
        )
    # The reusable region is positive and strictly decreasing.
    if not np.all(weights[:overlap] > 0.0):
        raise AssertionError("lambda must be positive on the reusable region.")
    if not np.all(np.diff(weights[:overlap]) < 0.0):
        raise AssertionError("lambda must strictly decrease in time-to-execution.")
    # Executed actions are fully generated.
    if not np.all(targets[:execution_horizon] == 1.0):
        raise AssertionError(
            f"tau' must be exactly 1 over the execution window; got {targets[:execution_horizon]}."
        )
    # Beyond it, positive and strictly decreasing.
    if not np.all(targets[execution_horizon:] > 0.0):
        raise AssertionError("tau' must stay positive beyond the execution window.")
    if not np.all(np.diff(targets[execution_horizon:]) < 0.0):
        raise AssertionError("tau' must strictly decrease beyond the execution window.")
    # The coordination between the two normalizations.
    if not np.isclose(weights[overlap - 1], endpoint, rtol=0.0, atol=1e-12):
        raise AssertionError(
            f"lambda_{overlap - 1} should equal exp(-alpha)={endpoint!r}, got "
            f"{weights[overlap - 1]!r}."
        )
    if not np.isclose(targets[-1], endpoint, rtol=0.0, atol=1e-12):
        raise AssertionError(
            f"tau'_{chunk_size - 1} should equal exp(-alpha)={endpoint!r}, got {targets[-1]!r}."
        )
    # A replan may never ask a position to run backwards in flow time.
    if not np.all(tau_out - tau_in >= 0.0):
        raise AssertionError(
            "tau_out must be >= tau_in at every position; a negative interval "
            "would integrate the flow backwards across a replan."
        )
    # Tail positions start cold.
    if not np.all(tau_in[overlap:] == 0.0):
        raise AssertionError(f"tau_in must be zero on the last H_e={execution_horizon} positions.")


def _validate_horizons(chunk_size: int, execution_horizon: int) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    if not 0 < execution_horizon < chunk_size:
        raise ValueError(
            "execution_horizon must satisfy 0 < H_e < H, got "
            f"H_e={execution_horizon}, H={chunk_size}."
        )
