"""The conditional probability path and its time parameter.

Flow-time convention for the whole package, stated once and asserted by the
tests:

    tau = 0  is the source (noise) distribution
    tau = 1  is the demonstrated action distribution
    inference integrates with positive dt, from 0 to 1

The flow time is always called tau. Some other code bases use the opposite
convention (OpenPI's PyTorch pi0 puts noise at t=1 and actions at t=0), so a
formula or sampler borrowed from them needs tau = 1 - t rather than a straight
copy of its constants.

Independent of models, solvers, sources, and the Push-T task.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NamedTuple, TypeAlias

import torch
from torch import nn

from pusht_drake.fm.schedules import flow_interval, validate_schedules


class PathSample(NamedTuple):
    """A point on the conditional path together with the velocity that defines it.

    Both fields come from one path definition, so a caller cannot pair an
    interpolation from one path with a target velocity from another.
    """

    x_t: torch.Tensor  # (B, H, A)
    velocity: torch.Tensor  # (B, H, A)


def broadcast_tau(tau: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Reshape a flow time to broadcast against a (B, H, A) chunk.

    Accepts the synchronized (B, 1) clock and the per-position (B, H, 1)
    profile, and nothing else. A (B, H) profile missing its trailing axis is
    the usual mistake: it would broadcast against the action axis instead of
    the horizon, giving every coordinate a different time with no error, so
    this raises instead.
    """

    batch_size = like.shape[0]
    if tau.shape == (batch_size, 1):
        return tau.reshape((batch_size,) + (1,) * (like.ndim - 1))
    if like.ndim == 3 and tau.shape == (batch_size, like.shape[1], 1):
        return tau
    raise ValueError(
        f"Expected tau shape {(batch_size, 1)}"
        + (f" or {(batch_size, like.shape[1], 1)}" if like.ndim == 3 else "")
        + f", got {tuple(tau.shape)}. A (B, H) profile missing its trailing axis "
        "is the usual cause."
    )


@dataclass(frozen=True)
class LinearPath:
    """The straight-line conditional path, x_tau = (1 - tau) * x_0 + tau * x_1.

    Its conditional velocity is constant along the path, which is why the target
    is simply x_1 - x_0.
    """

    def sample(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        tau: torch.Tensor,
    ) -> PathSample:
        """Evaluate the path and its target velocity at the given flow time(s).

        source and target are (B, H, A) in normalized learning coordinates.
        tau is either

        (B, 1)
            one time per chunk, broadcast across H and A, so every action token
            sits at the same point on the path. This is the synchronized clock
            every restart arm trains under.
        (B, H, 1)
            one time per horizon position, broadcast across A only. Under
            horizon-coupled training a position's flow time is scheduled by its
            time-to-execution, so the tokens of one chunk sit at different
            points on their own paths.

        The conditional velocity is target - source either way: the path is
        straight, so its velocity does not depend on where along it you are, and
        heterogeneous times change only x_t.
        """

        if source.shape != target.shape:
            raise ValueError(
                "Source and target must have identical shapes, got "
                f"{tuple(source.shape)} and {tuple(target.shape)}."
            )
        if source.ndim < 2:
            raise ValueError("Source and target must include a batch dimension.")
        expanded_tau = broadcast_tau(tau, source)
        return PathSample(
            x_t=(1.0 - expanded_tau) * source + expanded_tau * target,
            velocity=target - source,
        )


TimeSampling: TypeAlias = Literal["uniform"]
TimeSampler: TypeAlias = Callable[..., torch.Tensor]


def sample_uniform_tau(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw one independent uniform flow time per chunk.

    This is the default and the objective every other sampler must match.
    Returns (B, 1) times in [0, 1). Consumes the ambient RNG unless a
    generator is supplied.
    """

    return torch.rand((batch_size, 1), device=device, dtype=dtype, generator=generator)


class HorizonTimeProfile(nn.Module):
    """Per-position training flow times, scheduled by time-to-execution.

    The horizon-coupled arm's replacement for the synchronized clock. Position
    k is trained on the interval [tau_in_k, tau_out_k] that a replan actually
    hands it (pusht_drake.fm.schedules), so training samples the same intervals
    the receding horizon produces rather than a uniform time the rollout never
    visits.

    One xi ~ U(0, 1) is drawn per batch row and shared by the whole chunk. xi
    is a progress coordinate through one joint update, not a flow time: one
    replan advances every position together, so their local times move together
    even though the intervals, and therefore the local times themselves, differ
    across positions. Drawing an independent time per position would train a
    state no replan can produce.

    xi is drawn at (B, 1, 1), consuming exactly the B values the scalar
    samplers consume, so a coupled arm and a restart arm walk identical RNG
    streams and stay comparable on identical randomness.

    Registers only non-persistent buffers, so it contributes no state_dict
    keys: this object is attached to a policy whose state_dict must keep
    exactly the velocity model's keys.
    """

    def __init__(
        self,
        prediction_horizon: int,
        execution_horizon: int,
        *,
        alpha: float,
    ) -> None:
        super().__init__()
        validate_schedules(prediction_horizon, execution_horizon, alpha)
        self.prediction_horizon = prediction_horizon
        self.execution_horizon = execution_horizon
        self.alpha = float(alpha)
        tau_in, tau_out = flow_interval(prediction_horizon, execution_horizon, alpha)
        # Both halves of "non-persistent buffer" matter here. Buffers so the
        # tensors are materialized at construction and moved by .to(device)
        # along with the policy: built lazily on first use they are instead
        # created inside the compiled region, become cudagraph-managed, and the
        # next step reads one that has already been overwritten. Non-persistent
        # so they contribute no state_dict keys and every existing checkpoint
        # still loads under a strict load_state_dict.
        self.register_buffer(
            "tau_in",
            torch.as_tensor(tau_in, dtype=torch.float32).reshape(1, -1, 1),
            persistent=False,
        )
        self.register_buffer(
            "tau_out",
            torch.as_tensor(tau_out, dtype=torch.float32).reshape(1, -1, 1),
            persistent=False,
        )

    def bounds(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(tau_in, tau_out) as (1, H, 1) tensors."""

        del device, dtype  # kept for call-site compatibility; the buffers carry both
        return self.tau_in, self.tau_out

    def map(self, xi: torch.Tensor) -> torch.Tensor:
        """Map a (B, 1) progress coordinate onto (B, H, 1) flow times.

        The draw happens elsewhere, in the same TIME_SAMPLERS["uniform"] call
        every other arm uses, so a coupled arm and a restart arm consume
        bitwise-identical randomness (same epsilon, same xi, same order) and an
        arm comparison is paired rather than merely same-seeded.
        """

        if xi.ndim != 2 or xi.shape[1] != 1:
            raise ValueError(f"Expected xi shape (B, 1), got {tuple(xi.shape)}.")
        return self.tau_in + xi.reshape(-1, 1, 1) * (self.tau_out - self.tau_in)

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        return self.map(xi)


# Explicit membership, so a typo at a boundary raises KeyError.
TIME_SAMPLERS: dict[str, TimeSampler] = {
    "uniform": sample_uniform_tau,
}
