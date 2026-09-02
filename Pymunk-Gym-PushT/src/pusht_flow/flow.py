"""Conditional flow matching: the source, the flow time, and the regression target.

Conventions the rest of the package depends on, and that are easy to break
without an error:

Direction. tau = 0 is noise, tau = 1 is data, dt is positive. The straight
conditional path is a_tau = (1-tau) * a0 + tau * a1, so the sample-wise target
velocity is the constant a1 - a0.

Reduction. The loss sums over the chunk's (H, A) coordinates and takes the mean
over the batch only. An elementwise mean would divide the gradient by
H * A = 32, which is equivalent to a 32x smaller learning rate.

Draw order. The source residual is drawn before the flow time, from the same
stream. Any reordering changes every subsequent sample, so two arms that should
differ only in their source would also differ in their noise.

The three source constructions differ in exactly one place, what supplies the
mean of a0:

    gaussian           a0 = eps
    warmprior_binary   a0 = a1 + sigma * eps  on warm positions, eps elsewhere
    forecast_weight    a0 = lambda_k * a1 + eps

The last one keeps the residual at unit scale and modulates the mean, which is
what makes lambda_k = 0 reduce exactly to the context-free source rather than
to a degenerate point mass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pusht_flow.config import MethodConfig
from pusht_flow.schedules import (
    flow_interval,
    forecast_weight,
    validate_schedules,
    warm_mask,
)


def _as_profile(values: np.ndarray, device) -> torch.Tensor:
    """A (H,) float64 schedule as a (1, H, 1) float32 tensor on the device."""

    return (
        torch.from_numpy(np.ascontiguousarray(values))
        .float()
        .reshape(1, -1, 1)
        .to(device)
    )


@dataclass(frozen=True)
class HorizonProfiles:
    """The per-position schedules a method needs, materialized once as tensors.

    Held as (1, H, 1) so every use broadcasts against a (B, H, A) chunk or a
    (B, H, 1) flow-time profile without a reshape at the call site.
    """

    forecast_lambda: torch.Tensor | None
    warm: torch.Tensor | None
    tau_in: torch.Tensor
    tau_out: torch.Tensor

    @classmethod
    def build(
        cls,
        method: MethodConfig,
        *,
        chunk_size: int,
        execution_horizon: int,
        device: torch.device | str = "cpu",
    ) -> HorizonProfiles:
        validate_schedules(chunk_size, execution_horizon, method.alpha)
        lambdas = None
        warm = None
        if method.source_mode == "forecast_weight":
            lambdas = _as_profile(
                forecast_weight(chunk_size, execution_horizon, method.alpha), device
            )
        elif method.source_mode == "warmprior_binary":
            warm = _as_profile(
                warm_mask(chunk_size, execution_horizon, method.warm_count), device
            )
        tau_in, tau_out = flow_interval(chunk_size, execution_horizon, method.alpha)
        return cls(
            forecast_lambda=lambdas,
            warm=warm,
            tau_in=_as_profile(tau_in, device),
            tau_out=_as_profile(tau_out, device),
        )


def build_training_source(
    data_chunk: torch.Tensor,
    residual: torch.Tensor,
    method: MethodConfig,
    profiles: HorizonProfiles,
) -> torch.Tensor:
    """a0 for one training batch, shape (B, H, A).

    At training time the data action stands in for the temporal forecast: no
    forecast exists during training, since it would be the model's own earlier
    output, and the data action is the natural proxy for it. It is never an
    input at deployment; rollout.py substitutes the previous replan's endpoint
    estimate into exactly the same expression.
    """

    if method.source_mode == "gaussian":
        return residual
    if method.source_mode == "warmprior_binary":
        if profiles.warm is None:
            raise ValueError(f"{method.key}: a binary warm source needs a warm mask.")
        warm_source = data_chunk + method.warmprior_sigma * residual
        return profiles.warm * warm_source + (1.0 - profiles.warm) * residual
    if method.source_mode == "forecast_weight":
        if profiles.forecast_lambda is None:
            raise ValueError(f"{method.key}: a weighted source needs lambda_k.")
        return profiles.forecast_lambda * data_chunk + residual
    raise ValueError(f"Unknown source_mode {method.source_mode!r}.")


def sample_training_flow_times(
    batch_size: int,
    method: MethodConfig,
    profiles: HorizonProfiles,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Flow times for one training batch.

    A restart method sees the ordinary synchronized clock, one scalar per
    chunk, shape (B, 1).

    A persistent method sees the profile the receding horizon actually
    produces. Every position must end at its own scheduled target and begins at
    the target of the position it occupied one replan earlier, so it is trained
    over the interval from tau_in_k to tau_out_k. A single xi drawn from U(0,1)
    is shared by the whole chunk and traverses all those intervals together,
    because that is what one joint update does: xi is a progress coordinate
    through one update, not a flow time. The resulting local times generally
    differ across positions, since the intervals do.
    """

    if method.flow_mode == "restart":
        return torch.rand(batch_size, 1, device=device, generator=generator)

    xi = torch.rand(batch_size, 1, 1, device=device, generator=generator)
    return profiles.tau_in + xi * (profiles.tau_out - profiles.tau_in)


def flow_matching_loss(
    model,
    condition: torch.Tensor,
    data_chunk: torch.Tensor,
    source_chunk: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Mean over the batch of the squared error, summed over the chunk.

    tau may be (B, 1) or (B, H, 1); it is broadcast to the chunk's shape so a
    single expression covers both regimes.
    """

    tau_path = tau if tau.dim() == 3 else tau.reshape(-1, 1, 1)
    noisy = (1.0 - tau_path) * source_chunk + tau_path * data_chunk
    target = data_chunk - source_chunk
    predicted = model(condition, noisy, tau)
    squared = (predicted - target).pow(2)
    # Sum over (H, A), mean over the batch. See the module docstring.
    return squared.sum(dim=(1, 2)).mean()


def training_step_tensors(
    data_chunk: torch.Tensor,
    method: MethodConfig,
    profiles: HorizonProfiles,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(source_chunk, tau) for one batch, drawing in the fixed order.

    The residual comes first and the flow time second. Keeping that order in
    one function is what stops it from drifting apart between arms.
    """

    residual = torch.randn(
        data_chunk.shape,
        device=data_chunk.device,
        dtype=data_chunk.dtype,
        generator=generator,
    )
    source_chunk = build_training_source(data_chunk, residual, method, profiles)
    tau = sample_training_flow_times(
        data_chunk.shape[0],
        method,
        profiles,
        device=data_chunk.device,
        generator=generator,
    )
    return source_chunk, tau
