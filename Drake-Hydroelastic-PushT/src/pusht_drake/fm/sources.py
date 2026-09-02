"""Source distributions for the flow's tau=0 state. Task-free.

VanillaSource is the open-loop baseline's source and the control that says
what a richer source bought. WarmPreviewSource and ForecastWeightSource are
the receding-horizon sources, defined only where H_e < H_p.

A source constructs X_0. It never sees the environment, the replay buffer,
or a preview cache: whatever it needs about the previous replan arrives as an
explicit WarmContext built by the caller, because the choice of which preview
to train on is a training-policy decision rather than a property of the
distribution.

Flow times follow the convention in pusht_drake.fm.path: tau=0 is the source.
This module imports nothing else from the package.
"""

from __future__ import annotations

import abc
from typing import NamedTuple

import torch
from torch import nn

from pusht_drake.fm.schedules import forecast_weight, validate_schedules


class WarmContext(NamedTuple):
    """A source mean for the leading chunk positions plus its per-sample validity."""

    mean: torch.Tensor  # (B, context_length, A), normalized learning coordinates
    valid: torch.Tensor  # (B, 1), 1.0 when a previous chunk existed


class SourceSample(NamedTuple):
    """One draw from a source distribution together with its parameters."""

    value: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


class ActionSource(nn.Module, metaclass=abc.ABCMeta):
    """Construct the flow source X_0; it never sees environment or cache state.

    Two class attributes declare the context contract, so callers ask the source
    what it needs instead of probing it with getattr:

    requires_context
        The source raises when asked to sample without a WarmContext. A
        missing context is a programmer error, not a fallback to a cold
        Gaussian.
    context_length
        How many leading chunk positions that context describes; zero when the
        source is stateless.
    """

    requires_context: bool = False
    context_length: int = 0

    @abc.abstractmethod
    def sample(
        self,
        condition: torch.Tensor,
        shape: tuple[int, ...],
        *,
        generator: torch.Generator | None = None,
        context: WarmContext | None = None,
        noise: torch.Tensor | None = None,
    ) -> SourceSample:
        """Return a source sample shaped like the action chunk.

        noise replaces the drawn epsilon. Supplying it is what makes two
        source configurations comparable on identical randomness.
        """

    @staticmethod
    def _epsilon(
        condition: torch.Tensor,
        shape: tuple[int, ...],
        generator: torch.Generator | None,
        noise: torch.Tensor | None,
    ) -> torch.Tensor:
        if noise is None:
            return torch.randn(
                shape,
                device=condition.device,
                dtype=condition.dtype,
                generator=generator,
            )
        if tuple(noise.shape) != tuple(shape):
            raise ValueError(
                f"Explicit noise must have shape {tuple(shape)}, got {tuple(noise.shape)}."
            )
        return noise.to(device=condition.device, dtype=condition.dtype)


def cold_context(
    source: ActionSource,
    batch_size: int,
    *,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> WarmContext | None:
    """Build the context for samples that have no previous chunk to reuse.

    None for a stateless source, otherwise a present but invalid context
    (zero mean, zero validity), which is what a first replan and a cold
    diagnostic arm both are. Passing None to a source that requires a
    context is a programmer error, not a cold start.
    """

    if not source.requires_context:
        return None
    return WarmContext(
        mean=torch.zeros(
            (batch_size, source.context_length, action_dim),
            device=device,
            dtype=dtype,
        ),
        valid=torch.zeros((batch_size, 1), device=device, dtype=dtype),
    )


class VanillaSource(ActionSource):
    """The observation-independent standard Gaussian source.

    The simplest X_0: the baseline recipe's source, and the control that says
    what any richer source bought.
    """

    def sample(
        self,
        condition: torch.Tensor,
        shape: tuple[int, ...],
        *,
        generator: torch.Generator | None = None,
        context: WarmContext | None = None,
        noise: torch.Tensor | None = None,
    ) -> SourceSample:
        del context
        value = self._epsilon(condition, shape, generator, noise)
        return SourceSample(
            value=value,
            mean=torch.zeros_like(value),
            std=torch.ones_like(value),
        )


class WarmPreviewSource(ActionSource):
    """Reuse the unexecuted forecast as the next source mean (WarmPrior).

    Defined only where the policy plans farther than it executes (H_e < H_p).
    Unused by the baseline recipe; kept as the adoption path.

    A caller must always supply a context. The first replan of an episode
    passes an explicit invalid one (valid=0) rather than None, which reduces
    this source exactly to a standard Gaussian.

    Warm depth. context_length defaults to the whole overlap H_p - H_e, which
    is correct at H8/E4, where the overlap coincides with the executed band.
    It is a parameter because the two stop coinciding as H_p - H_e grows.
    Measured at H_p=15/H_e=2: warming all 13 overlap positions is clearly
    worse than warming the first 5 (success -0.216, 3/3 seeds). The mechanism
    is that a deep anchor pins a plan segment revised ~47% of its own
    magnitude per replan, and revised ~6 more times before it ever executes.
    A shorter context is the same code path with a smaller warm.

    sigma is an uncalibrated dose, not a tuned one. sigma=1.0 leaves the warm
    positions carrying full unit noise, so only the mean moves; a sweep found
    both 0.5 and 1.5 worse. The dose relative to the data still depends on the
    horizon: at H_p=16 the source is far wider than the data at k=0 and
    narrower past k~8. That is recorded rather than corrected, because a
    deterministic source would make a wrong preview uncorrectable, which is
    also why sigma=0 was never tested.
    """

    requires_context = True

    def __init__(
        self,
        prediction_horizon: int,
        execution_horizon: int,
        *,
        sigma: float = 1.0,
        context_length: int | None = None,
    ) -> None:
        super().__init__()
        if not 1 <= execution_horizon < prediction_horizon:
            raise ValueError(
                "WarmPreviewSource needs 1 <= H_e < H_p, got "
                f"{execution_horizon} and {prediction_horizon}."
            )
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}.")
        overlap = prediction_horizon - execution_horizon
        context_length = overlap if context_length is None else int(context_length)
        if not 1 <= context_length <= overlap:
            raise ValueError(
                f"context_length must be in [1, {overlap}] (the overlap H_p - H_e), "
                f"got {context_length}."
            )
        self.prediction_horizon = prediction_horizon
        self.execution_horizon = execution_horizon
        self.context_length = context_length
        self.sigma = sigma

    def _validate_context(
        self,
        context: WarmContext | None,
        shape: tuple[int, ...],
    ) -> WarmContext:
        if context is None:
            raise ValueError(
                "WarmPreviewSource requires an explicit WarmContext. Pass "
                "cold_context(...) for a first replan; None is a programmer error, "
                "not a cold start."
            )
        batch_size, _, action_dim = shape
        expected_mean = (batch_size, self.context_length, action_dim)
        if tuple(context.mean.shape) != expected_mean:
            raise ValueError(
                f"Context mean must have shape {expected_mean}, got {tuple(context.mean.shape)}."
            )
        if tuple(context.valid.shape) != (batch_size, 1):
            raise ValueError(
                f"Context validity must have shape {(batch_size, 1)}, got "
                f"{tuple(context.valid.shape)}."
            )
        return context

    def sample(
        self,
        condition: torch.Tensor,
        shape: tuple[int, ...],
        *,
        generator: torch.Generator | None = None,
        context: WarmContext | None = None,
        noise: torch.Tensor | None = None,
    ) -> SourceSample:
        context = self._validate_context(context, shape)
        epsilon = self._epsilon(condition, shape, generator, noise)
        warm = self.context_length
        mean = torch.zeros_like(epsilon)
        std = torch.ones_like(epsilon)
        # An invalid row keeps mean 0 and std 1, so it is exactly the cold source.
        valid = context.valid.reshape(-1, 1, 1).to(epsilon.dtype)
        mean = torch.cat((valid * context.mean, mean[:, warm:]), dim=1)
        std = torch.cat(
            (
                torch.full_like(std[:, :warm], self.sigma) * valid + std[:, :warm] * (1.0 - valid),
                std[:, warm:],
            ),
            dim=1,
        )
        return SourceSample(value=mean + std * epsilon, mean=mean, std=std)


class ForecastWeightSource(ActionSource):
    """Continuous reuse of the previous forecast as the source mean.

    X_0 = lambda_k * a^1 + eps, with lambda_k the squared-exponential horizon
    schedule of pusht_drake.fm.schedules. Serves both the restart arm
    (Forecast Weight) and the persistent one (Fully Coupled); they share this
    source exactly and differ only in the flow-time schedule and in whether
    the partially generated state survives a replan.

    The residual stays at unit scale and the mean is what moves. That is the
    difference from WarmPreviewSource, which modulates the standard deviation
    instead, and it is what makes lambda_k = 0 reduce exactly to the
    context-free source rather than to a degenerate point mass. The last H_e
    positions have lambda_k = 0 structurally, having entered the horizon at
    this replan with no previous forecast covering them, so they are pure eps
    by construction rather than through a separate code path.

    context_length is the whole overlap and is not a parameter. Unlike the
    binary mask, the continuous weight already decides how much each position
    reuses; truncating the context would impose a second, uncoordinated cutoff
    on top of a schedule whose job is to taper.

    A caller must always supply a context. The first replan of an episode
    passes an explicit invalid one (valid=0) rather than None, which reduces
    this source bitwise to a standard Gaussian.
    """

    requires_context = True

    def __init__(
        self,
        prediction_horizon: int,
        execution_horizon: int,
        *,
        alpha: float,
    ) -> None:
        super().__init__()
        if not 1 <= execution_horizon < prediction_horizon:
            raise ValueError(
                "ForecastWeightSource needs 1 <= H_e < H_p, got "
                f"{execution_horizon} and {prediction_horizon}."
            )
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}.")
        validate_schedules(prediction_horizon, execution_horizon, alpha)
        self.prediction_horizon = prediction_horizon
        self.execution_horizon = execution_horizon
        self.alpha = float(alpha)
        overlap = prediction_horizon - execution_horizon
        self.context_length = overlap
        # Non-persistent so the source still contributes zero state_dict keys:
        # the checkpoint's source_type metadata is the only record of which
        # source trained the weights, and a persistent buffer here would break
        # a strict load of every existing checkpoint. A buffer rather than a
        # lazy cache because a tensor first materialized inside the compiled
        # region becomes cudagraph-managed, and the next step then reads one
        # that has already been overwritten.
        self.register_buffer(
            "forecast_lambda",
            torch.as_tensor(
                forecast_weight(prediction_horizon, execution_horizon, alpha)[:overlap],
                dtype=torch.float32,
            ).reshape(1, -1, 1),
            persistent=False,
        )

    def _validate_context(
        self,
        context: WarmContext | None,
        shape: tuple[int, ...],
    ) -> WarmContext:
        if context is None:
            raise ValueError(
                "ForecastWeightSource requires an explicit WarmContext. Pass "
                "cold_context(...) for a first replan; None is a programmer error, "
                "not a cold start."
            )
        batch_size, _, action_dim = shape
        expected_mean = (batch_size, self.context_length, action_dim)
        if tuple(context.mean.shape) != expected_mean:
            raise ValueError(
                f"Context mean must have shape {expected_mean}, got {tuple(context.mean.shape)}."
            )
        if tuple(context.valid.shape) != (batch_size, 1):
            raise ValueError(
                f"Context validity must have shape {(batch_size, 1)}, got "
                f"{tuple(context.valid.shape)}."
            )
        return context

    def sample(
        self,
        condition: torch.Tensor,
        shape: tuple[int, ...],
        *,
        generator: torch.Generator | None = None,
        context: WarmContext | None = None,
        noise: torch.Tensor | None = None,
    ) -> SourceSample:
        context = self._validate_context(context, shape)
        epsilon = self._epsilon(condition, shape, generator, noise)
        overlap = self.context_length
        valid = context.valid.reshape(-1, 1, 1).to(epsilon.dtype)
        weighted = (
            valid
            * self.forecast_lambda.to(dtype=epsilon.dtype)
            * context.mean.to(device=epsilon.device, dtype=epsilon.dtype)
        )
        # The trailing H_e positions carry lambda = 0 structurally, so they are
        # concatenated as exact zeros rather than multiplied by a zero weight.
        mean = torch.cat((weighted, torch.zeros_like(epsilon[:, overlap:])), dim=1)
        std = torch.ones_like(epsilon)
        return SourceSample(value=mean + std * epsilon, mean=mean, std=std)
