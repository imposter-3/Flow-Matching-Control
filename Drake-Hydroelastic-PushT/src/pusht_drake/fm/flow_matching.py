"""Conditional flow matching for action-chunk generation.

Flow times follow the convention documented in pusht_drake.fm.path: tau=0 is
the source, tau=1 is the demonstrated action. Two invariants live here and
must not drift:

- The loss sums the squared error over the chunk's (H, A) coordinates and
  takes the mean over the batch only. An element mean differs from it by a
  constant at fixed H, A, and that constant (16 at H=8, A=2) is baked into
  every learning rate in the recipe.
- RNG order in training: the source epsilon is drawn first, the flow time
  second, both from the ambient stream. Inference uses explicit per-episode
  Generators instead.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from pusht_drake.fm.path import TIME_SAMPLERS, HorizonTimeProfile, LinearPath, TimeSampling
from pusht_drake.fm.solvers import (
    SOLVER_EVALUATIONS,
    SOLVERS,
    Solver,
    VelocityCorrection,
)
from pusht_drake.fm.sources import ActionSource, VanillaSource, WarmContext


class LossOutput(NamedTuple):
    """Scalar training loss plus detached-friendly diagnostic components."""

    total: torch.Tensor
    source_distance: torch.Tensor
    source_std: torch.Tensor


class FlowMatchingTerms(NamedTuple):
    """The objective's per-element terms, before any reduction is chosen.

    Training reduces these to scalars; diagnostics keep the element structure.
    Both read the same tensors, so a diagnostic cannot drift from the objective.
    """

    squared_error: torch.Tensor  # (B, H, A)
    target_velocity: torch.Tensor  # (B, H, A)
    source_std: torch.Tensor  # (B, H, A)


class FlowMatchingPolicy(nn.Module):
    """Generate action chunks by integrating a learned conditional velocity field."""

    def __init__(
        self,
        velocity_model: nn.Module,
        action_dim: int,
        chunk_size: int,
        # Measured: 5 is an interior optimum for the full policy. The baseline
        # recipe runs the textbook 10 instead (pusht_drake.fm.recipe passes it
        # explicitly), so this default only records the measured value.
        num_integration_steps: int = 5,
        source: ActionSource | None = None,
        solver: Solver = "euler",
        time_sampling: TimeSampling = "uniform",
        time_profile: HorizonTimeProfile | None = None,
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}.")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        if num_integration_steps <= 0:
            raise ValueError(
                f"num_integration_steps must be positive, got {num_integration_steps}."
            )
        if solver not in SOLVER_EVALUATIONS:
            raise ValueError(f"Unsupported solver {solver!r}.")
        if time_sampling not in TIME_SAMPLERS:
            raise ValueError(f"Unsupported time_sampling {time_sampling!r}.")
        self.velocity_model = velocity_model
        self.num_integration_steps = num_integration_steps
        self.solver = solver
        self.time_sampling = time_sampling
        self.path = LinearPath()
        self.source = source if source is not None else VanillaSource()
        # When set, training draws a per-position flow time from the horizon
        # schedule instead of the synchronized clock. A plain attribute, not a
        # module or buffer: the policy's state_dict must keep exactly the
        # velocity model's keys, and the profile holds nothing learnable.
        # torch.compile specializes on it once, since it never changes for a
        # given policy instance.
        if time_profile is not None and time_profile.prediction_horizon != chunk_size:
            raise ValueError(
                f"time_profile is built for H={time_profile.prediction_horizon} but the "
                f"policy has chunk_size={chunk_size}."
            )
        self.time_profile = time_profile

    @property
    def num_function_evaluations(self) -> int:
        """Velocity evaluations per generated chunk; the unit inference cost."""

        return self.num_integration_steps * SOLVER_EVALUATIONS[self.solver]

    def flow_matching_terms(
        self,
        condition: torch.Tensor,
        action_chunk: torch.Tensor,
        *,
        source_context: WarmContext | None = None,
        source_noise: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        history: torch.Tensor | None = None,
    ) -> FlowMatchingTerms:
        """Evaluate the conditional flow-matching objective before any reduction.

        The package's only implementation of the training formula.
        compute_losses reduces it to scalars; diagnostics pin the noise and
        the time and keep the per-element error.

        RNG: with both source_noise and tau supplied, no RNG is
        consumed. Otherwise the source is drawn first and the time second.
        """

        expected_shape = (condition.shape[0], self.chunk_size, self.action_dim)
        if action_chunk.shape != expected_shape:
            raise ValueError(
                f"Expected action chunk shape {expected_shape}, got {tuple(action_chunk.shape)}."
            )

        # The target chunk is already expressed in normalized learning coordinates.
        source = self.source.sample(
            condition, expected_shape, context=source_context, noise=source_noise
        )
        if tau is None:
            # A single draw site serves every arm. A coupled policy maps the
            # same (B, 1) uniform draw onto its per-position intervals rather
            # than drawing its own, so every arm walks an identical RNG stream.
            tau = TIME_SAMPLERS[self.time_sampling](
                condition.shape[0],
                device=action_chunk.device,
                dtype=action_chunk.dtype,
            )
            if self.time_profile is not None:
                tau = self.time_profile.map(tau)
        path = self.path.sample(source.value, action_chunk, tau)
        # The network predicts local velocity, not the target action endpoint.
        predicted_velocity = self.velocity_model(condition, path.x_t, tau, history)
        if predicted_velocity.shape != expected_shape:
            raise ValueError(
                f"Velocity model returned {tuple(predicted_velocity.shape)}; "
                f"expected {expected_shape}."
            )

        return FlowMatchingTerms(
            squared_error=(predicted_velocity - path.velocity).pow(2),
            target_velocity=path.velocity,
            source_std=source.std,
        )

    def compute_losses(
        self,
        condition: torch.Tensor,
        action_chunk: torch.Tensor,
        source_context: WarmContext | None = None,
        history: torch.Tensor | None = None,
    ) -> LossOutput:
        """Reduce the flow-matching terms to the scalars the training loop logs.

        Sum over (H, A), mean over batch; see the module docstring for why
        this is never an element mean.
        """

        terms = self.flow_matching_terms(
            condition, action_chunk, source_context=source_context, history=history
        )
        flow_matching = terms.squared_error.flatten(start_dim=1).sum(dim=1).mean()
        # Diagnostics stay attached here: detaching inside a compiled region
        # makes inductor return every output as non-differentiable. The caller
        # detaches instead.
        return LossOutput(
            total=flow_matching,
            source_distance=(terms.target_velocity.pow(2).flatten(start_dim=1).sum(dim=1).mean()),
            source_std=terms.source_std.mean(),
        )

    @torch.no_grad()
    def sample_actions(
        self,
        condition: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        source: torch.Tensor | None = None,
        context: WarmContext | None = None,
        noise: torch.Tensor | None = None,
        history: torch.Tensor | None = None,
        velocity_correction: VelocityCorrection | None = None,
    ) -> torch.Tensor:
        """Draw a source sample and integrate it into an action chunk.

        Returns the tau=1 chunk, (B, H, A), in normalized agent-relative
        coordinates. Decoding to absolute world metres belongs to the rollout
        adapter, not here.
        """

        batch_size = condition.shape[0]
        chunk_shape = (batch_size, self.chunk_size, self.action_dim)
        if source is not None:
            if source.shape != chunk_shape:
                raise ValueError(
                    f"Explicit source must have shape {chunk_shape}, got {tuple(source.shape)}."
                )
            initial_state = source.to(device=condition.device, dtype=condition.dtype)
        else:
            # A stochastic start makes generation stochastic for a fixed condition.
            initial_state = self.source.sample(
                condition,
                chunk_shape,
                generator=generator,
                context=context,
                noise=noise,
            ).value

        return self._integrate(
            initial_state,
            condition,
            history=history,
            velocity_correction=velocity_correction,
        )

    def _integrate(
        self,
        initial_state: torch.Tensor,
        condition: torch.Tensor,
        history: torch.Tensor | None = None,
        velocity_correction: VelocityCorrection | None = None,
    ) -> torch.Tensor:
        """Integrate the field from tau=0 to tau=1. Consumes no randomness.

        Kept eager, never torch.compiled: measured, inductor's reassociation
        drifted the integrated chunk by 2.3e-06, enough to make rollouts
        incomparable across compile settings. Only the loss may be compiled.
        """

        # The condition is fixed for the whole integration, so it is closed over
        # rather than threaded through the solver.
        if velocity_correction is None:

            def velocity_fn(x_t: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
                return self.velocity_model(condition, x_t, tau, history)

        else:

            def velocity_fn(x_t: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
                velocity = self.velocity_model(condition, x_t, tau, history)
                return velocity_correction(x_t, tau, velocity)

        return SOLVERS[self.solver](
            velocity_fn, initial_state, num_steps=self.num_integration_steps
        )
