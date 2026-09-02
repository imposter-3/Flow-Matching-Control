"""The two inference paths: restart and persistent.

Three of the four methods restart the flow at every replan and differ only in
how they build a0, so they share one path. The fourth carries the partially
generated state across replans and needs its own.

RestartPolicy
    Build a0, integrate 0 -> 1 in nfe Euler steps, execute the first H_e
    actions, discard everything, repeat.

PersistentPolicy
    Refine the current chunk to each position's scheduled target, execute the
    first H_e, shift the survivors, re-express them in the new frame, re-anchor
    their source mean, and continue over the new intervals.

The frame correction
--------------------

Actions in this package are relative to the replan-time agent position (see
data.py). So when the horizon slides, the coordinate system itself moves, and a
cached state expressed in the old frame is not a valid state in the new one.
Writing p for the agent position and s for the action scale, the frame
displacement in normalized coordinates is

    delta = (p_new - p_old) / s

An endpoint forecast is a point in action space, so it moves by -delta. A
partial state does not: substituting the source into the straight path shows it
carries the source mean with weight (1-tau) * lambda and the data endpoint with
weight tau, and both move, so the state moves by their combination:

    a_k  <-  a_k - (tau_in + (1 - tau_in) * lambda_old) * delta

with lambda_old the weight the state was actually built under. This is exact
for the straight-path parameterization.

A derivation done in a fixed frame has delta = 0 and the term vanishes, which
is precisely why it is easy to omit: any probe that holds the agent still
multiplies the missing term by zero. checks.py therefore tests it with a moving
agent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pusht_flow.config import MethodConfig, Recipe, agent_position
from pusht_flow.data import Normalizer
from pusht_flow.flow import HorizonProfiles


def reframe_weight(tau: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """tau + (1 - tau) * lambda: how far a partial state moves with the frame.

    A state at flow time tau built on a source mean of weight lambda carries
    the mean with weight (1-tau) * lambda and the data endpoint with weight
    tau. Both are points in action space, so both move by the frame
    displacement, and the state moves by their sum. At tau = 1 the weight is 1
    (a fully generated action is just a point); at tau = 0 with lambda = 0 it
    is 0 (pure noise does not live in the action frame).
    """

    return tau + (1.0 - tau) * lam


def reframe_state(
    state: torch.Tensor,
    tau: np.ndarray,
    lam: np.ndarray,
    delta: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Re-express a carried state in the current action frame."""

    weight = (
        torch.from_numpy(reframe_weight(tau, lam).reshape(1, -1, 1)).float().to(device)
    )
    displacement = torch.from_numpy(np.asarray(delta, dtype=np.float32))
    return state - weight * displacement.reshape(1, 1, -1).to(device)


def reanchor_state(
    state: torch.Tensor,
    tau: np.ndarray,
    *,
    new_lambda: np.ndarray,
    new_mean: torch.Tensor,
    old_lambda: np.ndarray,
    old_mean: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Replace the source mean a carried state was built under.

    At fixed flow time, endpoint and residual the state is affine in the source
    mean, so swapping the mean moves the state by (1 - tau) times the change in
    the mean itself. Both factors of that mean change across a replan: the
    weight, and the forecast it multiplies.
    """

    def as_column(values) -> torch.Tensor:
        return torch.from_numpy(
            np.ascontiguousarray(values, dtype=np.float32).reshape(1, -1, 1)
        ).to(device)

    one_minus = as_column(1.0 - tau)
    return state + one_minus * (
        as_column(new_lambda) * new_mean - as_column(old_lambda) * old_mean
    )


@dataclass
class CallCounter:
    """Counts whole-network forward evaluations, so NFE can be asserted.

    NFE = n must mean n evaluations of the network per replan. The endpoint
    forecast reuses the velocity already produced by the last integration step
    and so must add none; this counter is what proves it rather than assuming
    it.
    """

    forwards: int = 0
    replans: int = 0

    def reset(self) -> None:
        self.forwards = 0
        self.replans = 0


class Policy:
    """A trained velocity field plus the inference rule of one method."""

    def __init__(
        self,
        model,
        normalizer: Normalizer,
        method: MethodConfig,
        recipe: Recipe,
        *,
        device: torch.device | str = "cpu",
        counter: CallCounter | None = None,
    ) -> None:
        self.model = model
        self.normalizer = normalizer
        self.method = method
        self.recipe = recipe
        self.device = torch.device(device)
        self.counter = counter or CallCounter()
        self.profiles = HorizonProfiles.build(
            method,
            chunk_size=recipe.chunk_size,
            execution_horizon=recipe.execution_horizon,
            device=self.device,
        )
        # numpy views of the schedules, for the bookkeeping done outside torch.
        self._lambda = (
            self.profiles.forecast_lambda.reshape(-1).cpu().numpy()
            if self.profiles.forecast_lambda is not None
            else None
        )
        self._warm = (
            self.profiles.warm.reshape(-1).cpu().numpy()
            if self.profiles.warm is not None
            else None
        )
        self._tau_in = self.profiles.tau_in.reshape(-1).cpu().numpy()
        self._tau_out = self.profiles.tau_out.reshape(-1).cpu().numpy()

    def reset(self) -> None:
        """Clear per-episode state. Subclasses extend this, never replace it.

        One Policy is driven through 900 consecutive episodes by evaluate.py,
        so any state that survived a reset would show up as plausible metric
        drift, never as an error.
        """

    def _velocity(self, condition, chunk, tau):
        self.counter.forwards += 1
        return self.model(condition, chunk, tau)

    def _condition(self, observation: np.ndarray) -> torch.Tensor:
        features = self.normalizer.encode_observation(observation)
        return torch.from_numpy(features).float().reshape(1, -1).to(self.device)

    def _noise(self, generator: torch.Generator) -> torch.Tensor:
        # Always a full (1, H, A) draw, even when a caller uses only part of
        # it. The generator stream is part of the evaluation protocol: drawing
        # fewer elements here would shift every subsequent sample and change
        # every published number.
        return torch.randn(
            1,
            self.recipe.chunk_size,
            self.normalizer.action_dim,
            device=self.device,
            generator=generator,
        )

    def _decode(self, chunk: torch.Tensor, observation: np.ndarray) -> np.ndarray:
        normalized = chunk.detach().cpu().numpy()[0]
        anchor = agent_position(observation)
        raw = self.normalizer.decode_action(normalized, anchor)
        return self.normalizer.clip_action(raw)


class RestartPolicy(Policy):
    """Every replan integrates a fresh flow from 0 to 1.

    Used by CFM Restart, WarmPrior and Forecast Weight. They differ only in
    build_source, and none of them reads a cached flow state: the previous
    forecast may be reused, the previous computation never is.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: Previous replan's endpoint forecast, stored as absolute waypoints so
        #: it can be re-encoded against whatever anchor the next replan has.
        self.previous_absolute: np.ndarray | None = None

    def reset(self) -> None:
        super().reset()
        self.previous_absolute = None

    def build_source(
        self, observation: np.ndarray, generator: torch.Generator
    ) -> torch.Tensor:
        """a0 at inference: the training source with the forecast substituted."""

        residual = self._noise(generator)
        if not self.method.uses_forecast or self.previous_absolute is None:
            # No usable predecessor: every position keeps the context-free
            # source. This is the first replan of an episode.
            return residual

        # Re-encode the cached forecast against the current anchor. Caching
        # absolute waypoints and re-encoding here is what keeps the relative
        # representation valid across a moving frame.
        overlap = self.previous_absolute[self.recipe.execution_horizon :]
        aligned = np.zeros(
            (self.recipe.chunk_size, self.normalizer.action_dim), dtype=np.float32
        )
        aligned[: len(overlap)] = overlap
        mean = self.normalizer.encode_action(
            aligned[None], agent_position(observation)[None]
        )
        mean_t = torch.from_numpy(mean).float().to(self.device)

        if self.method.source_mode == "warmprior_binary":
            warm = self.profiles.warm
            warm_source = mean_t + self.method.warmprior_sigma * residual
            return warm * warm_source + (1.0 - warm) * residual
        return self.profiles.forecast_lambda * mean_t + residual

    @torch.no_grad()
    def predict(
        self, observation: np.ndarray, *, nfe: int, generator: torch.Generator
    ) -> np.ndarray:
        condition = self._condition(observation)
        chunk = self.build_source(observation, generator)
        step = 1.0 / nfe
        for index in range(nfe):
            tau = torch.full(
                (1, 1), index * step, device=self.device, dtype=chunk.dtype
            )
            chunk = chunk + step * self._velocity(condition, chunk, tau)
        self.counter.replans += 1
        absolute = self._decode(chunk, observation)
        if self.method.uses_forecast:
            self.previous_absolute = absolute
        return absolute


class PersistentPolicy(Policy):
    """The partially generated chunk survives the replan.

    State carried across a replan, per position: the partial flow state, the
    local flow time it reached, the endpoint forecast that anchored its source,
    and the weight that forecast entered under. The last two are what the
    reweighting needs, and keeping the weight explicit is what stops it from
    being re-derived incorrectly from the new position.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.state: torch.Tensor | None = None
        self.tau: np.ndarray | None = None
        #: Most recent endpoint forecast, already shifted, in absolute waypoints.
        #: This is what the next replan re-anchors the source mean onto.
        self.forecast_absolute: np.ndarray | None = None
        #: Endpoint forecast that built the current state, in absolute waypoints.
        self.anchor_absolute: np.ndarray | None = None
        #: The weight that forecast entered the source under, per position.
        self.anchor_lambda: np.ndarray | None = None
        self.previous_agent: np.ndarray | None = None

    def _encode_forecast(
        self, absolute: np.ndarray, observation: np.ndarray
    ) -> torch.Tensor:
        encoded = self.normalizer.encode_action(
            absolute[None], agent_position(observation)[None]
        )
        return torch.from_numpy(encoded).float().to(self.device)

    @torch.no_grad()
    def predict(
        self, observation: np.ndarray, *, nfe: int, generator: torch.Generator
    ) -> np.ndarray:
        condition = self._condition(observation)
        chunk_size = self.recipe.chunk_size
        execution_horizon = self.recipe.execution_horizon
        lambdas = self._lambda
        if lambdas is None:
            raise ValueError(f"{self.method.key}: a persistent policy needs lambda_k.")

        if self.state is None:
            # Cold start: nothing to carry, so every position begins at the
            # context-free source at flow time zero.
            state = self._noise(generator)
            tau_start = np.zeros(chunk_size, dtype=np.float64)
        else:
            state = self.state
            tau_start = self.tau.copy()

            # 1. Re-express the carried state in the current action frame.
            #    See the module docstring: a fixed-frame derivation has no such
            #    term, and in relative coordinates it is mandatory.
            delta = (
                agent_position(observation).astype(np.float32)
                - self.previous_agent.astype(np.float32)
            ) / self.normalizer.action_std
            state = reframe_state(
                state,
                tau_start,
                self.anchor_lambda,
                delta,
                device=self.device,
            )

            # 2. Re-anchor the source mean. Both factors moved: the weight,
            #    from lambda at the old position to lambda at the new one, and
            #    the forecast, from the one that built the state to the freshly
            #    aligned one. At fixed flow time, endpoint and residual the
            #    state is affine in that mean, so the repair is (1 - tau) times
            #    the change in the mean itself.
            state = reanchor_state(
                state,
                tau_start,
                new_lambda=lambdas,
                new_mean=self._encode_forecast(self.forecast_absolute, observation),
                old_lambda=self.anchor_lambda,
                old_mean=self._encode_forecast(self.anchor_absolute, observation),
                device=self.device,
            )

            # 3. Tail positions entered the horizon now: no predecessor, so
            #    they start from the context-free source at flow time zero.
            fresh = self._noise(generator)
            tail = chunk_size - execution_horizon
            state[:, tail:] = fresh[:, tail:]
            tau_start[tail:] = 0.0

            self.anchor_absolute = self.forecast_absolute
            self.anchor_lambda = lambdas.copy()

        # Integrate every position from where it is to its scheduled target,
        # in nfe steps of the shared progress coordinate xi. One network
        # evaluation advances the whole chunk even though positions travel
        # different amounts of local flow time: heterogeneous local times
        # change the step each row takes, not the number of model calls.
        tau_now = tau_start.copy()
        delta_tau = self._tau_out - tau_now
        delta_t = torch.from_numpy(delta_tau.reshape(1, -1, 1)).float().to(self.device)
        step = 1.0 / nfe
        velocity = None
        for _ in range(nfe):
            tau_t = torch.from_numpy(tau_now.reshape(1, -1, 1)).float().to(self.device)
            velocity = self._velocity(condition, state, tau_t)
            state = state + step * delta_t * velocity
            tau_now = tau_now + step * delta_tau
        self.counter.replans += 1

        # Endpoint estimate from the partial state, reusing the velocity the
        # last integration step already produced: no extra network call. A
        # distant position may never be integrated to tau = 1, so no terminal
        # iterate exists for it and this is the only endpoint available.
        one_minus_now = (
            torch.from_numpy((1.0 - tau_now).reshape(1, -1, 1)).float().to(self.device)
        )
        endpoint = state + one_minus_now * velocity
        forecast_absolute = self._decode(endpoint, observation)

        # Shift: the action at position k + H_e now occupies position k.
        shifted_state = torch.zeros_like(state)
        shifted_state[:, : chunk_size - execution_horizon] = state[
            :, execution_horizon:
        ]
        shifted_tau = np.zeros_like(tau_now)
        shifted_tau[: chunk_size - execution_horizon] = tau_now[execution_horizon:]

        shifted_forecast = np.zeros_like(forecast_absolute)
        shifted_forecast[: chunk_size - execution_horizon] = forecast_absolute[
            execution_horizon:
        ]
        shifted_anchor = np.zeros_like(forecast_absolute)
        shifted_lambda = np.zeros_like(lambdas)
        if self.anchor_absolute is None:
            # After a cold start the state was built from the context-free
            # source, so its anchor weight is zero everywhere.
            shifted_anchor[: chunk_size - execution_horizon] = forecast_absolute[
                execution_horizon:
            ]
        else:
            shifted_anchor[: chunk_size - execution_horizon] = self.anchor_absolute[
                execution_horizon:
            ]
            shifted_lambda[: chunk_size - execution_horizon] = self.anchor_lambda[
                execution_horizon:
            ]

        self.state = shifted_state
        self.tau = shifted_tau
        self.forecast_absolute = shifted_forecast
        self.anchor_absolute = shifted_anchor
        self.anchor_lambda = shifted_lambda
        self.previous_agent = agent_position(observation).astype(np.float32).copy()
        return forecast_absolute


def build_policy(
    model,
    normalizer: Normalizer,
    method: MethodConfig,
    recipe: Recipe,
    *,
    device: torch.device | str = "cpu",
    counter: CallCounter | None = None,
) -> Policy:
    """Pick the inference path the method's flow_mode names."""

    cls = PersistentPolicy if method.flow_mode == "persistent" else RestartPolicy
    return cls(model, normalizer, method, recipe, device=device, counter=counter)
