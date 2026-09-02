"""A trained checkpoint bound as the numpy Policy the rollout consumes.

The evaluator (pusht_drake.sim) must not import torch, so this module is
where the two sides meet. The adapter owns the tensors, the generator and the
representation, and exposes only the numpy Protocol the rollout calls.

Each replan encodes the single observation frame, samples one chunk with the
per-episode generator, decodes once against the replan-time pusher position,
clips in absolute coordinates, and counts what the clip changed.

Warm cache. Under receding horizon the previous forecast's unexecuted tail
becomes the next flow's source mean, under two rules:

- The cache holds decoded, clipped, absolute world metres, and the tail is
  re-encoded against the new replan-time pusher position. Translating the
  normalized tensor instead raises no error and steers wrong: normalized
  coordinates are relative to an anchor that has moved, so a shifted tensor
  is a plan aimed at where the pusher used to be.
- The first replan of an episode gets an explicitly invalid context, never a
  missing one. valid=0 reduces a warm source bitwise to the vanilla one;
  absent metadata instead scores a warm source as a cold one.

The cache holds what the policy predicted, not what the executor ultimately
commanded: the Drake guard chain (square, fence, leash) sits on the far side
of the layering boundary and the policy cannot see it. Re-anchoring against
the measured pusher each replan absorbs the accumulated part of that
difference. Caching post-clip suffices only where the clip is the sole
modification, which holds in the sibling Pymunk-Gym-PushT study but not here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pusht_drake.fm.checkpoint import CheckpointPayload, build_policy, load_payload
from pusht_drake.fm.schedules import flow_interval, forecast_weight
from pusht_drake.fm.solvers import coupled_euler_integrate
from pusht_drake.fm.sources import WarmContext, cold_context
from pusht_drake.observation import OBSERVATION_DIM, agent_position

#: A replan whose anchor moved further than this since the last committed
#: waypoint did not follow that plan at all (a reset, a teleport), so its
#: forecast tail describes a trajectory that never happened. Coarse on
#: purpose: it catches teleports, not off-by-one shifts. Consecutive waypoint
#: spacings fall below the arm's own tracking error often enough that no
#: geometric test can catch a mis-timed shift.
TELEPORT_TOLERANCE_M = 0.02


class FMRolloutPolicy:
    """Bind a trained flow-matching checkpoint for closed-loop control.

    Satisfies the rollout Policy protocol (sim.interface). The recurrent
    rollout state (the action generator, the clip counters) lives here rather
    than on the network, so it never enters state_dict and cannot leak between
    episodes.
    """

    def __init__(
        self,
        payload: CheckpointPayload,
        device: str | torch.device = "cpu",
        num_integration_steps: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.payload = payload
        self.representation = payload.representation
        self.policy = build_policy(payload, self.device)
        if num_integration_steps is not None:
            # An inference-only setting: NFE never enters the training
            # weights, and the eval artifact reports the value actually used.
            if num_integration_steps < 1:
                raise ValueError(f"num_integration_steps must be >= 1, got {num_integration_steps}")
            self.policy.num_integration_steps = int(num_integration_steps)
        self.prediction_horizon = payload.representation.prediction_horizon
        #: Read by the Drake executor (duck-typed): the replan cadence, in
        #: waypoints.
        self.execution_horizon = payload.horizons.execution_horizon
        # Read off the source, not off model.warm_context_length. That field is
        # guarded to warm_preview specs and is None for a forecast-weight one,
        # whose context length is derived (the whole overlap), so reading the
        # metadata made every forecast-weight episode die in _warm_context with
        # "unsupported operand type(s) for +: 'int' and 'NoneType'". The source
        # declares context_length so callers never have to reconstruct it from
        # whichever metadata field happens to carry it.
        self.context_length = self.policy.source.context_length
        self._warm = self.policy.source.requires_context
        self._generator: torch.Generator | None = None
        self._previous_chunk: np.ndarray | None = None
        self._committed: int | None = None
        self.clipped_coordinates = 0
        self.predicted_coordinates = 0
        self.cold_fallbacks = 0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        num_integration_steps: int | None = None,
    ) -> FMRolloutPolicy:
        return cls(load_payload(Path(checkpoint_path)), device, num_integration_steps)

    def reset(self, seed: int) -> None:
        """Start an episode with a fresh, explicitly seeded sampling stream.

        Off the global RNG, so an episode is a pure function of (checkpoint,
        env seed, action seed).
        """

        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        self._generator = generator
        # Drop the forecast cache. This object outlives episodes (the harness
        # builds one per worker stripe and loops), so without this the second
        # episode of a stripe would warm-start from the last plan of the
        # first one.
        self._previous_chunk = None
        self._committed = None
        self.clipped_coordinates = 0
        self.predicted_coordinates = 0
        self.cold_fallbacks = 0

    def notify_committed(self, n: int) -> None:
        """The executor reports how many waypoints it actually committed.

        Pushed rather than inferred, because nothing about the resulting
        motion distinguishes a correct shift from an off-by-one.
        """

        self._committed = int(n)

    def invalidate_warm_cache(self) -> None:
        """The executing plan was discarded out of band; the cache describes it."""

        self._previous_chunk = None
        self._committed = None

    def _warm_context(self, anchor: np.ndarray) -> WarmContext | None:
        """The previous forecast's tail, re-anchored, or an invalid context."""

        if not self._warm:
            return None
        cold = cold_context(
            self.policy.source,
            1,
            action_dim=self.representation.action_dim,
            device=self.device,
            dtype=torch.float32,
        )
        shift = self._committed
        if self._previous_chunk is None or shift is None:
            self.cold_fallbacks += 1
            return cold
        tail = self._previous_chunk[shift : shift + self.context_length]
        if len(tail) < self.context_length:
            self.cold_fallbacks += 1
            return cold
        if np.linalg.norm(anchor - self._previous_chunk[shift - 1]) > TELEPORT_TOLERANCE_M:
            # The pusher is nowhere near where that plan would have put it, so
            # the plan is not a preview of anything that happened.
            self.cold_fallbacks += 1
            return cold
        mean = self.representation.encode_action(tail, anchor)
        return WarmContext(
            mean=torch.from_numpy(mean).float().to(self.device).unsqueeze(0),
            valid=torch.ones((1, 1), device=self.device, dtype=torch.float32),
        )

    def predict_chunk(self, observation: np.ndarray) -> np.ndarray:
        """One (H, 2) chunk of absolute pusher targets, world meters, clipped."""

        if self._generator is None:
            raise RuntimeError("reset(seed) must be called before predicting a chunk.")
        observation = np.asarray(observation, dtype=np.float32).reshape(OBSERVATION_DIM)
        condition = self.representation.encode_observation(observation)
        condition_tensor = torch.from_numpy(condition).float().to(self.device).unsqueeze(0)
        anchor = agent_position(observation)
        context = self._warm_context(anchor)
        with torch.inference_mode():
            normalized = self.policy.sample_actions(
                condition_tensor, context=context, generator=self._generator
            )
        # Decode once against the replan-time pusher, then clip in absolute
        # coordinates; the clipped chunk is what executes and what is reported.
        raw_chunk = self.representation.decode_action(
            normalized.detach().float().cpu().numpy()[0], anchor
        )
        clipped = self.representation.clip_action(raw_chunk)
        self.clipped_coordinates += int(np.count_nonzero(clipped != raw_chunk))
        self.predicted_coordinates += clipped.size
        # Cache the clipped chunk: the next warm start must describe the plan
        # that was actually issued. Consumed once; a second call without an
        # intervening notify_committed falls back to cold rather than reusing
        # a stale shift.
        self._previous_chunk = clipped.astype(np.float32)
        self._committed = None
        return clipped.astype(np.float64)

    @property
    def clip_rate(self) -> float:
        if self.predicted_coordinates == 0:
            return 0.0
        return self.clipped_coordinates / self.predicted_coordinates


class CoupledRolloutPolicy:
    """Rollout policy whose partially generated chunk survives the replan.

    Satisfies the rollout Policy protocol (sim.interface). FMRolloutPolicy
    restarts the flow from a fresh source every replan and reuses only the
    previous forecast; this class reuses the previous computation. Each
    horizon position carries its flow state and the local flow time it
    reached, and a replan resumes that state instead of starting over.

    Three repairs run before every warm integration, in this order:

    1. Reframe. Actions are relative to the replan-time pusher, so when the
       horizon slides the coordinate frame moves with the agent. Substituting
       the source into the path shows the state carries the source mean with
       weight (1-tau)*lambda and the data endpoint with weight tau, and
       both move, so the state moves by their combination::

           a_k  <-  a_k - [tau_in + (1 - tau_in) * lambda_old] * delta
           delta = (p_new - p_old) / action_std

       lambda_old is the weight the state was built under, kept explicit
       rather than re-derived from the new position. This term is exact for
       the straight path, and it is invisible to any probe that holds the
       agent still: such a probe multiplies the whole correction by zero and
       passes whether the term is present or absent.
    2. Re-anchor. Both factors of the source mean moved across the replan:
       the weight (from lambda_{k+H_e} to lambda_k) and the forecast it
       multiplies. At fixed flow time the state is affine in the source mean, so
       it moves by (1-tau) times the change in the mean itself.
    3. Refresh the tail. The last H_e positions entered the horizon at
       this replan and have no predecessor, so they get fresh noise at tau=0.

    previous_agent is the measured pusher from the observation, never the
    last commanded waypoint: the executor guards every anchor (square, fence,
    leash) and the arm tracks with error, so commanded and measured differ and
    the frame displacement would be wrong by the tracking error every replan.

    The shift is reported by the executor rather than assumed. tau_in_k is
    defined as tau'_{k+H_e}, so a commitment of anything other than H_e
    mis-times every carried flow time without raising. notify_committed
    carries the count that was actually committed; a missing or disagreeing
    count restarts the flow cold rather than guessing.

    torch.no_grad rather than torch.inference_mode: this class caches a
    tensor across calls, and inference tensors carry restricted semantics
    outside the block that made them.
    """

    def __init__(
        self,
        payload: CheckpointPayload,
        device: str | torch.device = "cpu",
        num_integration_steps: int | None = None,
    ) -> None:
        if payload.model.flow_mode != "persistent":
            raise ValueError(
                f"CoupledRolloutPolicy needs a persistent checkpoint, got "
                f"source_type={payload.model.source_type!r}."
            )
        self.device = torch.device(device)
        self.payload = payload
        self.representation = payload.representation
        self.policy = build_policy(payload, self.device)
        self.prediction_horizon = payload.representation.prediction_horizon
        self.execution_horizon = payload.horizons.execution_horizon
        self.overlap = self.prediction_horizon - self.execution_horizon
        self.num_integration_steps = payload.horizons.num_integration_steps
        if num_integration_steps is not None:
            if num_integration_steps < 1:
                raise ValueError(f"num_integration_steps must be >= 1, got {num_integration_steps}")
            self.num_integration_steps = int(num_integration_steps)
            # Mirrored so an artifact reporting policy.num_function_evaluations
            # quotes the value in force. The integration loop below reads this
            # class's field: this path never calls the policy's own _integrate.
            self.policy.num_integration_steps = self.num_integration_steps

        alpha = payload.model.alpha
        self._lambda = forecast_weight(self.prediction_horizon, self.execution_horizon, alpha)
        tau_in, tau_out = flow_interval(self.prediction_horizon, self.execution_horizon, alpha)
        self._tau_in, self._tau_out = tau_in, tau_out
        self._action_std = np.asarray(self.representation.action_std, dtype=np.float32)

        self._generator: torch.Generator | None = None
        self.clipped_coordinates = 0
        self.predicted_coordinates = 0
        self.cold_fallbacks = 0
        self._committed: int | None = None
        self._drop_carried_state()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        num_integration_steps: int | None = None,
    ) -> CoupledRolloutPolicy:
        return cls(load_payload(Path(checkpoint_path)), device, num_integration_steps)

    def _drop_carried_state(self) -> None:
        self._state: torch.Tensor | None = None
        self._tau: np.ndarray | None = None
        self._forecast_absolute: np.ndarray | None = None
        self._anchor_absolute: np.ndarray | None = None
        self._anchor_lambda: np.ndarray | None = None
        self._previous_agent: np.ndarray | None = None
        self._last_chunk: np.ndarray | None = None

    def reset(self, seed: int) -> None:
        """Start an episode: a fresh, explicitly seeded action-sampling stream."""

        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        self._generator = generator
        # This object outlives episodes (the harness builds one per worker
        # stripe and loops), so the carried flow is dropped here; otherwise
        # episode two would resume episode one's plan.
        self._drop_carried_state()
        self._committed = None
        self.clipped_coordinates = 0
        self.predicted_coordinates = 0
        self.cold_fallbacks = 0

    def notify_committed(self, n: int) -> None:
        self._committed = int(n)

    def invalidate_warm_cache(self) -> None:
        """The executing plan was discarded out of band; the carried flow describes it."""

        self._drop_carried_state()
        self._committed = None

    def _column(self, values: np.ndarray) -> torch.Tensor:
        return (
            torch.from_numpy(np.ascontiguousarray(values.reshape(1, -1, 1))).float().to(self.device)
        )

    def _encode(self, absolute: np.ndarray, anchor: np.ndarray) -> torch.Tensor:
        encoded = self.representation.encode_action(absolute, anchor)
        return torch.from_numpy(encoded).float().to(self.device).unsqueeze(0)

    def _decode_and_clip(self, normalized: torch.Tensor, anchor: np.ndarray) -> np.ndarray:
        raw = self.representation.decode_action(
            normalized.detach().float().cpu().numpy()[0], anchor
        )
        clipped = self.representation.clip_action(raw)
        self.clipped_coordinates += int(np.count_nonzero(clipped != raw))
        self.predicted_coordinates += clipped.size
        return clipped

    def _resumable(self, anchor: np.ndarray) -> bool:
        """Whether the carried flow describes the plan that actually executed."""

        if self._state is None or self._last_chunk is None:
            return False
        shift = self._committed
        if shift is None or shift != self.execution_horizon:
            # Either nothing was committed, or the executor committed a width the
            # schedules were not built for. Both mis-time every carried tau.
            return False
        if np.linalg.norm(anchor - self._last_chunk[shift - 1]) > TELEPORT_TOLERANCE_M:
            # The pusher is nowhere near where that plan would have put it, so the
            # plan previews nothing that happened.
            return False
        return True

    def predict_chunk(self, observation: np.ndarray) -> np.ndarray:
        """One (H, 2) chunk of absolute pusher targets, world meters, clipped."""

        if self._generator is None:
            raise RuntimeError("reset(seed) must be called before predicting a chunk.")
        observation = np.asarray(observation, dtype=np.float32).reshape(OBSERVATION_DIM)
        anchor = agent_position(observation)
        condition = (
            torch.from_numpy(self.representation.encode_observation(observation))
            .float()
            .to(self.device)
            .unsqueeze(0)
        )
        horizon, action_dim = self.prediction_horizon, self.representation.action_dim

        with torch.no_grad():
            # The full chunk of noise is drawn unconditionally, cold branch or
            # warm, so the coupled arm's RNG stream stays aligned with every
            # other arm's. Drawing only the tail when warm desynchronizes the
            # action stream at the first replan and breaks pairing with no
            # error raised.
            noise = torch.randn(
                (1, horizon, action_dim),
                device=self.device,
                dtype=torch.float32,
                generator=self._generator,
            )
            lambdas = self._lambda

            if not self._resumable(anchor):
                self.cold_fallbacks += 1
                state = noise
                tau_start = np.zeros(horizon, dtype=np.float64)
                # Built from the context-free source, so it carries no
                # forecast weight. Setting this to lambda would make the next
                # replan subtract a mean the state never carried: wrong
                # exactly once per episode.
                anchor_lambda = np.zeros(horizon, dtype=np.float64)
                anchor_absolute = None
            else:
                shift = self.execution_horizon
                # The action at position k + H_e now occupies position k.
                state = torch.zeros_like(self._state)
                state[:, : self.overlap] = self._state[:, shift:]
                tau_start = np.zeros(horizon, dtype=np.float64)
                tau_start[: self.overlap] = self._tau[shift:]
                forecast = np.zeros((horizon, action_dim), dtype=np.float32)
                forecast[: self.overlap] = self._forecast_absolute[shift:]
                carried_anchor = np.zeros((horizon, action_dim), dtype=np.float32)
                carried_anchor[: self.overlap] = self._anchor_absolute[shift:]
                anchor_lambda = np.zeros(horizon, dtype=np.float64)
                anchor_lambda[: self.overlap] = self._anchor_lambda[shift:]

                tau_column = self._column(tau_start)
                # 1. Reframe for the agent that moved under the plan.
                delta = (anchor - self._previous_agent) / self._action_std
                weight = tau_column + (1.0 - tau_column) * self._column(anchor_lambda)
                displacement = (
                    torch.from_numpy(np.asarray(delta, dtype=np.float32))
                    .to(self.device)
                    .reshape(1, 1, -1)
                )
                state = state - weight * displacement
                # 2. Re-anchor: both the weight and the forecast it multiplies moved.
                state = state + (1.0 - tau_column) * (
                    self._column(lambdas) * self._encode(forecast, anchor)
                    - self._column(anchor_lambda) * self._encode(carried_anchor, anchor)
                )
                # 3. The trailing H_e entered the horizon now: fresh noise at tau=0.
                state[:, self.overlap :] = noise[:, self.overlap :]
                tau_start[self.overlap :] = 0.0
                anchor_absolute = forecast
                anchor_lambda = lambdas.copy()

            result = coupled_euler_integrate(
                lambda x_t, tau: self.policy.velocity_model(condition, x_t, tau),
                state,
                self._column(tau_start),
                self._column(self._tau_out),
                num_steps=self.num_integration_steps,
            )
            # Endpoint from the partial state, reusing the velocity the last
            # integration step already produced, so no extra network call. A
            # distant position is scheduled to stop short of tau=1, so no
            # terminal iterate exists for it and this is the only endpoint
            # available.
            endpoint = result.state + (1.0 - result.tau) * result.velocity

        forecast_absolute = self._decode_and_clip(endpoint, anchor)

        self._state = result.state
        self._tau = result.tau.detach().cpu().numpy().reshape(-1).astype(np.float64)
        self._forecast_absolute = forecast_absolute
        # The anchor must be the exact absolute waypoints the mean was built
        # from, and _decode_and_clip clips, so the clipped array is what gets
        # cached. Otherwise the state and its declared mean sit apart by the
        # clip amount near the action-square boundary.
        self._anchor_absolute = forecast_absolute if anchor_absolute is None else anchor_absolute
        self._anchor_lambda = anchor_lambda
        self._previous_agent = anchor.astype(np.float32).copy()
        self._last_chunk = forecast_absolute
        self._committed = None
        return forecast_absolute.astype(np.float64)

    @property
    def clip_rate(self) -> float:
        if self.predicted_coordinates == 0:
            return 0.0
        return self.clipped_coordinates / self.predicted_coordinates


def build_rollout_policy(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    num_integration_steps: int | None = None,
) -> FMRolloutPolicy | CoupledRolloutPolicy:
    """Bind a checkpoint to the rollout class its trained source requires.

    The dispatch reads the payload rather than a caller's argument: the source
    carries no state_dict keys, so a coupled checkpoint driven by the restart
    sampler loads cleanly and produces plausible, wrong actions.
    """

    payload = load_payload(Path(checkpoint_path))
    cls = CoupledRolloutPolicy if payload.model.flow_mode == "persistent" else FMRolloutPolicy
    return cls(payload, device, num_integration_steps)
