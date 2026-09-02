"""What a saved run records, and how it is read back.

A checkpoint carries the weights together with everything needed to interpret
them: the representation (what the numbers mean, in metres), the method's
algorithmic fields (which source trained the weights), and the control
horizons. The flow source contributes no state_dict keys, so this metadata is
the only record of which source trained the weights; a warm checkpoint would
otherwise load into a vanilla policy without error and produce plausible,
wrong actions. Every field below is therefore required, never defaulted.

Readers are strict: the payload key set is asserted, the stored method must
match config.METHODS under its own key field for field, and a foreign
artifact type raises with a reason. The payload holds primitives and tensors
only, so it loads under torch.load with weights_only=True.

The number of integration steps is not stored: it is an inference budget, not
a property of the weights. Loading injects the package's operating point
(config.NFE) and callers may override it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from pusht_drake.config import FORECAST_SOURCE_TYPES, METHODS, NFE, Method
from pusht_drake.fm.flow_matching import FlowMatchingPolicy
from pusht_drake.fm.models import TransformerVelocityModel
from pusht_drake.fm.path import HorizonTimeProfile
from pusht_drake.fm.recipe import TrainConfig
from pusht_drake.fm.representation import Representation
from pusht_drake.fm.sources import (
    ActionSource,
    ForecastWeightSource,
    VanillaSource,
    WarmPreviewSource,
)

ARTIFACT_TYPE = "pusht_drake_policy"
FORMAT_VERSION = 2

#: The algorithmic identity of a method; the display name stays in config.py.
METHOD_FIELDS = ("key", "source_type", "warm_depth", "alpha", "warm_sigma")

#: Top-level keys every payload this code writes must carry; load_payload
#: asserts the set.
TOP_LEVEL_KEYS = frozenset(
    {
        "artifact_type",
        "format_version",
        "state_dict",
        "representation",
        "method",
        "horizons",
        "step",
        "seed",
    }
)


@dataclass(frozen=True)
class HorizonSpec:
    """The control contract, plus the injected inference budget.

    prediction_horizon is what the policy plans; execution_horizon is what the
    executor commits to before re-querying. Both are properties of the trained
    artifact: a warm or coupled policy anchors its forecast cache on being
    re-queried at exactly this cadence. num_integration_steps is injected at
    load time from config.NFE and never stored.
    """

    prediction_horizon: int
    execution_horizon: int
    num_integration_steps: int = NFE

    def __post_init__(self) -> None:
        if not 1 <= self.execution_horizon <= self.prediction_horizon:
            raise ValueError(
                f"execution_horizon must be in 1..{self.prediction_horizon}, got "
                f"{self.execution_horizon}."
            )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "prediction_horizon": self.prediction_horizon,
            "execution_horizon": self.execution_horizon,
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> HorizonSpec:
        return cls(
            prediction_horizon=int(metadata["prediction_horizon"]),
            execution_horizon=int(metadata["execution_horizon"]),
        )


def method_metadata(method: Method) -> dict[str, Any]:
    return {name: getattr(method, name) for name in METHOD_FIELDS}


def method_from_metadata(metadata: dict[str, Any]) -> Method:
    """Resolve the stored method against config.METHODS, field for field.

    The table labels rows by key, so a payload whose fields disagree with the
    code's definition of its own key would mislabel a result; this raises
    instead of trusting it.
    """

    key = metadata.get("key")
    if key not in METHODS:
        raise ValueError(f"Unknown method {key!r}. Known: {', '.join(METHODS)}.")
    method = METHODS[key]
    for name in METHOD_FIELDS:
        if metadata.get(name) != getattr(method, name):
            raise ValueError(
                f"Stored method.{name}={metadata.get(name)!r} does not match the "
                f"definition of {key!r} ({getattr(method, name)!r}). Refusing to "
                "mislabel a result."
            )
    return method


@dataclass
class CheckpointPayload:
    """A parsed checkpoint; no network is built."""

    step: int
    seed: int
    representation: Representation
    model: Method
    horizons: HorizonSpec
    state_dict: dict[str, Any] = field(repr=False, default_factory=dict)


def build_payload(
    *,
    state_dict: dict[str, Any],
    representation: Representation,
    method: Method,
    horizons: HorizonSpec,
    step: int,
    seed: int,
) -> dict[str, Any]:
    """Assemble the dict that torch.save writes."""

    return {
        "artifact_type": ARTIFACT_TYPE,
        "format_version": FORMAT_VERSION,
        "state_dict": state_dict,
        "representation": representation.to_metadata(),
        "method": method_metadata(method),
        "horizons": horizons.to_metadata(),
        "step": int(step),
        "seed": int(seed),
    }


def parse_payload(raw: dict[str, Any]) -> CheckpointPayload:
    """Validate a raw payload dict and type it. See load_payload."""

    artifact = raw.get("artifact_type")
    if artifact != ARTIFACT_TYPE:
        if artifact == "pusht_fm_baseline":
            # The artifact_type written before the payload format was converted.
            raise ValueError(
                "This is an unconverted checkpoint from the original research "
                "repository; the shipped files under checkpoints/ are already "
                "converted."
            )
        raise ValueError(
            f"Expected artifact_type={ARTIFACT_TYPE!r}, got {artifact!r}. Foreign "
            "payloads decode into plausible, wrong actions; refusing."
        )
    missing = TOP_LEVEL_KEYS - raw.keys()
    if missing:
        raise ValueError(f"Payload is missing top-level keys {sorted(missing)}.")
    representation = Representation.from_metadata(raw["representation"])
    method = method_from_metadata(raw["method"])
    horizons = HorizonSpec.from_metadata(raw["horizons"])
    if horizons.prediction_horizon != representation.prediction_horizon:
        raise ValueError(
            f"horizons.prediction_horizon={horizons.prediction_horizon} disagrees "
            f"with representation.prediction_horizon={representation.prediction_horizon}."
        )
    return CheckpointPayload(
        step=int(raw["step"]),
        seed=int(raw["seed"]),
        representation=representation,
        model=method,
        horizons=horizons,
        state_dict=raw["state_dict"],
    )


def load_payload(checkpoint_path: Path) -> CheckpointPayload:
    """Parse a checkpoint file, rejecting payloads this package did not write."""

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError(f"{checkpoint_path} does not hold a checkpoint payload.")
    try:
        return parse_payload(raw)
    except ValueError as error:
        raise ValueError(f"{checkpoint_path}: {error}") from error


def save_payload(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def build_source(method: Method, horizons: HorizonSpec) -> ActionSource:
    """Rebuild the flow source a payload describes. The source carries no weights."""

    if method.source_type == "vanilla":
        return VanillaSource()
    if method.source_type == "warm_preview":
        return WarmPreviewSource(
            horizons.prediction_horizon,
            horizons.execution_horizon,
            sigma=method.warm_sigma,
            context_length=method.warm_depth,
        )
    if method.source_type in FORECAST_SOURCE_TYPES:
        return ForecastWeightSource(
            horizons.prediction_horizon,
            horizons.execution_horizon,
            alpha=method.alpha,
        )
    raise ValueError(f"No source builder for source_type={method.source_type!r}.")


def build_time_profile(method: Method, horizons: HorizonSpec) -> HorizonTimeProfile | None:
    """The per-position training flow times, or None for a synchronized arm."""

    if method.flow_mode != "persistent":
        return None
    return HorizonTimeProfile(
        horizons.prediction_horizon,
        horizons.execution_horizon,
        alpha=method.alpha,
    )


def build_velocity_model(*, condition_dim: int, action_dim: int, chunk_size: int):
    """Construct the velocity model from the frozen recipe.

    The architecture is not a per-checkpoint axis in this package: every arm
    trains the same 144,394-parameter transformer, so the dimensions come from
    the recipe and a mismatched state_dict raises on load.
    """

    recipe = TrainConfig()
    return TransformerVelocityModel(
        condition_dim=condition_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        d_model=recipe.d_model,
        num_blocks=recipe.num_blocks,
        num_heads=recipe.num_heads,
        ffn_dim=recipe.ffn_dim,
    )


def build_policy(payload: CheckpointPayload, device: torch.device) -> FlowMatchingPolicy:
    """Rebuild the exact policy a payload describes and load its weights."""

    representation = payload.representation
    action_dim = representation.action_dim
    policy = FlowMatchingPolicy(
        velocity_model=build_velocity_model(
            condition_dim=representation.condition_dim,
            action_dim=action_dim,
            chunk_size=representation.prediction_horizon,
        ),
        action_dim=action_dim,
        chunk_size=representation.prediction_horizon,
        num_integration_steps=payload.horizons.num_integration_steps,
        source=build_source(payload.model, payload.horizons),
        solver="euler",
        time_sampling="uniform",
        time_profile=build_time_profile(payload.model, payload.horizons),
    )
    # Strict by default: a shape mismatch raises here rather than mis-decoding.
    policy.load_state_dict(payload.state_dict)
    policy.to(device).eval()
    return policy
