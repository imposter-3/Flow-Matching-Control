"""The checkpoint format: what a saved policy holds and what is checked on load.

A checkpoint carries the weights together with everything needed to interpret
them: the normalizer statistics (what the numbers mean), the method's
algorithmic fields (how to run inference), and the recipe (how to rebuild the
architecture). Restoring weights against different statistics or a different
method produces wrong predictions with no error, so all of it travels as one
payload and is cross-checked on load.

The payload holds only tensors and plain Python values, so it loads under
torch.load with weights_only=True.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import NamedTuple

import torch

from pusht_flow.config import METHODS, MethodConfig, Recipe
from pusht_flow.data import Normalizer
from pusht_flow.model import build_model

ARTIFACT_TYPE = "pusht_flow_policy"
FORMAT_VERSION = 2

#: The algorithmic identity of a method; the display name stays in config.py.
METHOD_FIELDS = (
    "key",
    "source_mode",
    "warm_count",
    "flow_mode",
    "alpha",
    "warmprior_sigma",
)


def build_payload(
    *,
    model,
    normalizer: Normalizer,
    method: MethodConfig,
    recipe: Recipe,
    step: int,
    seed: int,
) -> dict:
    """Assemble the dict that torch.save writes."""

    return {
        "artifact_type": ARTIFACT_TYPE,
        "format_version": FORMAT_VERSION,
        "state_dict": model.state_dict(),
        "normalizer": normalizer.to_metadata(),
        "method": {name: getattr(method, name) for name in METHOD_FIELDS},
        "recipe": {f.name: getattr(recipe, f.name) for f in fields(recipe)},
        "step": int(step),
        "seed": int(seed),
    }


class LoadedCheckpoint(NamedTuple):
    """A parsed checkpoint. Named access keeps call sites hard to mix up."""

    model: object
    normalizer: Normalizer
    method: MethodConfig
    recipe: Recipe
    step: int
    seed: int


def parse_payload(
    payload: dict, *, device: torch.device | str = "cpu"
) -> LoadedCheckpoint:
    """Rebuild the model and its metadata from a payload.

    The stored method fields must match the named entry in METHODS exactly. A
    checkpoint whose payload disagrees with the code's definition of its own
    method key is refused rather than trusted: every result is labelled by
    key, so a mismatch would mislabel a table row.
    """

    artifact = payload.get("artifact_type")
    if artifact != ARTIFACT_TYPE:
        if artifact == "experiment_policy":
            # An earlier payload format, kept recognizable for a clear error.
            raise ValueError(
                "This checkpoint is in the earlier 'experiment_policy' format; "
                "the files shipped under checkpoints/ are already converted."
            )
        raise ValueError(
            f"Not a policy checkpoint of this package "
            f"(artifact_type={artifact!r}, expected {ARTIFACT_TYPE!r})."
        )

    stored = payload["method"]
    key = stored.get("key")
    if key not in METHODS:
        raise ValueError(
            f"Checkpoint names unknown method {key!r}. Known: {', '.join(METHODS)}."
        )
    method = METHODS[key]
    for name in METHOD_FIELDS:
        if stored.get(name) != getattr(method, name):
            raise ValueError(
                f"Checkpoint field method.{name}={stored.get(name)!r} does not "
                f"match the definition of {key!r} "
                f"({getattr(method, name)!r}). Refusing to mislabel a result."
            )

    recipe = Recipe(**payload["recipe"])
    normalizer = Normalizer.from_metadata(payload["normalizer"])
    model = build_model(
        condition_dim=normalizer.condition_dim,
        action_dim=normalizer.action_dim,
        recipe=recipe,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return LoadedCheckpoint(
        model=model,
        normalizer=normalizer,
        method=method,
        recipe=recipe,
        step=int(payload["step"]),
        seed=int(payload["seed"]),
    )


def save_checkpoint(
    path: Path,
    *,
    model,
    normalizer: Normalizer,
    method: MethodConfig,
    recipe: Recipe,
    step: int,
    seed: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(
        model=model,
        normalizer=normalizer,
        method=method,
        recipe=recipe,
        step=step,
        seed=seed,
    )
    torch.save(payload, path)


def load_checkpoint(path: Path, *, device: torch.device | str = "cpu"):
    """Load and validate one checkpoint file. See parse_payload."""

    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not hold a checkpoint payload.")
    try:
        return parse_payload(payload, device=device)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
