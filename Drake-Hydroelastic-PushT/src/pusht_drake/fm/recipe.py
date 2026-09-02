"""The frozen training recipe and the run configuration.

The optimizer recipe is kept off every CLI surface: this study varies the
source construction against a fixed optimizer, and an arm that also moved the
learning rate would be a second experiment, not a comparison.

The training budget is derived, never typed. The rule is training_epochs passes
at the realized updates per epoch. That number moves whenever the horizon or
the corpus moves, so max_train_steps=None means "derive it" and a typed value
raises unless budget_override is set. On the shipped corpus at H_p=16 the
derivation is 400 epochs x 106 updates = 42,400 steps.

The loss sums over (H, A), so its scale follows the horizon. The learning rate
still holds because the published runs clipped the gradient on every logged
step, which quotients the scale out of every effective update; if the logged
clip fraction ever drops below 1.0 that argument stops holding and the rate
must be revisited.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from pusht_drake.config import FORECAST_SOURCE_TYPES, SourceMode
from pusht_drake.fm.schedules import validate_schedules

DATA_LOADER_WORKERS = 4
# Batch order gets its own stream, offset so that two generators seeded from
# the same run seed cannot walk the same sequence.
DATA_ORDER_SEED_OFFSET = 1_000
TAIL_COSINE_START_FRACTION = 0.75
TAIL_COSINE_MIN_LR_RATIO = 0.1

# The selected recipe, frozen. AdamW with the bias/norm/position-embedding
# no-decay partition, 500-step linear warmup, tail cosine over the final
# quarter, grad clip 1.0, batch 512. No EMA anywhere.
BATCH_SIZE = 512
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 500
GRAD_CLIP_NORM: float | None = 1.0
ADAM_BETAS = (0.9, 0.999)

LOG_INTERVAL = 25

# The budget rule: passes over the training chunks. Steps are derived from it
# and the realized loader length, never typed (see the module docstring).
TRAINING_EPOCHS = 400


def tail_cosine_multiplier(
    step: int,
    total_steps: int,
    start_fraction: float = TAIL_COSINE_START_FRACTION,
) -> float:
    """Hold the base LR, then cosine-decay it over the final training quarter."""

    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}.")
    decay_start = int(total_steps * start_fraction)
    if step <= decay_start:
        return 1.0
    progress = min((step - decay_start) / (total_steps - decay_start), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return TAIL_COSINE_MIN_LR_RATIO + (1.0 - TAIL_COSINE_MIN_LR_RATIO) * cosine


def learning_rate_multiplier(
    step: int,
    total_steps: int,
    warmup_steps: int,
    start_fraction: float = TAIL_COSINE_START_FRACTION,
) -> float:
    """Optional linear warmup followed by the shared tail cosine decay."""

    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    return tail_cosine_multiplier(step, total_steps, start_fraction)


@dataclass
class TrainConfig:
    """Configuration for one training arm on the Drake corpus.

    The control contract is two horizons. prediction_horizon is what the
    policy plans; execution_horizon is what it commits to before being
    re-queried. The gap between them is the forecast overlap that a warm or
    forecast-weighted source consumes.
    """

    # Control contract, shared by every arm.
    prediction_horizon: int = 16
    execution_horizon: int = 2

    # The flow's tau=0 state; set per arm from config.METHODS.
    source_type: SourceMode = "vanilla"
    warmprior_sigma: float = 1.0
    # Shared horizon-decay rate for the forecast weight and the flow-maturity
    # schedule. Inert unless source_type is a forecast one.
    alpha: float = 2.0
    # How many of the overlap's positions are warmed; None warms all of them.
    warm_depth: int | None = None

    # Velocity model. ffn_dim has no default on the model constructor; this is
    # the single source of the value.
    d_model: int = 64
    num_blocks: int = 3
    num_heads: int = 4
    ffn_dim: int = 88

    # None derives training_epochs x updates_per_epoch at the realized loader
    # length. A typed value raises unless budget_override is set, because a
    # stale literal here would change the budget rule with no error.
    training_epochs: int = TRAINING_EPOCHS
    max_train_steps: int | None = None
    budget_override: bool = False
    torch_compile: bool = True

    seed: int = 0

    # The action box in metres, the frozen 0.48 m square centred on (0.5, 0)
    # from the environment config. The evaluation harness cross-checks the
    # trained box against the environment on every run.
    action_low: tuple[float, float] = field(default=(0.26, -0.24))
    action_high: tuple[float, float] = field(default=(0.74, 0.24))


def scheduled_steps(config: TrainConfig, updates_per_epoch: int) -> int:
    """Resolve the training budget against the realized loader length.

    max_train_steps=None derives training_epochs x updates_per_epoch. A typed
    value must agree with that or carry budget_override; otherwise a stale
    literal would change the budget rule with no error.
    """

    if updates_per_epoch < 1:
        raise ValueError(f"updates_per_epoch must be >= 1, got {updates_per_epoch}.")
    derived = config.training_epochs * updates_per_epoch
    if config.max_train_steps is None:
        return derived
    if config.max_train_steps != derived and not config.budget_override:
        raise ValueError(
            f"max_train_steps={config.max_train_steps} disagrees with the budget rule "
            f"({config.training_epochs} epochs x {updates_per_epoch} updates/epoch = "
            f"{derived}). Leave it unset to derive it, set training_epochs to train "
            "longer, or pass budget_override to run off-rule."
        )
    return int(config.max_train_steps)


def validate_config(config: TrainConfig) -> None:
    """Reject option combinations before any data or model is built."""

    if config.prediction_horizon < 1:
        raise ValueError(f"prediction_horizon must be >= 1, got {config.prediction_horizon}.")
    if not 1 <= config.execution_horizon <= config.prediction_horizon:
        raise ValueError(
            f"execution_horizon must be in 1..{config.prediction_horizon}, got "
            f"{config.execution_horizon}."
        )
    overlap = config.prediction_horizon - config.execution_horizon
    if config.source_type in FORECAST_SOURCE_TYPES:
        if overlap < 1:
            raise ValueError(
                f"source_type={config.source_type!r} needs a forecast overlap, i.e. "
                f"execution_horizon < prediction_horizon; got {config.execution_horizon} "
                f"and {config.prediction_horizon}. Both horizon schedules divide by "
                "H_p - H_e."
            )
        if config.alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {config.alpha}.")
        # Raise on a broken schedule before any data or model is built.
        validate_schedules(config.prediction_horizon, config.execution_horizon, config.alpha)
    if config.source_type == "warm_preview":
        if overlap < 1:
            raise ValueError(
                "source_type='warm_preview' needs a forecast overlap, i.e. "
                f"execution_horizon < prediction_horizon; got {config.execution_horizon} "
                f"and {config.prediction_horizon}."
            )
        if config.warmprior_sigma <= 0.0:
            raise ValueError(f"warmprior_sigma must be positive, got {config.warmprior_sigma}.")
        if config.warm_depth is not None and not 1 <= config.warm_depth <= overlap:
            raise ValueError(
                f"warm_depth must be in 1..{overlap} (the overlap), got {config.warm_depth}."
            )
    if config.training_epochs < 1:
        raise ValueError(f"training_epochs must be >= 1, got {config.training_epochs}.")
    if config.max_train_steps is not None and config.max_train_steps <= 0:
        raise ValueError(f"max_train_steps must be positive, got {config.max_train_steps}.")
    if config.d_model % config.num_heads:
        raise ValueError(
            f"d_model {config.d_model} must be divisible by num_heads {config.num_heads}."
        )
    if config.ffn_dim <= 0:
        raise ValueError(f"ffn_dim must be positive, got {config.ffn_dim}.")
    low = np.asarray(config.action_low, dtype=float)
    high = np.asarray(config.action_high, dtype=float)
    if low.shape != (2,) or high.shape != (2,) or np.any(low >= high):
        raise ValueError(f"Bad action box: low {config.action_low}, high {config.action_high}.")


def set_seed(seed: int) -> None:
    """Seed every RNG stream that model construction and training draw from."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
