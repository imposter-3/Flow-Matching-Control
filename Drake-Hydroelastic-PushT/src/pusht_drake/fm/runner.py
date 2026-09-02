"""Training runner that builds every component, runs the loop, and saves.

Construction happens in __init__, learn() owns the optimization loop. Data
arrives as numpy arrays, not a path: this tier must not import zarr, so the
entry point (pusht_drake.train) loads the corpus and hands the columns in.

No resume and no warm start. A full run is under two minutes of optimizer
time on one GPU, so a crashed run is simply restarted.

The determinism contract, in construction order. set_seed comes first, before
any torch module exists; the data build consumes no RNG at all; the DataLoader
gets an explicit CPU generator at seed + 1000 so batch order cannot depend on
how far model construction advanced the global stream; the velocity model is
then the only consumer of the global CPU stream. Each optimizer step draws
exactly twice from the device's global stream: the source epsilon first, the
flow time second. Reordering any of this changes the run.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from pusht_drake.config import Method
from pusht_drake.fm.checkpoint import (
    HorizonSpec,
    build_payload,
    build_source,
    build_time_profile,
    build_velocity_model,
    save_payload,
)
from pusht_drake.fm.dataset import ChunkDataset, observations_from_columns
from pusht_drake.fm.flow_matching import FlowMatchingPolicy
from pusht_drake.fm.recipe import (
    ADAM_BETAS,
    BATCH_SIZE,
    DATA_LOADER_WORKERS,
    DATA_ORDER_SEED_OFFSET,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    LOG_INTERVAL,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    TrainConfig,
    learning_rate_multiplier,
    scheduled_steps,
    set_seed,
    validate_config,
)
from pusht_drake.fm.representation import Representation
from pusht_drake.fm.sources import WarmContext


def _skips_weight_decay(name: str, parameter: torch.Tensor) -> bool:
    """Biases, normalization gains, and embeddings are excluded from decay."""

    return parameter.ndim < 2 or "position_embedding" in name


def build_optimizer(policy: FlowMatchingPolicy, device: torch.device) -> torch.optim.Optimizer:
    """Group parameters so those skipping weight decay are decayed at 0.

    There is no config argument: the recipe is the frozen constants in
    pusht_drake.fm.recipe, which this signature reads directly.
    """

    decay: list[torch.Tensor] = []
    no_decay: list[torch.Tensor] = []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        target = no_decay if _skips_weight_decay(name, parameter) else decay
        target.append(parameter)
    groups: list[dict[str, Any]] = [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        fused=True if device.type == "cuda" else None,
    )


class Runner:
    """Build every component once, then train and save one checkpoint."""

    def __init__(
        self,
        method: Method,
        config: TrainConfig,
        *,
        states: np.ndarray,
        slider_states: np.ndarray,
        actions: np.ndarray,
        episode_ends: np.ndarray,
        device: torch.device | None = None,
    ) -> None:
        validate_config(config)
        self.method = method
        self.config = config

        # Seed before any torch module is constructed: initialization draws
        # from the global stream, in construction order.
        set_seed(config.seed)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        observations = observations_from_columns(states, slider_states)
        actions = np.asarray(actions, dtype=np.float32)
        episode_ends = np.asarray(episode_ends, dtype=np.int64)

        self.representation = Representation.fit(
            observations,
            actions,
            episode_ends,
            prediction_horizon=config.prediction_horizon,
            action_low=np.asarray(config.action_low),
            action_high=np.asarray(config.action_high),
        )

        dataset = ChunkDataset(
            observations,
            actions,
            episode_ends,
            chunk_size=config.prediction_horizon,
            representation=self.representation,
            execution_horizon=config.execution_horizon,
        )
        self.data_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=DATA_LOADER_WORKERS,
            pin_memory=self.device.type == "cuda",
            persistent_workers=DATA_LOADER_WORKERS > 0,
            # Without this the sampler seeds itself from the global RNG at each
            # epoch, which the model's construction has already advanced. Batch
            # order would then move whenever the architecture changed, breaking
            # paired comparisons on everything but the init seed.
            generator=torch.Generator().manual_seed(config.seed + DATA_ORDER_SEED_OFFSET),
        )
        if len(self.data_loader) == 0:
            raise ValueError(
                f"Dataset yields {len(dataset)} chunks < one batch of {BATCH_SIZE}; "
                "training would run zero steps."
            )

        self.horizons = HorizonSpec(
            prediction_horizon=config.prediction_horizon,
            execution_horizon=config.execution_horizon,
        )
        action_dim = self.representation.action_dim
        velocity_model = build_velocity_model(
            condition_dim=self.representation.condition_dim,
            action_dim=action_dim,
            chunk_size=config.prediction_horizon,
        )
        # The same constructors the checkpoint reader uses, fed the same method
        # that gets written, so a run cannot train against a source its own
        # payload would rebuild differently.
        self.source = build_source(method, self.horizons)
        # None for every synchronized arm, so their training is bitwise unchanged.
        self.time_profile = build_time_profile(method, self.horizons)
        self.policy = FlowMatchingPolicy(
            velocity_model=velocity_model,
            action_dim=action_dim,
            chunk_size=config.prediction_horizon,
            num_integration_steps=1,
            source=self.source,
            solver="euler",
            time_sampling="uniform",
            time_profile=self.time_profile,
        ).to(self.device)

        self.optimizer = build_optimizer(self.policy, self.device)
        self.scheduled_steps = scheduled_steps(config, len(self.data_loader))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: learning_rate_multiplier(
                step, self.scheduled_steps, WARMUP_STEPS
            ),
        )

        compile_active = config.torch_compile and self.device.type == "cuda"
        compute_losses = self.policy.compute_losses
        if compile_active:
            # Compile the pure loss/backward boundary; fused Adam remains eager,
            # and the ODE integration is never compiled (FlowMatchingPolicy).
            compute_losses = torch.compile(
                compute_losses,
                fullgraph=True,
                options={
                    "triton.cudagraphs": True,
                    # Preserve eager CUDA RNG consumption for paired experiments.
                    "fallback_random": True,
                },
            )
        self.loss_function = compute_losses
        self.compile_active = compile_active
        self.num_chunks = len(dataset)

        print(
            f"method={method.key} seed={config.seed} chunks={self.num_chunks:,} "
            f"updates_per_epoch={len(self.data_loader)} steps={self.scheduled_steps:,} "
            f"compile={compile_active}"
        )

    def oracle_context(
        self, action_chunk: torch.Tensor, warm_flag: torch.Tensor
    ) -> WarmContext | None:
        """The training-time preview: the target chunk's own leading positions.

        At rollout the warm mean is the previous forecast's tail, re-anchored;
        here the ground truth stands in for it. Those are the same physical
        timesteps (the previous plan's position H_e + m is this plan's
        position m), so the oracle is the zero-forecast-error limit of what
        deployment supplies rather than a different quantity.

        valid stays a tensor. As a Python bool it bakes into the compiled
        graph as a constant and recompiles once per value.
        """

        if not self.source.requires_context:
            return None
        return WarmContext(
            mean=action_chunk[:, : self.source.context_length],
            valid=warm_flag.reshape(-1, 1),
        )

    def learn(self, checkpoint_path: Path) -> dict[str, Any]:
        """Optimize the policy, then save the payload and a summary."""

        device = self.device
        non_blocking = device.type == "cuda"
        global_step = 0
        final_loss = float("nan")
        # The published loss table quotes the mean of the final 25 logged
        # steps; keep the same window so the summary is comparable.
        recent: list[torch.Tensor] = []
        started = time.time()

        self.policy.train()
        for _ in itertools.count():
            for batch in self.data_loader:
                condition, action_chunk, warm_flag = batch
                condition = condition.to(device, non_blocking=non_blocking)
                action_chunk = action_chunk.to(device, non_blocking=non_blocking)
                warm_flag = warm_flag.to(device, non_blocking=non_blocking)
                context = self.oracle_context(action_chunk, warm_flag)

                self.optimizer.zero_grad(set_to_none=True)
                losses = self.loss_function(condition, action_chunk, context)
                losses.total.backward()
                # CUDA graph outputs reuse storage; clone every read scalar.
                loss_value = losses.total.detach().clone()
                recent.append(loss_value)
                if len(recent) > LOG_INTERVAL:
                    recent.pop(0)
                if GRAD_CLIP_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), GRAD_CLIP_NORM)
                self.optimizer.step()
                self.scheduler.step()
                global_step += 1

                if global_step % LOG_INTERVAL == 0 or global_step == self.scheduled_steps:
                    final_loss = float(loss_value.item())
                    print(
                        f"  step {global_step:>6d}/{self.scheduled_steps}  "
                        f"loss {final_loss:9.4f}  "
                        f"lr {self.optimizer.param_groups[0]['lr']:.2e}  "
                        f"{time.time() - started:7.1f}s"
                    )
                if global_step >= self.scheduled_steps:
                    break
            if global_step >= self.scheduled_steps:
                break

        if global_step == 0:
            raise RuntimeError("The data loader produced no optimizer batches.")

        payload = build_payload(
            state_dict=self.policy.state_dict(),
            representation=self.representation,
            method=self.method,
            horizons=self.horizons,
            step=global_step,
            seed=self.config.seed,
        )
        save_payload(checkpoint_path, payload)
        summary = {
            "method": self.method.key,
            "seed": self.config.seed,
            "steps": global_step,
            "final_loss": float(loss_value.item()),
            "final_loss_window": float(torch.stack(recent).mean().item()),
            "parameters": sum(p.numel() for p in self.policy.parameters()),
            "num_chunks": self.num_chunks,
            "updates_per_epoch": len(self.data_loader),
            "torch_compile_active": self.compile_active,
            "device": self.device.type,
            "config": {
                key: (list(value) if isinstance(value, tuple) else value)
                for key, value in asdict(self.config).items()
            },
            "wall_clock_seconds": round(time.time() - started, 1),
        }
        Path(checkpoint_path).with_suffix(".train.json").write_text(json.dumps(summary, indent=2))
        print(f"  saved {checkpoint_path}  ({summary['wall_clock_seconds']}s)")
        return summary
