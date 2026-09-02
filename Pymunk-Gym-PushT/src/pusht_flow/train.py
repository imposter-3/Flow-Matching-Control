"""Train one method at one seed under the frozen recipe.

Every method trains under RECIPE, which no method can override, so the arms
differ only in their source construction and their flow-time schedule. Width,
depth, learning rate and step budget are identical across arms, which is what
makes the comparison a comparison.

Run one training:

    uv run python -m pusht_flow.train --method coupled_a2 --seed 0

Determinism contract: one global stream, seeded once by torch.manual_seed,
consumed in a fixed order: model parameter initialization, then the loader's
shuffle seed at the first iteration, then per step one residual draw followed
by one flow-time draw. Reordering any of these changes the run.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pusht_flow import paths
from pusht_flow.checkpoint import save_checkpoint
from pusht_flow.config import METHODS, RECIPE, MethodConfig
from pusht_flow.data import build_dataset
from pusht_flow.flow import HorizonProfiles, flow_matching_loss, training_step_tensors
from pusht_flow.model import build_model, count_parameters


def learning_rate_at(step: int, recipe) -> float:
    """Linear warmup, flat, then a cosine tail over the final fraction."""

    if step < recipe.warmup_steps:
        return recipe.learning_rate * (step + 1) / recipe.warmup_steps
    start = int(recipe.max_train_steps * recipe.cosine_start_fraction)
    if step < start:
        return recipe.learning_rate
    span = max(recipe.max_train_steps - start, 1)
    progress = min((step - start) / span, 1.0)
    return recipe.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def resolve_method(name: str) -> MethodConfig:
    if name not in METHODS:
        raise SystemExit(f"Unknown method {name!r}. Known: {', '.join(METHODS)}.")
    return METHODS[name]


def train(
    method: MethodConfig,
    seed: int,
    *,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    max_steps: int | None = None,
    log_interval: int = 100,
) -> Path:
    recipe = RECIPE
    steps = max_steps or recipe.max_train_steps

    # One global stream, seeded once, before any torch RNG is consumed. The
    # source residual is drawn before the flow time inside
    # training_step_tensors; that order is part of the recipe.
    torch.manual_seed(seed)

    dataset, normalizer = build_dataset(data_dir, chunk_size=recipe.chunk_size)
    loader = DataLoader(
        dataset,
        batch_size=recipe.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    model = build_model(
        condition_dim=normalizer.condition_dim,
        action_dim=normalizer.action_dim,
        recipe=recipe,
    ).to(device)
    profiles = HorizonProfiles.build(
        method,
        chunk_size=recipe.chunk_size,
        execution_horizon=recipe.execution_horizon,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=recipe.learning_rate,
        weight_decay=recipe.weight_decay,
    )

    print(
        f"method={method.key} seed={seed} device={device} "
        f"parameters={count_parameters(model):,} chunks={len(dataset):,} "
        f"steps={steps:,}"
    )

    step = 0
    started = time.time()
    model.train()
    while step < steps:
        for condition, data_chunk in loader:
            if step >= steps:
                break
            condition = condition.to(device, non_blocking=True)
            data_chunk = data_chunk.to(device, non_blocking=True)

            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at(step, recipe)

            source_chunk, tau = training_step_tensors(data_chunk, method, profiles)
            loss = flow_matching_loss(model, condition, data_chunk, source_chunk, tau)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.grad_clip_norm)
            optimizer.step()

            if step % log_interval == 0:
                elapsed = time.time() - started
                print(
                    f"  step {step:>6d}/{steps}  loss {loss.item():9.4f}  "
                    f"lr {learning_rate_at(step, recipe):.2e}  {elapsed:7.1f}s"
                )
            step += 1

    output = paths.checkpoint_path(method.key, seed, Path(output_dir))
    save_checkpoint(
        output,
        model=model,
        normalizer=normalizer,
        method=method,
        recipe=recipe,
        step=step,
        seed=seed,
    )
    summary = output.with_suffix(".train.json")
    summary.write_text(
        json.dumps(
            {
                "method": method.key,
                "seed": seed,
                "steps": step,
                "final_loss": float(loss.item()),
                "parameters": count_parameters(model),
                "num_chunks": len(dataset),
                "wall_clock_seconds": round(time.time() - started, 1),
            },
            indent=2,
        )
    )
    print(f"  saved {output}  ({time.time() - started:.1f}s)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="cfm_restart", choices=list(METHODS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=paths.DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=paths.CHECKPOINT_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override the recipe's step budget. Smoke tests only.",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train(
        resolve_method(args.method),
        args.seed,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=device,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
