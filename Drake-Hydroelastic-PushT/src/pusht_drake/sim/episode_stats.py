"""Reduce per-episode records to evaluation metrics. Numpy only.

The co-primary axes are mean_max_coverage (clip-free) and success_rate.
mean_reward is a guard, never primary, because the clip at tau saturates most
episodes and transmits only a sliver of a coverage change. The
receding-horizon-only fields are absent (open loop has no preview), and the
Drake-side means (tracking error, settle time, realtime factor) are added.
"""

from __future__ import annotations

from typing import Any

import numpy as np

CVAR_FRACTION = 0.1
NEAR_MISS_BAND = 0.05


def aggregate_records(records: list[dict[str, Any]], *, score_tau: float) -> dict[str, float]:
    """The metric table one evaluation run reports."""
    # One engine per aggregate. Mixing episodes from two simulators into one
    # mean is what the backend field exists to prevent, and a mean is exactly
    # where that mixing leaves no trace. A record with no backend key reads as
    # "drake", so older committed artifacts still aggregate.
    engines = {r.get("backend", "drake") for r in records}
    if len(engines) > 1:
        raise ValueError(
            f"refusing to aggregate {sorted(engines)} into one set of metrics; "
            f"these episodes came from different simulators"
        )

    if not records:
        raise ValueError("no episode records to aggregate")
    rewards = sorted(r["max_reward"] for r in records)
    tail = max(1, round(len(rewards) * CVAR_FRACTION))

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in records]))

    metrics = {
        "eval/mean_max_coverage": mean("max_coverage"),
        "eval/success_rate": mean("success"),
        "eval/mean_reward": mean("max_reward"),
        "eval/mean_final_coverage": mean("final_coverage"),
        "eval/mean_final_reward": mean("final_reward"),
        "eval/reward_p10": float(np.percentile(rewards, 10)),
        "eval/cvar10": float(np.mean(rewards[:tail])),
        "eval/mean_length": mean("length"),
        "eval/mean_replans": mean("num_replans"),
        "eval/terminated_rate": mean("terminated"),
        "eval/truncated_rate": mean("truncated"),
        "eval/sim_error_rate": float(
            np.mean([r.get("termination_reason") == "sim_error" for r in records])
        ),
        "eval/tolerance_success_rate": mean("tolerance_success"),
        # nanmean: a sim_error record carries NaN errors and must not poison
        # the aggregate of the episodes that finished
        "eval/mean_final_trans_err_m": float(np.nanmean([r["final_trans_err_m"] for r in records])),
        "eval/mean_final_rot_err_rad": float(np.nanmean([r["final_rot_err_rad"] for r in records])),
        "eval/clip_rate": mean("clip_rate"),
        # A warm policy must report exactly one cold replan per episode; a
        # drift here is the forecast cache de-synchronizing, which produces
        # plausible motion and no other symptom.
        "eval/cold_fallbacks": float(np.mean([r.get("cold_fallbacks", 0) for r in records])),
        "eval/peak_to_final_drop": float(
            np.mean([r["max_reward"] - r["final_reward"] for r in records])
        ),
        "eval/smoothness": mean("smoothness_m"),
        # Guard tick rates: how often the demonstration-time chain touched a
        # command. fence > 0 means the certified square-in-fence invariant is
        # not holding; leash > 0 means the policy asked for jumps the
        # demonstrator's 5 cm leash would have rate-limited.
        # .get: pre-spline artifacts (the recorded staircase baseline) lack
        # the key; treat them as never-clipped rather than KeyError.
        "eval/spline_clip_rate": float(np.mean([r.get("spline_clip_rate", 0.0) for r in records])),
        "eval/square_tick_rate": mean("square_tick_rate"),
        "eval/fence_tick_rate": mean("fence_tick_rate"),
        "eval/leash_tick_rate": mean("leash_tick_rate"),
        "eval/boundary_jump": mean("boundary_jump_mean_m"),
        "eval/within_chunk_step": mean("within_chunk_step_mean_m"),
        "eval/tracking_err_mean": mean("tracking_err_mean_m"),
        "eval/tracking_err_max": float(np.max([r["tracking_err_max_m"] for r in records])),
        "eval/mean_settle_time": mean("settle_time_s"),
        "eval/mean_wall_time": mean("wall_time_s"),
        "eval/mean_realtime_factor": mean("realtime_factor"),
        "eval/near_miss_rate": float(
            np.mean(
                [
                    (not r["success"]) and r["max_coverage"] >= score_tau - NEAR_MISS_BAND
                    for r in records
                ]
            )
        ),
        "eval/coverage_deficit": float(
            np.mean([max(0.0, score_tau - r["max_coverage"]) for r in records])
        ),
    }
    buckets = {
        "eval/bucket_below_050": sum(r["max_reward"] < 0.50 for r in records),
        "eval/bucket_050_090": sum(0.50 <= r["max_reward"] < 0.90 for r in records),
        "eval/bucket_090_099": sum(0.90 <= r["max_reward"] < 0.99 for r in records),
        "eval/bucket_above_099": sum(r["max_reward"] >= 0.99 for r in records),
    }
    metrics.update({key: float(value) / len(records) for key, value in buckets.items()})
    return metrics
