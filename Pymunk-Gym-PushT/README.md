# Pymunk-Gym-PushT

Flow matching action-chunk policies on state-based Push-T, using gym-pusht and
pymunk physics. Four methods are trained under a shared recipe and evaluated
under a fixed protocol, and the package reproduces the main comparison at one
network evaluation per replan (Table 1). Twelve trained checkpoints ship with
the repository, so the table reproduces without any training. Retraining from
scratch is supported and documented below.

## The four methods

All methods share the same network (144,394 parameters), data, optimizer and
step budget. They differ in three places: how the source distribution a0 is
built, which flow times training samples, and whether anything is carried
across a replan. The chunk length is H = 16, the execution horizon is H_e = 2,
and alpha = 2 throughout.

| Key | Name | Source a0 | Training flow time | Across replans |
|---|---|---|---|---|
| cfm_restart | CFM Restart | eps | one scalar, U(0,1) | nothing reused |
| warm2 | WarmPrior (Warm2) | forecast + sigma * eps on the first 2 positions, eps elsewhere | one scalar, U(0,1) | forecast reused |
| forecast_weight_a2 | Forecast Weight (alpha=2) | lambda_k * forecast + eps | one scalar, U(0,1) | forecast reused |
| coupled_a2 | Full Coupled (alpha=2) | lambda_k * forecast + eps | per-position interval, shared progress xi | partial flow state carried |

The two horizon schedules, with k the position and eps standard normal noise
(at training time the dataset action stands in for the forecast):

~~~
lambda_k = exp(-alpha * ((k+1) / (H - H_e))^2)    for k < H - H_e, else 0
tau'_k   = 1                                       for k < H_e
tau'_k   = exp(-alpha * (k - H_e + 1) / (H - H_e)) for H_e <= k < H
~~~

The persistent method trains each position over the interval from
tau_in_k = tau'_(k+H_e) to tau'_k, which is exactly the interval the receding
horizon produces at deployment, and its inference carries the partially
generated chunk across replans (shift by H_e, re-express in the new action
frame, re-anchor the source mean, refresh the tail with noise, continue).

## Layout

~~~
pyproject.toml         dependencies and build metadata; uv.lock is the record
checkpoints/           12 provided policies: <key>-s<seed>.pt
src/pusht_flow/
  config.py            task facts, the shared recipe, the methods, the protocol
  paths.py             project-root anchored default locations
  schedules.py         lambda_k, tau'_k, the flow interval, their invariants
  data.py              replay download, normalization, chunk dataset
  model.py             the velocity field (adaLN action expert)
  flow.py              sources, training flow times, the flow matching loss
  env.py               the scored Push-T environment
  checkpoint.py        the payload contract: save, load, validate
  rollout.py           the two inference paths: restart and persistent
  train.py             training CLI (one method, one seed)
  evaluate.py          evaluation CLI (one method, one seed)
  sweep.py             all 12 evaluation cells, parallel and resumable
  checks.py            the pre-flight check suite
data/                  the replay, created on first training run
results/rollouts/      one CSV per cell, created by the sweep
~~~

## Installation

### Prerequisites

- Python 3.11
- uv (https://docs.astral.sh/uv/)

No GPU is needed to reproduce the table; a GPU helps only for retraining.

1. **Sync the environment**, from this directory:

~~~
uv sync
~~~

2. **Run the pre-flight checks**:

~~~
uv run python -m pusht_flow.checks
~~~

They take a few seconds and validate the schedules, the method isolation, the
NFE accounting, the frame correction and the checkpoint contract before any
compute is spent.

## Reproducing Table 1 from the provided checkpoints

One command. The sweep evaluates every cell (4 methods x 3 training seeds =
12 cells, 900 episodes each) on CPU, one worker per cell, and finishes by
printing the table's numbers:

~~~
uv run python -m pusht_flow.sweep
~~~

On the reference machine (28 logical cores) this takes about four minutes.
The sweep is resumable: finished cells are detected and skipped, so an
interrupted run can simply be started again. The --smoke flag is a reduced
run for checking the installation: 20 episodes and one action block per cell,
into results-smoke/, never mixed with the data of record.

The data of record is one CSV per cell at
results/rollouts/<method>-s<seed>-nfe1.csv, one row per episode (the file
name records the operating point), with the columns

~~~
method, train_seed, env_seed, action_seed, nfe, alpha, max_coverage,
success_090, success_095, episode_length, checkpoint, final_coverage,
mean_reward, terminated, truncated, num_replans
~~~

The two table metrics are max_coverage and success_090; the last five columns
are debug records only. Each cell prints its own mean max coverage and
success rate as it finishes; those per-cell lines are the table's per-seed
values. A table cell is their two-stage reduction: the metric averaged over
the 900 episodes of each training seed, then mean +/- sample std (denominator
n-1) across the three seeds. The sweep prints that reduction, one line per
method, when it ends; once every cell is present, running it again re-prints
the numbers without recomputing anything. The expected values are below. On a
machine matching the reference environment the agreement is exact at three
decimals; on other hardware small deviations are expected, well inside the
printed +/- values.

A single cell can also be run directly:

~~~
uv run python -m pusht_flow.evaluate --method coupled_a2 --seed 0
~~~

## Expected results

Table 1: the main comparison at NFE = 1. Mean +/- sample std across the three
training seeds, 2700 episodes per cell; higher is better, bold marks the best
cell per column.

| Method | Max Coverage | Success@0.90 |
|---|---|---|
| CFM Restart | 0.508 +/- 0.015 | 0.133 +/- 0.049 |
| WarmPrior (Warm2) | 0.548 +/- 0.024 | 0.121 +/- 0.019 |
| Forecast Weight (alpha=2) | 0.721 +/- 0.064 | 0.440 +/- 0.076 |
| Full Coupled (alpha=2) | **0.862 +/- 0.004** | **0.703 +/- 0.013** |

Per-seed values behind Table 1, for checking a partial run:

| Method | Metric | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| CFM Restart | Max Coverage | 0.501 | 0.498 | 0.525 |
| CFM Restart | Success@0.90 | 0.188 | 0.096 | 0.114 |
| WarmPrior (Warm2) | Max Coverage | 0.524 | 0.549 | 0.571 |
| WarmPrior (Warm2) | Success@0.90 | 0.130 | 0.134 | 0.100 |
| Forecast Weight (alpha=2) | Max Coverage | 0.719 | 0.785 | 0.658 |
| Forecast Weight (alpha=2) | Success@0.90 | 0.413 | 0.526 | 0.380 |
| Full Coupled (alpha=2) | Max Coverage | 0.860 | 0.867 | 0.859 |
| Full Coupled (alpha=2) | Success@0.90 | 0.694 | 0.718 | 0.698 |

## Seeds and determinism

Nothing in the protocol is drawn at run time; every seed is a deterministic
function of its index, so every method sees identical scene initializations
and identical sampling noise.

| Axis | Values | Set by |
|---|---|---|
| Training seed | 0, 1, 2 | torch.manual_seed once per run, before data loading |
| Environment seed | 1000 to 1299, 300 values | env.reset with seed = 1000 + i |
| Action-noise seed | block + i for blocks 1000, 2000, 3000 | a fresh torch.Generator per episode |

The environment seed fixes the initial scene; the action seed fixes the
policy's sampling noise; keeping them independent is what makes the grid
paired across methods.

Evaluation runs on CPU, and that is part of the protocol rather than a
fallback: the per-episode noise generator is device-bound, so CPU and CUDA
draw different streams, and every published episode ran on CPU. The sweep
driver pins CPU; evaluate.py defaults to CPU and treats any other --device as
off-protocol. CPU is also the faster device here, since the network is
144,394 parameters queried at batch size 1 and GPU launch overhead dominates.

## Retraining from scratch

The replay (the standard Push-T human demonstration set, 206 episodes) is
fetched automatically on the first training run: a 30 MB zip from
https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip is
downloaded, verified against a pinned sha256 and extracted under data/. If the
URL is unreachable, place the same pusht.zip at data/pusht.zip by hand; the
checksum is still enforced.

~~~
for method in cfm_restart warm2 forecast_weight_a2 coupled_a2; do
  for seed in 0 1 2; do
    uv run python -m pusht_flow.train --method $method --seed $seed
  done
done
~~~

Each run takes about 73 seconds on the reference GPU (17,600 steps at batch
512) and writes checkpoints/<method>-s<seed>.pt plus a small summary next to
it, overwriting the provided checkpoint of the same name. Final training
losses of the runs behind the table, as a wiring check:

| Method | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| CFM Restart | 7.246 | 6.972 | 6.931 |
| WarmPrior (Warm2) | 7.191 | 6.816 | 6.725 |
| Forecast Weight (alpha=2) | 5.810 | 5.784 | 5.700 |
| Full Coupled (alpha=2) | 4.322 | 4.238 | 4.041 |

Losses differ between methods because a source nearer the data gives a
lower-variance regression target; only the within-method spread would indicate
a wiring fault. On hardware and torch versions matching the reference
environment, retraining reproduces the provided checkpoints; elsewhere expect
statistically equivalent rather than bit-identical results.

## Scoring and aggregation

Coverage is the environment's own measure of how much of the goal region the
block overlaps. Per episode the maximum coverage over all steps is recorded;
Max Coverage is its mean over episodes, and Success@0.90 is the fraction of
episodes whose maximum exceeds 0.90. The 0.90 threshold is chosen because the
demonstration data itself peaks at 0.9014: across all 25,650 demonstration
frames none exceeds 0.95, so a 0.95 bar would score imitation of behaviour
the data never contains. A stricter 0.95 reading is still written to every
CSV row for auditing, and episode termination stays the environment's native
rule, untouched, so trajectories are identical to the unmodified environment.

Aggregation is two-stage: each training seed is reduced to its own mean over
900 episodes, then the table reports mean +/- sample std (denominator n-1)
across the three seeds. The error bar is disagreement between independently
trained policies, not episode spread.

## Reproducibility notes

- The environment of record is Python 3.11 with the packages pinned in
  uv.lock (torch 2.13.0, gym-pusht 0.1.6, pymunk 6.11.1, numpy 2.4.6).
  Exact three-decimal agreement is expected under that environment; contact
  dynamics amplify floating point differences, so other platforms may drift
  slightly, well inside the printed +/- values.
- Three training seeds means three samples behind every +/- value; read the
  error bars accordingly.
- The provided checkpoints are the original trained weights, unmodified. As a
  provenance check, retraining coupled_a2 at seed 0 under the reference
  environment reproduced its provided checkpoint bit for bit, at the recorded
  final loss.
