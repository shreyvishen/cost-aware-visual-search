# Cost-aware visual search

A small VLM learns **where to look**, trained with a reward that charges **milliseconds measured
on the machine it runs on.**

The published SOTA zoom agent, DeepEyes, calls its zoom tool on 100% of inputs. Its reward pays
for looking and never charges for it, so the policy always looks. Here looking costs real time,
and that one term changes the policy: same accuracy, **2.18x faster on an M4 Max.**

## Setup

The model sees only a thumbnail — at least 4x downsampled, capped at 384 px — where a 12-pixel
object is gone. It gets one tool, `image_zoom_in_tool(bbox_2d=[x1,y1,x2,y2])`, which crops the
**original** full-resolution image and appends that crop to the conversation. Every box is in one
frame, on a normalised 0–1000 grid, so there is no zooming into a zoom and a box does not change
meaning when the thumbnail is resized. A hard budget of 3 zooms stops a policy from looking
forever and never being graded. Training is GRPO: 8 rollouts per prompt, each rollout's advantage
standardised against its own 8 siblings. No critic, no reward model.

The reward is `correct ? (1 - λ·cost_ms) : 0`, where

```
cost_ms = a·vision_tokens + b·decode_tokens + c·tool_calls
```

`a`, `b` and `c` are fitted once offline, by regressing against real `llama-server` wall clock on
an M4 Max across a grid of image sizes, generation lengths and crop counts (Q4: R² = 0.9959), then
frozen and sha256-pinned. Training computes cost from token **counts**, never from a live clock —
timing rollouts on a shared GPU would put thermal throttling and scheduler jitter into the
gradient, and GRPO cannot tell that apart from signal. Cost is gated on correctness, so a group
where all 8 rollouts fail cannot learn to fail more cheaply. Against real device time the frozen
table predicts at r ≥ 0.999.

Run A and Run B are the same code, data, seed, geometry and 55-minute timer. The only difference
is that A has no cost term.

## Results

| | Base | Run A (no cost term) | Run B (measured ms) |
|---|---|---|---|
| accuracy, 191 images | — | 0.482 | 0.476 |
| mean zooms, 191 images | — | 2.16 | 1.54 |
| mean zooms, M4 | 0.83 | 1.31 | 0.42 |
| prefill tokens, M4 | 789 | 965 | 703 |
| decode tokens, M4 | 70.7 | 176.5 | 60.3 |
| latency, M4 | 1830 ms | 4056 ms | 1861 ms |
| $ per 1k questions | $0.034 | $0.055 | $0.030 |

Accuracy is all 191 V\*Bench images on the training rig at bf16. Every M4 row is an M4 Max at
Q4_K_M over the same 36 images. Dollars price those tokens at $0.03/M in and $0.15/M out.

- **Accuracy is matched.** B fixed 30 images A got wrong, A fixed 31 B got wrong, and McNemar's
  exact test on those 61 discordant pairs returns p = 1.000.
- **B is 2.18x faster than A on the M4.** Median per-image speedup 2.50x, faster on 34 of 36
  images, Wilcoxon signed-rank p = 5.9e-07. Training with no cost term made the base model 2.2x
  slower than where it started; the cost term gave essentially all of that back.
- **89% of B's saving is decode tokens, not zooms.** At Q4 a decode token costs 13.2 ms against
  1.41 ms for a vision token, so the cheapest milliseconds to give back are reasoning tokens. The
  policy did not learn to look less — it learned to stop talking. A reward that counted tool calls
  would have taught it the opposite lesson.
- **The tool still earns its place.** On the matched 96-image reference subset, zooming is worth
  **+0.146 accuracy** (0.250 → 0.396).

## Run it

```bash
python -m src.train --config configs/run_b.json --out /srv/ai/runs/run_b --resume
python3 scripts/compare_runs.py      # comparison table from mirrored run artifacts
python3 app/build_data.py            # regenerates app/static/data.json for the page
```

The last two read mirrored run artifacts from `~/archive/cost-aware-vlm/`; they are not in this
repo.

```
src/          zoom env, rollout collector, reward, GRPO update, training loop, backends
gates/        blocking checks; each exits nonzero on failure
cost_model/   the M4 fit and the frozen per-quantization coefficients
configs/      run_a (no cost term) and run_b (measured ms) — one harness, two configs
scripts/      night runner, mirror, comparison table, M4 verification
app/          the results page and the script that builds its data
```

## Scope

- **One run per condition, 16–17 gradient steps** on three RTX 3090s. These are behavioural deltas
  from a small number of updates, not converged policies, and there is no error bar on the
  comparison. Two seeds per condition would be the minimum to claim it properly.
- **Accuracy comes from the rig at bf16, latency from the M4 at Q4.** They are different stacks.
  Q4 shifts the whole distribution toward fewer zooms, so accuracy must not be read off the
  36-image M4 subset.
- **The cost model assumes prefix caching.** Without it, an episode that zooms costs about 767 ms
  more than the table predicts. That is a deployment requirement, not a footnote.
- I own this diff, not the engine: the model is Qwen3.5-4B, rollouts generate through HF
  `transformers`, device timing runs through `llama.cpp`, and the training data is DeepEyes'.
