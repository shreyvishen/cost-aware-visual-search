"""Gate 1 — difficulty band (blocking, GOAL §9).

Measures thumbnail-only accuracy: the model sees the thumbnail, gets NO zoom, and answers.

    0.15 <= acc <= 0.45   -> in band, train
    acc > 0.45            -> the thumbnail already answers it; zooming earns nothing and
                             there is no gradient. Shrink the thumbnail.
    acc < 0.15            -> unsolvable even with a look. Grow the thumbnail.

We adjust the thumbnail, never the model.

This runs through the REAL environment (`src.rollout.collect` with `max_zooms=0`), not a
separately built prompt, so the number it reports is the number training will actually see.
The polarity balance matters: the raw corpus answers "yes" 82% of the time, so an unbalanced
probe measures the class prior instead of the difficulty.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src import data as data_mod  # noqa: E402
from src import rollout  # noqa: E402
from src.backends import get_backend  # noqa: E402
from src.train import score_episodes, summarize  # noqa: E402
from src.reward import CostModel  # noqa: E402

LO, HI = 0.15, 0.45


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--thumb-max-side", type=int, default=384)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--zooms", type=int, default=0,
                    help="0 = thumbnail-only (the gate). >0 measures how much zooming buys.")
    ap.add_argument("--non-binary", action="store_true",
                    help="drop yes/no golds. Chance is ~0.5 on a binary corpus, which puts "
                         "the [0.15,0.45] band below the floor and makes it unreachable.")
    ap.add_argument("--out", default=str(REPO / ".notes" / "g1_band.json"))
    args = ap.parse_args()

    pool = data_mod.load_train(limit=args.n * 12, seed=args.seed)
    if args.non_binary:
        pool = [s for s in pool if data_mod.answer_shape(s.gold) != "yes_no"]
        print(f"non-binary pool: {len(pool)} samples")
    else:
        pool = data_mod.balance_polarity(pool, seed=args.seed)
    pool = pool[: args.n]
    print(f"probing {len(pool)} train samples, max_zooms={args.zooms}")

    backend = get_backend("hf", model=None, device=args.device)
    cfg = {"max_zooms": args.zooms, "max_new_tokens": 200, "downsample": args.downsample,
           "thumb_max_side": args.thumb_max_side, "crop_max_side": 512}
    eps = rollout.collect(backend, backend.proc, backend.tok, pool, 1, cfg, temperature=0.0)
    score_episodes(eps, CostModel(mode="none"), 0.0)
    row = summarize(eps)

    acc = row["accuracy"]
    thumb_px = round(args.thumb_max_side)
    verdict = "IN_BAND" if LO <= acc <= HI else ("TOO_EASY" if acc > HI else "TOO_HARD")
    rec = None
    if verdict == "TOO_EASY":
        rec = {"downsample": args.downsample * 2, "thumb_max_side": max(96, thumb_px // 2)}
    elif verdict == "TOO_HARD":
        rec = {"downsample": max(2, args.downsample // 2), "thumb_max_side": thumb_px * 2}

    report = {"n": len(eps), "accuracy": acc, "max_zooms": args.zooms,
              "non_binary": args.non_binary, "tool_rate": row["tool_rate"],
              "mean_zooms": row["mean_zooms"], "band": [LO, HI], "verdict": verdict,
              "downsample": args.downsample, "thumb_max_side": args.thumb_max_side,
              "invalid_format_rate": row["invalid_format_rate"],
              "answer_source": row["answer_source"],
              "mean_decode_tokens": row["mean_decode_tokens"],
              "recommendation": rec}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))
    if verdict == "IN_BAND":
        print(f"GATE 1 PASS: thumbnail-only accuracy {acc:.3f} is inside [{LO}, {HI}]")
        return 0
    print(f"GATE 1 FAIL ({verdict}): accuracy {acc:.3f} outside [{LO}, {HI}]. "
          f"Adjust the thumbnail: {rec}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
