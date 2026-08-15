#!/usr/bin/env python
"""Gate 1 — thumbnail-only difficulty band. BLOCKING. (GOAL S9, Phase 4)

Show the BASE model the thumbnail and nothing else. No zoom tool, no crops. Measure
accuracy. The whole project needs the task to sit in a band where zooming is what
decides the answer:

    acc < 0.15  -> unsolvable even with a zoom. No gradient. Thumbnail too small.
    acc > 0.45  -> answerable without zooming. No gradient. Thumbnail too big.
    0.15..0.45  -> PASS.

**On failure, change the thumbnail, not the model** (GOAL S9). The script prints the
recommended `DOWNSAMPLE` / `THUMB_MAX_SIDE` and exits nonzero.

Usage:
    python gates/g1_difficulty.py --backend vllm --n 200
    python gates/g1_difficulty.py --backend hf --n 200 --model /srv/ai/models/current/qwen35-4b

Never run this on GPU 0, 1, or 5. Set CUDA_VISIBLE_DEVICES=2,3 first.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.contract import DOWNSAMPLE, THUMB_MAX_SIDE  # noqa: E402
from src.data import answer_correct, load_train  # noqa: E402

BAND_LO = 0.15
BAND_HI = 0.45

#: No tool, no scaffold. We are measuring what the thumbnail alone supports.
PROMPT = (
    "<image>\n{question}\n"
    "Answer with a short phrase. Put the final answer between <answer> and </answer> tags."
)


def make_thumbnail(img, downsample: int, max_side: int):
    """The exact image the policy sees at step 0.

    Downsample by `downsample`, then clamp the longest side to `max_side`. Both
    limits apply, and the smaller result wins -- that is what the env does, so the
    gate must not measure a different picture.
    """
    w, h = img.size
    tw, th = max(1, w // downsample), max(1, h // downsample)
    longest = max(tw, th)
    if longest > max_side:
        scale = max_side / longest
        tw, th = max(1, int(tw * scale)), max(1, int(th * scale))
    return img.resize((tw, th))


def extract_answer(text: str) -> str:
    """Read `<answer>...</answer>`; fall back to the last non-empty line."""
    import re

    m = re.findall(r"<answer>(.*?)</answer>", text, re.S | re.I)
    if m:
        return m[-1].strip()
    m = re.search(r"<answer>(.*)", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def recommend(acc: float, downsample: int, max_side: int) -> str:
    if acc > BAND_HI:
        return (
            f"acc {acc:.3f} > {BAND_HI}: the thumbnail already answers it, so zooming "
            f"earns nothing and there is no gradient. SHRINK the thumbnail: "
            f"DOWNSAMPLE {downsample} -> {downsample * 2}, or THUMB_MAX_SIDE "
            f"{max_side} -> {max(128, max_side // 2)}. Re-run this gate."
        )
    if acc < BAND_LO:
        return (
            f"acc {acc:.3f} < {BAND_LO}: the thumbnail is past unreadable, so even a "
            f"correct zoom cannot rescue the episode and reward stays flat at zero. "
            f"GROW the thumbnail: DOWNSAMPLE {downsample} -> "
            f"{max(1, downsample // 2)}, or THUMB_MAX_SIDE {max_side} -> "
            f"{max_side * 2}. Re-run this gate."
        )
    return f"acc {acc:.3f} is inside [{BAND_LO}, {BAND_HI}]. Keep DOWNSAMPLE={downsample}, THUMB_MAX_SIDE={max_side}."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    ap.add_argument("--model", default=None, help="passed through to get_backend")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--downsample", type=int, default=DOWNSAMPLE)
    ap.add_argument("--max-side", type=int, default=THUMB_MAX_SIDE)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=32)
    # ON by default, and it must stay on. Unbalanced, answering "yes" to everything
    # scores 0.527 on this train split -- already above the 0.45 band ceiling. The
    # gate would then measure the class prior, not whether the thumbnail is readable.
    ap.add_argument("--no-balance", dest="balance", action="store_false", default=True,
                    help="do NOT balance the yes/no golds 50/50 (not recommended; see the 0.824 yes-bias)")
    ap.add_argument("--out", default=".notes/g1_difficulty.json")
    ap.add_argument("--dump", default=".notes/g1_predictions.jsonl")
    args = ap.parse_args()

    # Imported lazily and by name: src.backends is built by another track, and this
    # gate must not fail at import time while that is still in flight.
    try:
        from src.backends import get_backend
    except Exception as exc:  # pragma: no cover
        print(f"FAIL: cannot import src.backends.get_backend ({exc}). "
              f"The backend track owns that module; this gate only calls it.", file=sys.stderr)
        return 2

    samples = load_train(limit=args.n * 3 if args.balance else args.n, seed=args.seed)
    if args.balance:
        from src.data import balance_polarity

        samples = balance_polarity(samples, seed=args.seed)
    samples = samples[: args.n]
    if len(samples) < args.n:
        print(f"WARN: asked for {args.n} samples, got {len(samples)}", file=sys.stderr)
    if not samples:
        print("FAIL: no samples loaded", file=sys.stderr)
        return 2

    backend = get_backend(args.backend) if args.model is None else get_backend(args.backend, model=args.model)

    t0 = time.time()
    records = []
    n_correct = 0
    for start in range(0, len(samples), args.batch):
        chunk = samples[start : start + args.batch]
        requests = [
            {
                "prompt": PROMPT.format(question=s.question),
                "images": [make_thumbnail(s.image, args.downsample, args.max_side)],
                "prompt_token_ids": None,
            }
            for s in chunk
        ]
        outs = backend.generate(
            requests,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            stop=["</answer>"],
        )
        for s, o in zip(chunk, outs):
            text = o.get("text", "") or ""
            pred = extract_answer(text)
            ok = answer_correct(pred, s.gold, s.options)
            n_correct += ok
            records.append(
                {
                    "sid": s.sid,
                    "question": s.question,
                    "gold": s.gold,
                    "raw": text,
                    "pred": pred,
                    "correct": bool(ok),
                    "shape": s.meta.get("shape"),
                    "polarity": s.meta.get("polarity"),
                    "thumb": list(make_thumbnail(s.image, args.downsample, args.max_side).size),
                    "orig": list(s.image.size),
                    "finish_reason": o.get("finish_reason"),
                }
            )
        done = min(start + args.batch, len(samples))
        print(f"  {done}/{len(samples)}  running acc {n_correct / done:.3f}", flush=True)

    acc = n_correct / len(records)
    by_shape = collections.defaultdict(lambda: [0, 0])
    for r in records:
        b = by_shape[r["shape"]]
        b[0] += r["correct"]
        b[1] += 1
    empty = sum(1 for r in records if not r["pred"])

    # A yes-only policy would score this much on the same set. If `acc` is close to
    # it, the gate is reading the class prior, not the resolution.
    yes_only = sum(1 for r in records if r["polarity"] == "yes") / len(records)

    print("\n=== Gate 1: thumbnail-only difficulty band ===")
    print(f"backend       : {args.backend}")
    print(f"samples       : {len(records)} (seed {args.seed}, balance={args.balance})")
    print(f"thumbnail     : DOWNSAMPLE={args.downsample} THUMB_MAX_SIDE={args.max_side}")
    print(f"elapsed       : {time.time() - t0:.1f}s")
    print(f"empty answers : {empty}")
    print(f"yes-only baseline on this set : {yes_only:.3f}")
    print(f"\nTHUMB-ONLY ACCURACY : {acc:.4f}   band [{BAND_LO}, {BAND_HI}]")
    if abs(acc - yes_only) < 0.05:
        print("  WARN: accuracy is within 0.05 of the always-yes baseline. This gate may be "
              "reading the class prior, not the thumbnail. Re-run WITHOUT --no-balance.")
    print("\nby answer shape:")
    for shape, (c, n) in sorted(by_shape.items(), key=lambda kv: -kv[1][1]):
        print(f"  {shape:22s} {c:4d}/{n:4d}  {c / n:.3f}")
    print("\n" + recommend(acc, args.downsample, args.max_side))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "acc": acc,
                "n": len(records),
                "band": [BAND_LO, BAND_HI],
                "downsample": args.downsample,
                "max_side": args.max_side,
                "backend": args.backend,
                "seed": args.seed,
                "balanced": args.balance,
                "empty_answers": empty,
                "yes_only_baseline": yes_only,
                "by_shape": {k: v for k, v in by_shape.items()},
                "pass": BAND_LO <= acc <= BAND_HI,
                "recommendation": recommend(acc, args.downsample, args.max_side),
            },
            indent=2,
        )
    )
    with open(args.dump, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"\n[wrote {args.out} and {args.dump}]")

    if BAND_LO <= acc <= BAND_HI:
        print("PASS")
        return 0
    print("FAIL: out of band. Adjust the thumbnail, not the model.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
