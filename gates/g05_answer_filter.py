#!/usr/bin/env python
"""Gate 0.5 — the answer filter that replaces the LLM judge. BLOCKING.

DeepEyes grades its reward with a served Qwen-2.5-72B judge (~140 GB). We do not
have that, and we do not want it: a judge injects stochastic reward noise into a
GRPO group of 8. We grade with `src.data.answer_correct` instead, deterministically.

The gate asks one question: **what fraction of the train golds can that checker
actually grade?** GOAL S9 sets the bar at 0.40.

    PASS  strict surviving fraction >= 0.40
    PASS  strict < 0.40 but WIDENED (normalized containment) >= 0.40
    FAIL  both below 0.40

Both numbers are always printed. Widening is the documented fallback, not a way to
paint a red gate green -- the strict number is reported either way.

Usage:
    python gates/g05_answer_filter.py [--rows N] [--report PATH]
"""
from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import (  # noqa: E402
    TRAIN_PARQUET,
    answer_correct,
    answer_is_scorable,
    answer_shape,
    normalize_answer,
)

THRESHOLD = 0.40
SHAPES = [
    "yes_no",
    "single_word",
    "number",
    "short_phrase",
    "attribute_sentence",
    "relational_sentence",
    "long_sentence",
]


def scan(n_rows: int) -> tuple[list[str], list[str]]:
    """Return (golds, questions) for up to n_rows. Light columns only — no images."""
    import pyarrow.parquet as pq

    if not TRAIN_PARQUET.exists():
        print(f"FAIL: train parquet missing at {TRAIN_PARQUET}", file=sys.stderr)
        sys.exit(2)

    pf = pq.ParquetFile(TRAIN_PARQUET)
    tbl = pf.read(columns=["reward_model", "extra_info"])
    rm = tbl.column("reward_model").to_pylist()
    ei = tbl.column("extra_info").to_pylist()
    golds = [(r or {}).get("ground_truth") or "" for r in rm]
    qs = [((e or {}).get("question") or "").strip() for e in ei]
    if n_rows and n_rows < len(golds):
        idx = random.Random(0).sample(range(len(golds)), n_rows)
        idx.sort()
        golds = [golds[i] for i in idx]
        qs = [qs[i] for i in idx]
    return golds, qs


def selfconsistency(golds: list[str], k: int = 400) -> tuple[int, int]:
    """A gold must grade itself correct. If it does not, the checker is broken."""
    sample = random.Random(7).sample(golds, min(k, len(golds)))
    ok = sum(1 for g in sample if answer_correct(g, g))
    return ok, len(sample)


def polarity_balance(golds: list[str]) -> collections.Counter:
    c = collections.Counter()
    for g in golds:
        if answer_shape(g) == "yes_no":
            c["yes" if normalize_answer(g).split()[0] in ("yes", "yeah", "yep", "true", "correct") else "no"] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=0, help="0 = every row (22,362)")
    ap.add_argument("--report", default=".notes/g05_report.md")
    args = ap.parse_args()

    golds, qs = scan(args.rows)
    total = len(golds)
    if total < 2000:
        print(f"FAIL: scanned only {total} rows, need >= 2000", file=sys.stderr)
        return 2

    shapes = collections.Counter(answer_shape(g) for g in golds)
    strict = [g for g in golds if answer_is_scorable(g)]
    wide = [g for g in golds if answer_is_scorable(g, widened=True)]
    f_strict = len(strict) / total
    f_wide = len(wide) / total

    rejected = [g for g in golds if not answer_is_scorable(g)]
    rejected_wide = [g for g in golds if not answer_is_scorable(g, widened=True)]
    common = collections.Counter(golds).most_common(20)
    sc_ok, sc_n = selfconsistency(golds)
    pol = polarity_balance(golds)

    L: list[str] = []
    p = L.append
    p("# Gate 0.5 — deterministic answer filter\n")
    p(f"Parquet: `{TRAIN_PARQUET}`  ")
    p(f"Rows scanned: **{total:,}**  ")
    p(f"Threshold: surviving fraction >= **{THRESHOLD:.2f}** (GOAL S9)\n")
    p("## Result\n")
    p(f"- **STRICT surviving fraction: {f_strict:.4f}** ({len(strict):,}/{total:,})")
    p(f"- **WIDENED surviving fraction: {f_wide:.4f}** ({len(wide):,}/{total:,})")
    verdict = "PASS (strict)" if f_strict >= THRESHOLD else ("PASS (widened)" if f_wide >= THRESHOLD else "FAIL")
    p(f"- Verdict: **{verdict}**\n")
    p("## Answer shapes\n")
    p("| shape | count | fraction | strict-scorable |")
    p("|---|---:|---:|:--:|")
    for s in SHAPES:
        n = shapes.get(s, 0)
        if not n:
            continue
        ex = next((g for g in golds if answer_shape(g) == s), "")
        keep = "yes" if answer_is_scorable(ex) else "no"
        p(f"| `{s}` | {n:,} | {n / total:.4f} | {keep} |")
    p("")
    p("## Yes/no polarity balance\n")
    tot_pol = sum(pol.values()) or 1
    for k, v in pol.most_common():
        p(f"- `{k}`: {v:,} ({v / tot_pol:.3f} of yes/no golds, {v / total:.3f} of all rows)")
    maj = max(pol.values()) / tot_pol if pol else 0.0
    p("")
    p(f"**Majority-class baseline on the yes/no subset: {maj:.3f}.** A model that always "
      f"answers the majority class scores that much without looking at the image. The "
      f"trainer must know this: balance the yes/no golds, or the difficulty band and the "
      f"reward both read high for the wrong reason.\n")
    p("## Checker self-consistency\n")
    p(f"`answer_correct(gold, gold)` is True for **{sc_ok}/{sc_n}** sampled golds.\n")
    p("## 20 most common golds\n")
    p("| n | gold | shape | strict |")
    p("|---:|---|---|:--:|")
    for g, n in common:
        p(f"| {n} | {g} | `{answer_shape(g)}` | {'yes' if answer_is_scorable(g) else 'no'} |")
    p("")
    p(f"## 15 random REJECTED golds (strict) — {len(rejected):,} total\n")
    for g in random.Random(1).sample(rejected, min(15, len(rejected))) if rejected else []:
        p(f"- `{answer_shape(g)}` — {g}")
    p("")
    p(f"## 15 random REJECTED golds (widened) — {len(rejected_wide):,} total\n")
    for g in random.Random(2).sample(rejected_wide, min(15, len(rejected_wide))) if rejected_wide else []:
        p(f"- `{answer_shape(g)}` — {g}")
    p("")
    p("## 10 accepted golds with their extracted answer key\n")
    from src.data import answer_tail

    for g in random.Random(3).sample(strict, min(10, len(strict))) if strict else []:
        p(f"- {g}  ->  key `{answer_tail(g)}` (`{answer_shape(g)}`)")

    report = "\n".join(L) + "\n"
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)

    print(report)
    print(f"[report written to {out}]")

    if sc_ok < sc_n:
        print(f"FAIL: checker is not self-consistent ({sc_ok}/{sc_n}).", file=sys.stderr)
        return 1
    if f_strict >= THRESHOLD:
        print(f"PASS: strict {f_strict:.4f} >= {THRESHOLD}")
        return 0
    print(f"strict {f_strict:.4f} < {THRESHOLD} — falling back to widened containment (GOAL S9)")
    if f_wide >= THRESHOLD:
        print(f"PASS: widened {f_wide:.4f} >= {THRESHOLD} (strict was {f_strict:.4f})")
        return 0
    print(f"FAIL: strict {f_strict:.4f} AND widened {f_wide:.4f} both < {THRESHOLD}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
