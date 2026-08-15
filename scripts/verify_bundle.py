"""Definition of Done checker (GOAL §2).

The run is not done until these are on THIS Mac, sufficient to build the demo and prove B > A
without touching the rig:

    Run A — adapter (best + last) + metrics.jsonl + sampled rollouts/ + sampled crops/
            + eval/vstar_predictions*.jsonl + config.json
    Run B — the same

Exits nonzero if anything required is missing, and says exactly what. Run it as often as you
like; it only reads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ARCHIVE = Path.home() / "archive" / "cost-aware-vlm"

#: (relative path or glob, what it is, required?)
REQUIRED = [
    ("config.json", "run config + frozen cost model", True),
    ("metrics.jsonl", "per-step accuracy / tool rate / cost", True),
    ("adapters/last/adapter_model.safetensors", "final LoRA adapter", True),
    ("adapters/best/adapter_model.safetensors", "best LoRA adapter", False),
    ("rollouts/*.jsonl", "sampled rollouts (raw material for the demo)", True),
    ("crops/*/*_thumb.jpg", "the 'where it looked' thumbnails", True),
    ("eval/vstar_predictions_final.jsonl", "final V*Bench predictions", True),
    ("DONE", "run completed its timer and finalised", False),
]


def check_run(root: Path) -> tuple[bool, list[str]]:
    ok = True
    out = []
    if not root.exists():
        return False, [f"  MISSING the whole run directory {root}"]
    for pat, what, required in REQUIRED:
        hits = sorted(root.glob(pat)) if any(c in pat for c in "*?") else (
            [root / pat] if (root / pat).exists() else [])
        hits = [h for h in hits if h.exists()]
        if hits:
            size = sum(h.stat().st_size for h in hits)
            out.append(f"  ok       {pat:<48} {what} "
                       f"({len(hits)} file(s), {size/1e6:.1f} MB)")
        elif required:
            ok = False
            out.append(f"  MISSING  {pat:<48} {what}")
        else:
            out.append(f"  absent   {pat:<48} {what} (not required)")
    return ok, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(ARCHIVE))
    ap.add_argument("--runs", nargs="*", default=["run_a", "run_b"])
    args = ap.parse_args()
    arch = Path(args.archive)

    print(f"Definition of Done — checking {arch}\n")
    all_ok = True
    for name in args.runs:
        root = arch / name
        ok, lines = check_run(root)
        state = "COMPLETE" if ok else "INCOMPLETE"
        print(f"{name}: {state}")
        print("\n".join(lines))
        # a quick readout of what the run actually achieved
        m = root / "metrics.jsonl"
        if m.exists():
            rows = [json.loads(x) for x in m.read_text().splitlines() if x.strip()]
            trains = [r for r in rows if r.get("phase") == "train"]
            finals = [r for r in rows if r.get("phase") == "eval:final"]
            print(f"  -> {len(trains)} train steps logged"
                  + (f"; final eval acc={finals[-1]['accuracy']:.3f} "
                     f"tool_rate={finals[-1]['tool_rate']:.3f}" if finals else
                     "; no final eval yet"))
        print()
        all_ok = all_ok and ok

    if all_ok:
        print("ALL REQUIRED ARTIFACTS PRESENT. B > A can be evaluated offline from this Mac.")
        return 0
    print("NOT DONE — see MISSING lines above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
