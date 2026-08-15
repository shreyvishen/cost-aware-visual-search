"""Evaluate the BASE model on V*Bench, through the exact same path A and B were evaluated on.

Without this, the headline table has no accuracy for the untrained model, and "training helped"
is an assertion rather than a measurement. Writes the same artifacts a training run writes, so
`compare_runs.py` and the web app read it with no special cases.

    python -m scripts.eval_base --out /srv/ai/runs/base
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src import data as data_mod  # noqa: E402
from src import rollout  # noqa: E402
from src.backends import get_backend  # noqa: E402
from src.reward import CostModel  # noqa: E402
from src.train import score_episodes, summarize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/srv/ai/runs/base")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--never-zoom-n", type=int, default=96)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "eval").mkdir(parents=True, exist_ok=True)

    # Identical geometry to run_a / run_b, read from run_a's own config so it cannot drift.
    cfg_src = json.loads((Path("/srv/ai/runs/run_a/config.json")).read_text())
    cfg = {k: cfg_src[k] for k in
           ("max_zooms", "max_new_tokens", "downsample", "thumb_max_side", "crop_max_side")}
    print("geometry:", cfg)

    samples = data_mod.load_vstar(limit=args.limit)
    print(f"loaded {len(samples)} V*Bench samples")
    options = {s.sid: s.options for s in samples}

    backend = get_backend("hf", model=None, device=args.device)
    cost = CostModel(mode="none")

    def run(tag: str, subset, max_zooms: int) -> dict:
        c = dict(cfg); c["max_zooms"] = max_zooms
        t0 = time.perf_counter()
        eps = rollout.collect(backend, backend.proc, backend.tok, subset, 1, c, temperature=0.0)
        score_episodes(eps, cost, 0.0, options)
        row = summarize(eps)
        row.update({"phase": f"eval:{tag}", "step": 0,
                    "t_eval_s": round(time.perf_counter() - t0, 1)})
        with open(out / "eval" / f"vstar_predictions_{tag}.jsonl", "w") as f:
            for e in eps:
                d = e.to_json()
                d["boxes"] = getattr(e, "meta_boxes", [])
                f.write(json.dumps(d) + "\n")
        print(f"eval[{tag}] n={row['n']} acc={row['accuracy']:.4f} "
              f"zooms={row['mean_zooms']:.2f} decode={row['mean_decode_tokens']:.1f} "
              f"({row['t_eval_s']:.0f}s)")
        return row

    rows = [run("final", samples, cfg["max_zooms"]),
            run("never_zoom", samples[: args.never_zoom_n], 0)]

    with open(out / "metrics.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "config.json").write_text(json.dumps(
        {"run_id": "BASE", "cost_mode": "none", "note": "untrained base model, same geometry",
         **cfg}, indent=2))
    (out / "DONE").write_text(json.dumps({"finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
