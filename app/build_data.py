"""Build app/static/data.json — the single source for the page.

Everything is derived from the run artifacts, so the page cannot drift from the data.
Only what earns its place goes in: no tool-call rate, no box-validity, no Run C.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARC = Path.home() / "archive" / "cost-aware-vlm"
OUT = REPO / "app" / "static" / "data.json"

# Token pricing the user asked for, USD per million tokens.
PRICE_IN, PRICE_OUT = 0.03, 0.15


def jl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def cost_usd(prefill: float, decode: float) -> float:
    return (prefill * PRICE_IN + decode * PRICE_OUT) / 1e6


def main() -> None:
    m4 = jl(REPO / "eval" / "m4_latency.jsonl")

    def tag(t, q="q4"):
        return [r for r in m4 if r.get("tag") == t and r.get("quant") == q]

    # ---- headline: base vs A vs B on the same 36 images, M4 @ Q4 ----------------
    models = [("base", "Base", "base_q4"), ("a", "Run A", "run_a_final"),
              ("b", "Run B", "run_b_final")]
    headline = []
    for key, label, t in models:
        d = tag(t)
        if not d:
            continue
        pf = st.mean(x["prompt_n_total"] for x in d)
        dc = st.mean(x["decode_tokens"] for x in d)
        headline.append({
            "key": key, "label": label, "n": len(d),
            "prefill_tokens": round(pf, 1), "decode_tokens": round(dc, 1),
            "prefill_ms": round(st.mean(x["prefill_ms"] for x in d)),
            "decode_ms": round(st.mean(x["decode_ms"] for x in d)),
            "latency_ms": round(st.mean(x["actual_ms"] for x in d)),
            "zooms": round(st.mean(x["tool_calls"] for x in d), 2),
            "usd_per_1k": round(cost_usd(pf, dc) * 1000, 4),
        })

    # ---- rig accuracy, the only accuracy worth quoting (191 images) -------------
    acc = {}
    for key, run in (("a", "run_a"), ("b", "run_b")):
        fin = jl(ARC / run / "eval" / "vstar_predictions_final.jsonl")
        nz = jl(ARC / run / "eval" / "vstar_predictions_never_zoom.jsonl")
        if not fin:
            continue
        nz_ids = {r["sid"] for r in nz}
        sub = [r for r in fin if r["sid"] in nz_ids]
        acc[key] = {
            "full_n": len(fin),
            "full": round(sum(r["correct"] for r in fin) / len(fin), 4),
            "ref_n": len(sub),
            "ref_zoom": round(sum(r["correct"] for r in sub) / len(sub), 4) if sub else None,
            "ref_nozoom": round(sum(r["correct"] for r in nz) / len(nz), 4) if nz else None,
            "zooms": round(st.mean(r["n_tool_calls"] for r in fin), 2),
        }

    # ---- the 2x2, and two worked samples from each cell -------------------------
    fa = {r["sid"]: r for r in jl(ARC / "run_a" / "eval" / "vstar_predictions_final.jsonl")}
    fb = {r["sid"]: r for r in jl(ARC / "run_b" / "eval" / "vstar_predictions_final.jsonl")}
    shared = sorted(set(fa) & set(fb))
    cells = {"both_right": [], "only_a": [], "only_b": [], "both_wrong": []}
    for s in shared:
        a, b = fa[s], fb[s]
        k = ("both_right" if a["correct"] and b["correct"] else
             "only_a" if a["correct"] else
             "only_b" if b["correct"] else "both_wrong")
        cells[k].append(s)
    samples = {}
    for k, ids in cells.items():
        picked = []
        for s in ids[:2]:
            a, b = fa[s], fb[s]
            picked.append({
                "sid": s, "question": a["question"], "gold": a["gold"],
                "a": {"answer": a["answer"], "correct": a["correct"],
                      "zooms": a["n_tool_calls"], "decode": a["decode_tokens"],
                      "boxes": a.get("boxes", [])},
                "b": {"answer": b["answer"], "correct": b["correct"],
                      "zooms": b["n_tool_calls"], "decode": b["decode_tokens"],
                      "boxes": b.get("boxes", [])},
            })
        samples[k] = picked
    matrix = {k: len(v) for k, v in cells.items()}

    # ---- training curves: reward, accuracy, zooms, lambda per step --------------
    curves = {}
    for key, run in (("a", "run_a"), ("b", "run_b")):
        rows = [r for r in jl(ARC / run / "metrics.jsonl") if r.get("phase") == "train"]
        curves[key] = [{
            "step": r["step"], "reward": round(r["mean_reward"], 4),
            "reward_std": round(r.get("reward_std", 0), 4),
            "accuracy": round(r["accuracy"], 4),
            "zooms": round(r["mean_zooms"], 3),
            "decode": round(r["mean_decode_tokens"], 1),
            "lam": r.get("lam", 0.0),
            "groups_used": r.get("groups_used"), "groups_total": r.get("groups_total"),
            "kl": round(r.get("kl", 0), 5),
        } for r in rows]

    # ---- hyperparameters: shared, then what differs ----------------------------
    ca = json.loads((ARC / "run_a" / "config.json").read_text())
    cb = json.loads((ARC / "run_b" / "config.json").read_text())
    shared_keys = ["group_size", "prompts_per_step", "max_zooms", "max_new_tokens",
                   "temperature", "lora_rank", "lr", "kl_coef", "clip_eps",
                   "max_grad_norm", "downsample", "thumb_max_side", "crop_max_side",
                   "seed", "minutes", "train_pool", "pool_filter"]
    hyper = {"shared": {k: ca.get(k) for k in shared_keys},
             "differs": {
                 "a": {"cost_mode": ca.get("cost_mode"), "lambda": ca.get("lam_target"),
                       "steps": len(curves.get("a", []))},
                 "b": {"cost_mode": cb.get("cost_mode"), "lambda": cb.get("lam_target"),
                       "coeffs": cb.get("cost_model"), "steps": len(curves.get("b", []))}}}

    # ---- quantization sweep: whatever has landed -------------------------------
    quant = []
    for q in ("q4", "q8", "bf16"):
        for key, label, t in [("base", "Base", f"base_{q}"),
                              ("a", "Run A", "run_a_final" if q == "q4" else f"run_a_{q}_final"),
                              ("b", "Run B", "run_b_final" if q == "q4" else f"run_b_{q}_final")]:
            d = tag(t, q)
            if not d:
                continue
            pf = st.mean(x["prompt_n_total"] for x in d)
            dc = st.mean(x["decode_tokens"] for x in d)
            quant.append({
                "quant": q, "model": key, "label": label, "n": len(d),
                "prefill_tokens": round(pf, 1), "decode_tokens": round(dc, 1),
                "latency_ms": round(st.mean(x["actual_ms"] for x in d)),
                "zooms": round(st.mean(x["tool_calls"] for x in d), 2),
                "usd_per_1k": round(cost_usd(pf, dc) * 1000, 4),
            })

    coeffs = json.loads((REPO / "cost_model" / "coeffs_q4.json").read_text())
    data = {
        "headline": headline, "accuracy": acc, "matrix": matrix, "samples": samples,
        "curves": curves, "hyper": hyper, "quant": quant,
        "pricing": {"in_per_m": PRICE_IN, "out_per_m": PRICE_OUT},
        "coeffs": {"a": coeffs["a_ms_per_vision_token"], "b": coeffs["b_ms_per_decode_token"],
                   "c": coeffs["c_ms_per_tool_call"], "r2": coeffs["r2"]},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"  headline models : {[h['key'] for h in headline]}")
    print(f"  2x2             : {matrix}")
    print(f"  curves          : A {len(curves.get('a',[]))} steps, B {len(curves.get('b',[]))} steps")
    print(f"  quant rows      : {len(quant)}")


if __name__ == "__main__":
    main()
