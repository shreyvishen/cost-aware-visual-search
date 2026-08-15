#!/usr/bin/env python3
"""Fit cost_ms = intercept + a*vision_tokens + b*decode_tokens + c*tool_calls.

Ordinary least squares, numpy only. Fits twice -- with and without the intercept
-- reports both, and ships whichever scores the better R^2. `intercept_ms` stays
in the json either way; the reward uses only a, b, c.

Usage:
    python3 fit.py --quant q4          # writes coeffs_q4.json
    python3 fit.py --all               # every quant present in the csv
"""

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "m4_measurements.csv")

FORMULA_NOTE = (
    "UNITS: vision_tokens come from cost_model/vision_tokens.py::vision_tokens_for, "
    "which reproduces the rig's HF Qwen2VLImageProcessor exactly -- Qwen smart_resize "
    "with factor=patch16*merge2=32, min_pixels=65536 (floor 64 tokens), "
    "max_pixels=16777216, banker's rounding; tokens=(w_bar/32)*(h_bar/32). Checked "
    "against 28 sizes run through the real processor at "
    "/srv/ai/models/current/qwen35-4b, 28/28 exact. Training MUST import that "
    "function or the reward units are silently wrong. NOTE llama.cpp's clip uses "
    "DIFFERENT clamps (min_pixels=9216, max_pixels=4194304, half-away rounding) -- "
    "vision_tokens_llamacpp covers that, and it is only for predicting Mac demo "
    "latency. The two rules agree exactly on every image size in this fit "
    "(256/384/512/768/1024 square), so a is in HF units across the measured range."
)


def load(quant, split="fit"):
    """Timed rows for one quant. rep 0 is the discarded per-config warm-up.

    split="fit" returns the design points, split="val" the held-out ones.
    Validation config_ids start with "v" and never enter a fit.
    """
    rows = []
    with open(CSV_PATH, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["quant"] != quant or int(r["is_warmup"]):
                continue
            is_val = r["config_id"].startswith("v")
            if is_val != (split == "val"):
                continue
            rows.append(r)
    if not rows:
        raise SystemExit(f"no timed rows for quant={quant}")
    X = np.array([[float(r["vision_tokens"]), float(r["decode_tokens"]),
                   float(r["tool_calls"])] for r in rows])
    y = np.array([float(r["total_ms"]) for r in rows])
    return rows, X, y


def ols(X, y, intercept=True):
    A = np.hstack([np.ones((len(X), 1)), X]) if intercept else X
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta
    resid = y - pred
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    n, k = A.shape
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else float("nan")
    rmse = float(np.sqrt(ss_res / n))
    if intercept:
        b0, a, b, c = beta
    else:
        b0, (a, b, c) = 0.0, beta
    return {
        "intercept_ms": float(b0), "a": float(a), "b": float(b), "c": float(c),
        "r2": r2, "adj_r2": adj, "rmse_ms": rmse,
        "mape_pct": float(np.mean(np.abs(resid / y)) * 100.0),
        "max_abs_resid_ms": float(np.abs(resid).max()),
        "cond": float(np.linalg.cond(A)),
    }


def vifs(X):
    """Variance inflation per regressor. >10 means a, b, c are not separable."""
    out = []
    for j in range(X.shape[1]):
        others = np.hstack([np.ones((len(X), 1)), np.delete(X, j, axis=1)])
        beta, *_ = np.linalg.lstsq(others, X[:, j], rcond=None)
        resid = X[:, j] - others @ beta
        ss_tot = float(((X[:, j] - X[:, j].mean()) ** 2).sum())
        r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
        out.append(float("inf") if r2 >= 1.0 else 1.0 / (1.0 - r2))
    return out


def coeff_sha(a, b, c, intercept):
    payload = json.dumps([a, b, c, intercept], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def fit_quant(quant, write=True):
    rows, X, y = load(quant)
    with_i = ols(X, y, True)
    no_i = ols(X, y, False)
    shipped_name = "with_intercept" if with_i["r2"] >= no_i["r2"] else "no_intercept"
    shipped = with_i if shipped_name == "with_intercept" else no_i

    a, b, c = shipped["a"], shipped["b"], shipped["c"]
    # Keep the fitted intercept in the json even when the no-intercept fit wins,
    # so a reader can reconstruct the wall-clock prediction.
    intercept = with_i["intercept_ms"] if shipped_name == "no_intercept" \
        else shipped["intercept_ms"]

    v = vifs(X)
    notes = [
        f"Shipped the {shipped_name.replace('_', '-')} fit (R^2 "
        f"{shipped['r2']:.5f} vs {(no_i if shipped_name=='with_intercept' else with_i)['r2']:.5f}).",
        f"Measured on an M4 Max via llama-server (libllama+libmtmd, Metal, "
        f"-ngl 99), NOT llama-mtmd-cli: this build's mtmd-cli emits no "
        f"llama_perf block. Response variable = prompt_ms + predicted_ms from "
        f"the server timings object (image encode + prefill + decode); process "
        f"startup excluded.",
        f"RMSE {shipped['rmse_ms']:.1f} ms, MAPE {shipped['mape_pct']:.2f}%, "
        f"max |residual| {shipped['max_abs_resid_ms']:.1f} ms, VIF "
        f"(vision, decode, tool) = "
        f"({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}).",
        FORMULA_NOTE,
        "Reduced grid, not the full 240-run GOAL sec.18 grid; see fit_report.md "
        "for the exact points dropped. Measurements were serial by design "
        "(GOAL sec.19 fan-out deliberately not applied -- concurrent Metal work "
        "would poison every timing). Other work was running on the Mac.",
    ]
    if shipped["r2"] < 0.90:
        notes.insert(0, f"WARNING: R^2 = {shipped['r2']:.4f} is BELOW 0.90. The "
                        f"linear cost model does not explain this hardware well. "
                        f"Shipped anyway because training needs coefficients.")

    out = {
        "quant": quant,
        "a_ms_per_vision_token": a,
        "b_ms_per_decode_token": b,
        "c_ms_per_tool_call": c,
        "intercept_ms": intercept,
        "r2": shipped["r2"],
        "n_points": len(rows),
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": coeff_sha(a, b, c, intercept),
        "notes": " ".join(notes),
        "fit_detail": {
            "shipped": shipped_name,
            "with_intercept": with_i,
            "no_intercept": no_i,
            "vif": {"vision_tokens": v[0], "decode_tokens": v[1],
                    "tool_calls": v[2]},
            "n_configs": len({r["config_id"] for r in rows}),
            "reps_per_config_timed": len(rows) // max(
                1, len({r["config_id"] for r in rows})),
            "ranges": {
                "vision_tokens": [float(X[:, 0].min()), float(X[:, 0].max())],
                "decode_tokens": [float(X[:, 1].min()), float(X[:, 1].max())],
                "tool_calls": [float(X[:, 2].min()), float(X[:, 2].max())],
                "total_ms": [float(y.min()), float(y.max())],
            },
        },
    }
    # Score the frozen coefficients on the held-out points, if any exist. This
    # is the honest test of the reduced grid.
    try:
        vrows, VX, vy = load(quant, split="val")
    except SystemExit:
        vrows = None
    if vrows:
        pred = intercept + VX @ np.array([a, b, c])
        err = vy - pred
        ss_res = float(err @ err)
        ss_tot = float(((vy - vy.mean()) ** 2).sum())
        out_val = {
            "n_points": len(vrows),
            "n_configs": len({r["config_id"] for r in vrows}),
            "r2": 1.0 - ss_res / ss_tot,
            "rmse_ms": float(np.sqrt(ss_res / len(vy))),
            "mape_pct": float(np.mean(np.abs(err / vy)) * 100.0),
            "max_abs_err_ms": float(np.abs(err).max()),
            "mean_signed_err_ms": float(err.mean()),
        }
        out["validation"] = out_val
        notes.append(
            f"Held-out check on {out_val['n_configs']} grid cells the fit never "
            f"saw ({out_val['n_points']} timed runs): R^2 {out_val['r2']:.5f}, "
            f"MAPE {out_val['mape_pct']:.2f}%, RMSE {out_val['rmse_ms']:.1f} ms, "
            f"mean signed error {out_val['mean_signed_err_ms']:+.1f} ms.")
        out["notes"] = " ".join(notes)

    if write:
        path = os.path.join(HERE, f"coeffs_{quant}.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        print(f"wrote {path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        seen = []
        with open(CSV_PATH, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["quant"] not in seen:
                    seen.append(r["quant"])
        quants = seen
    else:
        quants = [args.quant]

    for q in quants:
        o = fit_quant(q)
        d = o["fit_detail"]
        print(f"{q}: a={o['a_ms_per_vision_token']:.6f} "
              f"b={o['b_ms_per_decode_token']:.6f} "
              f"c={o['c_ms_per_tool_call']:.4f} "
              f"int={o['intercept_ms']:.2f} R2={o['r2']:.5f} n={o['n_points']} "
              f"[shipped {d['shipped']}, with_i R2={d['with_intercept']['r2']:.5f}, "
              f"no_i R2={d['no_intercept']['r2']:.5f}]")


if __name__ == "__main__":
    main()
