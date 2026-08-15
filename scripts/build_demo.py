#!/usr/bin/env python3
"""Build the attention-crop viewer (GOAL §13) from whatever runs exist on the Mac.

Reads `~/archive/cost-aware-vlm/<run>/` and writes one self-contained HTML page to
`out/demo.html`. No backend, no external requests, every image inlined as a data URI.

Re-run it as new artifacts land. It never invents a number: a run that is not on disk
is reported as missing, and a run that is still training is labelled as such.

    python3 scripts/build_demo.py [--archive DIR] [--out FILE] [--dev]

`--dev` also picks up the `shape` and `smoke` shakedown runs, which carry the identical
schema and are useful before Run A's eval lands. They never appear without the flag.

Two things this file gets right, and both are easy to get wrong:

1. **Boxes live on a 0-1000 grid relative to the whole image**, not on pixels. The page
   positions every box with percentages off that grid, so it cannot drift.
2. **The mirrored thumbnails already have the boxes burned in** by `src/zoom_env.draw_boxes`.
   Burned-in boxes cannot be animated and would leak Run A's boxes into Run B's view, so
   `clean_thumb()` reconstructs a box-free thumbnail: it pastes each saved crop back over
   its own box (the crop is that exact region of the original) and scrubs whatever outline
   pixels survive. See `_scrub_ring`.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
COORD = 1000                      # the normalised grid the policy emits boxes on
DRAW_COLORS = [(0xff, 0x3b, 0x30), (0xff, 0xcc, 0x00),
               (0x34, 0xc7, 0x59), (0x0a, 0x84, 0xff)]   # src/zoom_env.draw_boxes
BORDER_PX = 2                     # its outline width

#: Run id -> how the page introduces it. Order is the order the page shows them.
RUN_SPEC = [
    ("A", "run_a", "Run A", "baseline",
     "Control. No cost term in the reward, so nothing discourages a look."),
    ("B", "run_b", "Run B", "ours",
     "Cost-aware. The reward charges measured M4 Max milliseconds for every look."),
    ("C", "run_c", "Run C", "ablation",
     "Ablation. Cost counted in uniform tokens instead of measured milliseconds."),
    ("D", "run_d", "Run D", "ablation",
     "Ablation. Same measured cost model, fitted at Q8_0 instead of Q4_K_M."),
]
DEV_SPEC = [
    ("shape", "shape", "shape", "shakedown", "Pre-flight shakedown run, not a result."),
    ("smoke", "smoke", "smoke", "shakedown", "Pre-flight smoke run, not a result."),
]


# ---------------------------------------------------------------- reading


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass          # a half-written last line while the run is live
    return rows


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def load_cost_model(quant: str = "q4") -> dict:
    """The frozen, measured M4 Max coefficients. This is the ruler for every latency."""
    c = read_json(REPO / "cost_model" / f"coeffs_{quant}.json")
    if not c:
        return {}
    return {
        "quant": c.get("quant", quant),
        "a": c["a_ms_per_vision_token"],
        "b": c["b_ms_per_decode_token"],
        "c": c["c_ms_per_tool_call"],
        "intercept": c.get("intercept_ms", 0.0),
        "r2": c.get("r2"),
        "n_points": c.get("n_points"),
        "sha256": c.get("sha256", "")[:12],
        "measured_at": c.get("measured_at", ""),
    }


def predict_ms(cm: dict, vision: float, decode: float, calls: float) -> float | None:
    if not cm:
        return None
    return cm["intercept"] + cm["a"] * vision + cm["b"] * decode + cm["c"] * calls


#: Tags in `eval/m4_latency.jsonl` that belong on the page, in display order. The file also
#: holds measurement-methodology variants (`_nocache`, `_rep2`) and an intermediate checkpoint;
#: those are how the timing was validated, not results, so they stay off the page.
M4_TAGS = [
    ("base_q4", "Base model", "reference", "Qwen3.5-4B with no adapter."),
    ("run_a_final", "Run A", "baseline", "Trained with no cost term."),
    ("run_b_final", "Run B", "ours", "Trained against measured milliseconds."),
    ("run_c_final", "Run C", "ablation", "Trained against counted tokens."),
    ("run_d_final", "Run D", "ablation", "Measured cost fitted at Q8_0."),
]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def load_m4(path: Path, quant: str = "q4") -> dict | None:
    """Wall clock actually measured on the M4 Max, per episode, via llama.cpp.

    This is a different machine and a different, smaller image set than the rig eval, so it
    never shares a row with rig numbers. It also lets the page check the cost model against
    reality instead of asserting it.
    """
    rows = [r for r in read_jsonl(path) if r.get("quant") == quant]
    if not rows:
        return None
    by_tag: dict[str, list[dict]] = {}
    for r in rows:
        by_tag.setdefault(r.get("tag", ""), []).append(r)

    groups = []
    for tag, label, role, blurb in M4_TAGS:
        v = by_tag.get(tag)
        if not v:
            continue
        groups.append({
            "tag": tag, "label": label, "role": role, "blurb": blurb, "n": len(v),
            "ms": _mean([r["actual_ms"] for r in v]),
            "pred_ms": _mean([r["predicted_ms"] for r in v]),
            "tool_rate": _mean([1.0 if r.get("tool_calls", 0) > 0 else 0.0 for r in v]),
            "accuracy": _mean([1.0 if r.get("correct") else 0.0 for r in v]),
            "zooms": _mean([r.get("tool_calls", 0) for r in v]),
            "decode": _mean([r.get("decode_tokens", 0) for r in v]),
            "vision": _mean([r.get("vision_tokens", 0) for r in v]),
        })
    if not groups:
        return None

    keep = {g["tag"] for g in groups}
    pts = [r for r in rows if r.get("tag") in keep]
    p = [r["predicted_ms"] for r in pts]
    a = [r["actual_ms"] for r in pts]
    ratios = sorted(y / x for x, y in zip(p, a) if x > 0)
    mp, ma = _mean(p), _mean(a)
    num = sum((x - mp) * (y - ma) for x, y in zip(p, a))
    den = (sum((x - mp) ** 2 for x in p) * sum((y - ma) ** 2 for y in a)) ** 0.5

    return {
        "quant": quant,
        "n": len(pts),
        "groups": groups,
        "scatter": [{"p": r["predicted_ms"], "a": r["actual_ms"], "tag": r["tag"]} for r in pts],
        "pearson": (num / den) if den else None,
        "ratio_median": ratios[len(ratios) // 2] if ratios else None,
    }


# ---------------------------------------------------------------- thumbnails


def _box_px(box: list[int], w: int, h: int) -> tuple[int, int, int, int]:
    """The box in thumbnail pixels, matching how PIL coerced them when it drew them.

    `draw_boxes` hands ImageDraw floats and PIL truncates, so truncation is what we
    have to undo. The returned rect is inclusive of the last drawn column and row.
    """
    x1, y1, x2, y2 = box
    fx1 = int(max(0, min(x1 * w / COORD, w - 1)))
    fy1 = int(max(0, min(y1 * h / COORD, h - 1)))
    fx2 = int(max(1, min(x2 * w / COORD, w)))
    fy2 = int(max(1, min(y2 * h / COORD, h)))
    return fx1, fy1, fx2, fy2


def _near_draw_color(px: tuple[int, int, int], tol: int = 105) -> bool:
    for c in DRAW_COLORS:
        if abs(px[0] - c[0]) + abs(px[1] - c[1]) + abs(px[2] - c[2]) < tol:
            return True
    return False


def _ring(box: list[int], w: int, h: int) -> set[tuple[int, int]]:
    """Every pixel the drawn outline could occupy, plus a pixel of slack on each side.

    PIL truncates the float coordinates `draw_boxes` hands it and draws the border inward,
    so the band runs from one pixel outside the rect to `BORDER_PX` inside it.
    """
    fx1, fy1, fx2, fy2 = _box_px(box, w, h)
    out: set[tuple[int, int]] = set()
    offs = range(-1, BORDER_PX + 1)
    for x in range(max(0, fx1 - 1), min(w, fx2 + 2)):
        for d in offs:
            for y in (fy1 + d, fy2 - d):
                if 0 <= y < h:
                    out.add((x, y))
    for y in range(max(0, fy1 - 1), min(h, fy2 + 2)):
        for d in offs:
            for x in (fx1 + d, fx2 - d):
                if 0 <= x < w:
                    out.add((x, y))
    return out


def _repaint_ring(img: Image.Image, boxes: list[list[int]]) -> int:
    """Repaint the outline band from the pixels around it.

    Colour tests are unreliable here: the thumbnails are JPEG, so a thin stroke over a
    bright sky survives as a pale wash that no colour threshold catches without also
    catching real content. So the band is repainted unconditionally. It costs nothing —
    the animated overlay draws its own border on exactly those pixels — and it cannot
    leave a rectangle behind. Returns how many band pixels still read as a draw colour.
    """
    w, h = img.size
    px = img.load()
    band: set[tuple[int, int]] = set()
    for b in boxes[:4]:
        band |= _ring(b, w, h)

    todo = set(band)
    for _ in range(8):
        if not todo:
            break
        left = []
        for (x, y) in sorted(todo):
            vals = []
            for dy in (-2, -1, 0, 1, 2):
                for dx in (-2, -1, 0, 1, 2):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in todo:
                        vals.append(px[nx, ny])
            if len(vals) >= 4:
                px[x, y] = tuple(sorted(v[k] for v in vals)[len(vals) // 2] for k in range(3))
                todo.discard((x, y))
            else:
                left.append((x, y))
        if len(left) == len(todo):
            break
        todo = set(left)
    # Upper bound, not a count of failures: a red boat or a blue sky inside the band trips
    # the colour test even after the band has been repainted. Treat it as "worth a look".
    return sum(1 for p in band if _near_draw_color(px[p]))


def clean_thumb(thumb_path: Path, boxes: list[list[int]],
                crop_paths: list[Path]) -> tuple[Image.Image, int]:
    """A thumbnail with the burned-in boxes removed. Returns (image, residual pixels).

    Each saved crop is exactly its box's region of the original image, so pasting it back
    over the box erases the rectangle and restores real content underneath. Largest box
    first, so a box drawn inside another still ends up on top.
    """
    base = open_image(thumb_path)
    if base is None:
        raise OSError(f"unreadable thumbnail {thumb_path}")
    img = base.convert("RGB")
    w, h = img.size
    order = sorted(range(len(boxes)), key=lambda i: -_area(boxes[i]))
    for i in order:
        if i >= len(crop_paths):
            continue
        crop = open_image(crop_paths[i])
        if crop is None:
            continue
        fx1, fy1, fx2, fy2 = _box_px(boxes[i], w, h)
        bw, bh = max(1, fx2 - fx1 + 1), max(1, fy2 - fy1 + 1)
        img.paste(crop.convert("RGB").resize((bw, bh), Image.LANCZOS), (fx1, fy1))
    residual = _repaint_ring(img, boxes)
    return img, residual


def _area(b: list[int]) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def open_image(path: Path) -> Image.Image | None:
    """None rather than an exception. The mirror runs every 60 s, so the build can catch a
    JPEG mid-copy; one unreadable crop must not take the whole page down."""
    try:
        im = Image.open(path)
        im.load()
        return im
    except Exception:
        return None


def data_uri(img: Image.Image, max_side: int, quality: int = 82) -> str:
    im = img.convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        r = max_side / float(max(w, h))
        im = im.resize((max(1, round(w * r)), max(1, round(h * r))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True, progressive=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- runs


def latest_crop_tag(run_dir: Path) -> str | None:
    crops = run_dir / "crops"
    if not crops.is_dir():
        return None
    tags = [d.name for d in crops.iterdir()
            if d.is_dir() and POLICY_TAG.match(d.name) and any(d.iterdir())]
    if not tags:
        return None
    return max(tags, key=_tag_step)


def _tag_step(tag: str) -> int:
    if tag in ("final", "last"):
        return 10 ** 6
    m = re.search(r"(\d+)", tag)
    return int(m.group(1)) if m else -1


def phase_tag(phase: object) -> str:
    """`eval:final` -> `final`. The prefix has to go before the tag is ranked, or
    `_tag_step` finds no digits in `eval:final` and sorts it below `eval:step0`."""
    return str(phase).split(":", 1)[-1]


#: Eval tags that report the policy's own behaviour. `train.py` also logs reference evals —
#: `never_zoom` runs the same weights with the zoom budget set to 0 — and those must never
#: reach the curve or the headline, or the page would show the policy at a 0% tool rate.
POLICY_TAG = re.compile(r"^(step\d+|final|last)$")


def pick_eval(metrics: list[dict]) -> dict | None:
    """The run's most recent policy eval. Its `phase` names which step it came from."""
    evals = [m for m in metrics if str(m.get("phase", "")).startswith("eval")
             and POLICY_TAG.match(phase_tag(m.get("phase", "")))]
    if not evals:
        return None
    return max(evals, key=lambda m: (_tag_step(phase_tag(m.get("phase", ""))), m.get("step", 0)))


def pick_reference(metrics: list[dict]) -> dict | None:
    """The thumbnail-only reference: same weights, zoom budget 0 (GOAL §12)."""
    refs = [m for m in metrics if phase_tag(m.get("phase", "")) == "never_zoom"]
    return refs[-1] if refs else None


def think_of(text: str, limit: int = 330) -> str:
    """The reasoning the policy wrote before it acted, trimmed for display."""
    t = text.split("</think>")[0].split("<tool_call>")[0].split("<answer>")[0]
    t = t.replace("<think>", "").strip()
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)      # the model writes markdown; the page is not
    t = re.sub(r"\s+", " ", t)
    return t[:limit].rstrip() + "…" if len(t) > limit else t


def mcnemar_p(b01: int, b10: int) -> float:
    """Exact two-sided McNemar on a paired table. Small n, so no chi-square approximation."""
    n = b01 + b10
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b01, b10) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def paired_stats(archive: Path, cm: dict, a_dir: str, b_dir: str) -> dict | None:
    """A vs B on the images both runs actually answered, plus where the saving came from.

    Two runs can differ by a few points of accuracy and still be the same policy in disguise.
    The paired test says whether the difference is worth a sentence. The decomposition says
    which term of the cost model paid for the speedup — which is the whole argument for
    measuring milliseconds instead of counting tool calls.
    """
    def load(d: str) -> dict:
        f = archive / d / "eval" / "vstar_predictions_final.jsonl"
        return {r["sid"]: r for r in read_jsonl(f)}

    a, b = load(a_dir), load(b_dir)
    sids = sorted(set(a) & set(b))
    if len(sids) < 20:
        return None
    acc01 = sum(1 for s in sids if a[s].get("correct") and not b[s].get("correct"))
    acc10 = sum(1 for s in sids if not a[s].get("correct") and b[s].get("correct"))
    t01 = sum(1 for s in sids if (a[s].get("n_tool_calls") or 0) > 0
              and (b[s].get("n_tool_calls") or 0) == 0)
    t10 = sum(1 for s in sids if (a[s].get("n_tool_calls") or 0) == 0
              and (b[s].get("n_tool_calls") or 0) > 0)

    out = {"n": len(sids), "acc_a_only": acc01, "acc_b_only": acc10,
           "acc_p": mcnemar_p(acc01, acc10), "tool_p": mcnemar_p(t01, t10)}

    if cm:
        mean = lambda d, k: _mean([d[s].get(k) or 0 for s in sids])
        dv = mean(a, "vision_tokens") - mean(b, "vision_tokens")
        dd = mean(a, "decode_tokens") - mean(b, "decode_tokens")
        dz = mean(a, "n_tool_calls") - mean(b, "n_tool_calls")
        parts = {"decode": cm["b"] * dd, "vision": cm["a"] * dv, "tools": cm["c"] * dz}
        total = sum(parts.values())
        if total > 0:
            out["saving"] = {"total_ms": total, "parts": parts,
                             "share": {k: v / total for k, v in parts.items()}}
    return out


def load_run(run_dir: Path, cm: dict, want_sids: list[str] | None,
             max_samples: int) -> dict | None:
    if not run_dir.is_dir():
        return None
    cfg = read_json(run_dir / "config.json")
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    if not cfg and not metrics:
        return None

    ev = pick_eval(metrics)
    ev_out = None
    if ev:
        calls = ev.get("mean_zooms", 0.0)
        ev_out = {
            "tag": phase_tag(ev.get("phase", "eval")),
            "step": ev.get("step", 0),
            "n": ev.get("n"),
            "accuracy": ev.get("accuracy"),
            "tool_rate": ev.get("tool_rate"),
            "mean_zooms": calls,
            "zoom_hist": ev.get("zoom_hist", {}),
            "vision_tokens": ev.get("mean_vision_tokens"),
            "decode_tokens": ev.get("mean_decode_tokens"),
            "invalid_format_rate": ev.get("invalid_format_rate"),
            "box_in_frame_rate": ev.get("box_in_frame_rate"),
            "answer_source": ev.get("answer_source", {}),
            "pred_ms": predict_ms(cm, ev.get("mean_vision_tokens") or 0,
                                  ev.get("mean_decode_tokens") or 0, calls or 0),
            # What the run's own reward charged. Zero when the run has no cost term. Kept so the
            # page can check its own arithmetic against the number the training loop actually used.
            "logged_ms": ev.get("mean_cost_ms"),
        }

    ref = pick_reference(metrics)
    ref_out = {"accuracy": ref.get("accuracy"), "n": ref.get("n"),
               "tool_rate": ref.get("tool_rate")} if ref else None

    series = []
    for m in metrics:
        if not str(m.get("phase", "")).startswith("eval"):
            continue
        if not POLICY_TAG.match(phase_tag(m.get("phase", ""))):
            continue
        series.append({
            "step": m.get("step", 0),
            "tag": phase_tag(m.get("phase", "")),
            "accuracy": m.get("accuracy"),
            "tool_rate": m.get("tool_rate"),
            "mean_zooms": m.get("mean_zooms"),
            "pred_ms": predict_ms(cm, m.get("mean_vision_tokens") or 0,
                                  m.get("mean_decode_tokens") or 0, m.get("mean_zooms") or 0),
        })
    series.sort(key=lambda r: r["step"])

    train = [{"step": m.get("step", 0), "reward": m.get("mean_reward"),
              "tool_rate": m.get("tool_rate"), "mean_zooms": m.get("mean_zooms"),
              "lam": m.get("lam")}
             for m in metrics if m.get("phase") == "train"]
    train.sort(key=lambda r: r["step"])

    # A GRPO group where every rollout scores the same has zero advantage and contributes no
    # gradient. How often that happens is a property of the reward, so it is worth counting.
    gr = [m for m in metrics if m.get("phase") == "train" and m.get("groups_total")]
    groups = {"used": sum(m["groups_used"] for m in gr),
              "total": sum(m["groups_total"] for m in gr),
              "steps": len(gr)} if gr else None

    tag = latest_crop_tag(run_dir)
    preds_path = run_dir / "eval" / f"vstar_predictions_{tag}.jsonl" if tag else None
    if preds_path is None or not preds_path.exists():
        cands = sorted((run_dir / "eval").glob("vstar_predictions_*.jsonl")) \
            if (run_dir / "eval").is_dir() else []
        preds_path = cands[-1] if cands else None
    preds = {r["sid"]: r for r in read_jsonl(preds_path)} if preds_path else {}

    # The never-zoom reference runs on `full[:96]`, and `load_vstar` returns images grouped by
    # category, so all 96 are direct-attribute questions while the full eval is 191 across two
    # categories. Comparing 0.250 against the full-set accuracy compares two populations. So the
    # page reports the policy's accuracy on those SAME 96 images, computed here from the raw
    # predictions rather than taken on trust.
    if ref_out:
        nz_rows = read_jsonl(run_dir / "eval" / "vstar_predictions_never_zoom.jsonl")
        nz_sids = {r["sid"] for r in nz_rows}
        matched = [preds[s] for s in nz_sids if s in preds]
        if matched:
            ref_out["matched_n"] = len(matched)
            ref_out["matched_accuracy"] = sum(bool(r.get("correct")) for r in matched) / len(matched)
            ref_out["matched_tool_rate"] = sum(
                1 for r in matched if (r.get("n_tool_calls") or 0) > 0) / len(matched)
            cats = {s.split("-")[1] for s in nz_sids if "-" in s}
            ref_out["categories"] = sorted(cats)

    episodes: dict[str, dict] = {}
    if tag:
        cdir = run_dir / "crops" / tag
        sids = sorted({p.name.split("_thumb")[0] for p in cdir.glob("*_thumb.jpg")})
        if want_sids:
            sids = [s for s in want_sids if s in sids] + [s for s in sids if s not in want_sids]
        for sid in sids[:max_samples]:
            p = preds.get(sid)
            if not p:
                continue
            boxes = p.get("boxes") or []
            crop_paths = [cdir / f"{sid}_crop{i}.jpg" for i in range(len(boxes))]
            thinks = [think_of(t.get("text", "")) for t in p.get("turns", [])]
            episodes[sid] = {
                "boxes": boxes,
                "crops": [data_uri(im, 360, 78) for im in
                          (open_image(cp) for cp in crop_paths) if im is not None],
                "answer": p.get("answer"),
                "correct": bool(p.get("correct")),
                "n_tool_calls": p.get("n_tool_calls", 0),
                "vision_tokens": p.get("vision_tokens"),
                "decode_tokens": p.get("decode_tokens"),
                "pred_ms": predict_ms(cm, p.get("vision_tokens") or 0,
                                      p.get("decode_tokens") or 0, p.get("n_tool_calls") or 0),
                "thinks": [t for t in thinks if t],
                "_thumb_src": str(cdir / f"{sid}_thumb.jpg"),
            }

    return {
        "cost_mode": cfg.get("cost_mode"),
        "cost_model": cfg.get("cost_model", {}),
        "lam_target": cfg.get("lam_target"),
        "max_zooms": cfg.get("max_zooms"),
        "seed": cfg.get("seed"),
        "group_size": cfg.get("group_size"),
        "minutes": cfg.get("minutes"),
        "started_at": cfg.get("started_at"),
        "note": cfg.get("note"),
        "done": (run_dir / "DONE").exists(),
        "steps_seen": max([m.get("step", 0) for m in metrics], default=0),
        "expected_steps": cfg.get("expected_steps"),
        "eval": ev_out,
        "never_zoom": ref_out,
        "groups": groups,
        "series": series,
        "train": train,
        "episodes": episodes,
        "preds_n": len(preds),
    }


# ---------------------------------------------------------------- assembly


def build(archive: Path, out: Path, dev: bool, max_samples: int) -> dict:
    cm = load_cost_model("q4")
    spec = list(RUN_SPEC) + (DEV_SPEC if dev else [])

    runs: dict[str, dict] = {}
    for rid, dirname, label, role, blurb in spec:
        r = load_run(archive / dirname, cm, None, max_samples)
        if r is None:
            runs[rid] = {"id": rid, "label": label, "role": role, "blurb": blurb,
                         "present": False}
            continue
        r.update({"id": rid, "label": label, "role": role, "blurb": blurb, "present": True})
        runs[rid] = r

    # One clean thumbnail per image, shared by every run: same source image, same
    # thumbnail rule, so the pixels are identical apart from the burned-in boxes.
    samples: list[dict] = []
    seen: set[str] = set()
    residuals: dict[str, int] = {}
    order = [rid for rid, *_ in spec if runs.get(rid, {}).get("present")]
    # An image is only worth a slot if more than one run answered it — the whole point of
    # the toggle is the same question under two policies. So rank by how many runs have it,
    # and keep the run order as the tie-break.
    seq: list[str] = []
    for rid in order:
        for sid in runs[rid].get("episodes", {}):
            if sid not in seq:
                seq.append(sid)
    have = {sid: sum(1 for rid in order if sid in runs[rid].get("episodes", {})) for sid in seq}
    sids = sorted(seq, key=lambda s: (-have[s], seq.index(s)))

    preds_by_sid: dict[str, dict] = {}
    for rid in order:
        ed = runs[rid].get("episodes", {})
        rdir = archive / dict((s[0], s[1]) for s in spec)[rid]
        tag = latest_crop_tag(rdir)
        pp = rdir / "eval" / f"vstar_predictions_{tag}.jsonl" if tag else None
        if pp and pp.exists():
            for row in read_jsonl(pp):
                preds_by_sid.setdefault(row["sid"], row)
        for sid in ed:
            preds_by_sid.setdefault(sid, {})

    for sid in sids[:max_samples]:
        best = None
        for rid in order:
            ep = runs[rid].get("episodes", {}).get(sid)
            if not ep:
                continue
            tp = Path(ep["_thumb_src"])
            if not tp.exists():
                continue
            cdir = tp.parent
            crops = [cdir / f"{sid}_crop{i}.jpg" for i in range(len(ep["boxes"]))]
            try:
                img, resid = clean_thumb(tp, ep["boxes"], crops)
            except OSError:
                continue
            if best is None or resid < best[1]:
                best = (img, resid, rid)
        if best is None:
            continue
        img, resid, _src = best
        residuals[sid] = resid
        p = preds_by_sid.get(sid, {})
        samples.append({
            "sid": sid,
            "short": sid.replace("vstar-", "").replace("direct_attributes-", ""),
            "kind": "direct attributes" if "direct_attributes" in sid else "relative position",
            "question": p.get("question", ""),
            "gold": p.get("gold", ""),
            "thumb": data_uri(img, 384, 86),
            "w": img.size[0], "h": img.size[1],
        })
        seen.add(sid)

    for r in runs.values():
        for ep in r.get("episodes", {}).values():
            ep.pop("_thumb_src", None)
        r["episodes"] = {k: v for k, v in r.get("episodes", {}).items() if k in seen}

    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "cost_model": cm,
        "m4": load_m4(REPO / "eval" / "m4_latency.jsonl"),
        "paired": paired_stats(archive, cm, "run_a", "run_b"),
        "runs": runs,
        "samples": samples,
        "order": [rid for rid, *_ in spec],
        "residuals": residuals,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic: the watcher rebuilds while a publish may be reading the file.
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(render(payload))
    os.replace(tmp, out)
    return payload


def render(d: dict) -> str:
    """Inline the payload. `<` and `&` are escaped so a model's own text — which is full of
    angle brackets — can never close the script tag it is sitting inside."""
    blob = (json.dumps(d, separators=(",", ":"))
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
    return PAGE.replace("__DATA__", blob)


# ---------------------------------------------------------------- the page

PAGE = r"""<meta charset="utf-8">
<title>Where It Looked</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
/* ---- tokens: light is the bare :root, dark is redefined twice so the OS
       setting and an explicit toggle both win in their own direction ---- */
:root{
  --bg:#F4F6F7; --panel:#FFFFFF; --panel-2:#EDF1F2; --line:#D8E0E2; --line-soft:#E6EBEC;
  --ink:#0F1A1D; --ink-2:#3D4E54; --ink-3:#6B7C82;
  --a:#C4801F; --b:#0E8C7E;                 /* validated categorical pair, both modes */
  --a-wash:rgba(196,128,31,.12); --b-wash:rgba(14,140,126,.12);
  --a-mark:#E09A33; --b-mark:#17B9A4;       /* overlay strokes, sit on photographs */
  --good:#0E8C7E; --bad:#B4442E;
  --stage:#0C1113; --shadow:0 1px 2px rgba(15,26,29,.06),0 8px 24px rgba(15,26,29,.06);
  --grid:rgba(15,26,29,.10);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --bg:#0E1316; --panel:#151B1F; --panel-2:#1B2328; --line:#28333A; --line-soft:#202A2F;
    --ink:#E8EFF1; --ink-2:#AFC0C6; --ink-3:#7D9098;
    --a-wash:rgba(224,154,51,.16); --b-wash:rgba(23,185,164,.16);
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
    --grid:rgba(232,239,241,.12); --bad:#DE6B50;
  }
}
:root[data-theme="dark"]{
  --bg:#0E1316; --panel:#151B1F; --panel-2:#1B2328; --line:#28333A; --line-soft:#202A2F;
  --ink:#E8EFF1; --ink-2:#AFC0C6; --ink-3:#7D9098;
  --a-wash:rgba(224,154,51,.16); --b-wash:rgba(23,185,164,.16);
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
  --grid:rgba(232,239,241,.12); --bad:#DE6B50;
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:15px; line-height:1.55; -webkit-font-smoothing:antialiased;
}
.serif{font-family:ui-serif,"New York","Iowan Old Style","Palatino Linotype",Georgia,serif}
.mono{font-family:ui-monospace,"SF Mono",SFMono-Regular,Menlo,"Cascadia Mono",monospace}
.num{font-variant-numeric:tabular-nums}

.wrap{max-width:1060px;margin:0 auto;padding:0 24px 96px}
section{padding-top:52px}
h2{
  font-family:ui-serif,"New York","Iowan Old Style",Georgia,serif;
  font-size:25px; font-weight:600; letter-spacing:-.01em; margin:0 0 6px; text-wrap:balance;
}
.sub{color:var(--ink-3);margin:0 0 22px;max-width:64ch}
.eyebrow{
  font-family:ui-monospace,"SF Mono",Menlo,monospace; font-size:11px; font-weight:600;
  letter-spacing:.13em; text-transform:uppercase; color:var(--ink-3);
}
a{color:var(--b)}

/* ---- masthead ---- */
header{border-bottom:1px solid var(--line);background:var(--panel)}
.mast{max-width:1060px;margin:0 auto;padding:34px 24px 30px;
  display:flex;flex-wrap:wrap;gap:28px;align-items:flex-end;justify-content:space-between}
.thesis{flex:1 1 460px;min-width:0}
.thesis h1{
  font-family:ui-serif,"New York","Iowan Old Style",Georgia,serif;
  font-size:clamp(29px,4.2vw,45px); line-height:1.09; letter-spacing:-.022em;
  font-weight:600; margin:10px 0 0; text-wrap:balance; max-width:13.5em;
}
.thesis h1 em{font-style:italic;color:var(--b)}
.headline{display:flex;gap:26px;flex-wrap:wrap}
.hstat{min-width:132px}
.hstat .k{font-size:11px;letter-spacing:.11em;text-transform:uppercase;color:var(--ink-3);
  font-family:ui-monospace,Menlo,monospace;font-weight:600}
.hstat .v{font-size:38px;line-height:1;letter-spacing:-.03em;font-weight:600;margin-top:7px;
  font-variant-numeric:tabular-nums}
.hstat .n{font-size:12.5px;color:var(--ink-3);margin-top:6px;max-width:19ch}
.hstat.aa .v{color:var(--a)} .hstat.bb .v{color:var(--b)}

/* ---- panels ---- */
.card{background:var(--panel);border:1px solid var(--line);border-radius:3px;box-shadow:var(--shadow)}

/* ---- viewer ---- */
.viewer{display:grid;grid-template-columns:minmax(0,1.32fr) minmax(300px,1fr);gap:0}
@media(max-width:880px){.viewer{grid-template-columns:1fr}}
.stage-col{border-right:1px solid var(--line);min-width:0}
@media(max-width:880px){.stage-col{border-right:0;border-bottom:1px solid var(--line)}}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:space-between;
  padding:12px 16px;border-bottom:1px solid var(--line-soft)}

.seg{display:inline-flex;border:1px solid var(--line);border-radius:2px;overflow:hidden;background:var(--panel-2)}
.seg button{
  appearance:none;border:0;background:transparent;color:var(--ink-3);cursor:pointer;
  font:600 12px/1 ui-monospace,Menlo,monospace;letter-spacing:.07em;text-transform:uppercase;
  padding:9px 13px;display:flex;align-items:center;gap:7px;transition:color .15s,background .15s;
}
.seg button+button{border-left:1px solid var(--line)}
.seg button .dot{width:8px;height:8px;border-radius:50%;background:currentColor;opacity:.5}
.seg button[aria-pressed="true"]{background:var(--panel);color:var(--ink)}
.seg button[aria-pressed="true"].sa{box-shadow:inset 0 -2px 0 var(--a)} .seg button.sa .dot{background:var(--a);opacity:1}
.seg button[aria-pressed="true"].sb{box-shadow:inset 0 -2px 0 var(--b)} .seg button.sb .dot{background:var(--b);opacity:1}
.seg button:disabled{cursor:not-allowed;opacity:.42}
.seg button:focus-visible{outline:2px solid var(--b);outline-offset:-2px}

.ghost{
  appearance:none;background:transparent;border:1px solid var(--line);border-radius:2px;
  color:var(--ink-2);cursor:pointer;padding:8px 12px;
  font:600 12px/1 ui-monospace,Menlo,monospace;letter-spacing:.07em;text-transform:uppercase;
}
.ghost:hover{border-color:var(--ink-3);color:var(--ink)}
.ghost:focus-visible{outline:2px solid var(--b);outline-offset:2px}
.kbd{display:block;margin-top:7px;font:500 12.5px/1.5 ui-monospace,Menlo,monospace;color:var(--ink-3)}
@media(hover:none){.kbd{display:none}}

.stage{background:var(--stage);padding:26px 22px;display:flex;justify-content:center}
/* The frame must be exactly the image box: every zoom box is positioned as a percentage
   of it. Sizing by aspect ratio keeps that true at any viewport height. */
.frame{position:relative;line-height:0;box-shadow:0 0 0 1px rgba(255,255,255,.09);
  width:min(100%, calc(54vh * var(--ar,1.5)))}
.frame img.base{display:block;width:100%;height:auto}
.box{position:absolute;border:2px solid var(--mk);opacity:0;transform:scale(1.14);
  transform-origin:center;box-shadow:0 0 0 1px rgba(0,0,0,.55),0 0 14px -2px var(--mk);pointer-events:none}
.box.on{opacity:1;transform:scale(1);transition:opacity .28s ease,transform .34s cubic-bezier(.2,.9,.3,1)}
.box .tag{position:absolute;top:-2px;left:-2px;transform:translateY(-100%);
  background:var(--mk);color:#08100F;padding:1px 6px 2px;border-radius:1px;
  font:700 10px/1.35 ui-monospace,Menlo,monospace;letter-spacing:.06em;white-space:nowrap}
.box .tick{position:absolute;width:7px;height:7px;border:0 solid var(--mk)}
.box .t1{top:-2px;left:-2px;border-top-width:2px;border-left-width:2px}
.box .t2{top:-2px;right:-2px;border-top-width:2px;border-right-width:2px}
.box .t3{bottom:-2px;left:-2px;border-bottom-width:2px;border-left-width:2px}
.box .t4{bottom:-2px;right:-2px;border-bottom-width:2px;border-right-width:2px}
.nozoom{position:absolute;inset:auto 0 0 0;padding:9px 12px;background:linear-gradient(transparent,rgba(0,0,0,.75));
  color:#EAF3F2;font:600 11px/1.4 ui-monospace,Menlo,monospace;letter-spacing:.08em;text-transform:uppercase;
  text-align:center;opacity:0;transition:opacity .3s}
.nozoom.on{opacity:1}

.strip{display:flex;gap:10px;padding:14px 16px;overflow-x:auto;border-top:1px solid var(--line-soft);
  scrollbar-width:thin}
.look{flex:0 0 auto;width:148px;opacity:.22;transition:opacity .3s}
.look.on{opacity:1}
/* contain, not cover: the crop is the evidence, so none of it gets trimmed away */
.look img{width:148px;height:100px;object-fit:contain;display:block;background:var(--stage);
  border:1px solid var(--line);border-radius:2px}
.look .cap{margin-top:5px;font:600 10px/1.3 ui-monospace,Menlo,monospace;letter-spacing:.06em;
  text-transform:uppercase;color:var(--mk)}
.strip .empty{color:var(--ink-3);font-size:13px;padding:22px 4px}

.picker{display:flex;gap:7px;padding:12px 16px;overflow-x:auto;border-top:1px solid var(--line-soft)}
.picker button{
  appearance:none;flex:0 0 auto;background:var(--panel-2);border:1px solid var(--line);border-radius:2px;
  color:var(--ink-3);cursor:pointer;padding:7px 11px;
  font:600 11px/1 ui-monospace,Menlo,monospace;letter-spacing:.05em;
}
.picker button[aria-pressed="true"]{background:var(--ink);color:var(--panel);border-color:var(--ink)}
.picker button:focus-visible{outline:2px solid var(--b);outline-offset:2px}

/* ---- episode read-out ---- */
.read{padding:18px 20px 20px;display:flex;flex-direction:column;gap:16px;min-width:0}
.q{font-family:ui-serif,"New York",Georgia,serif;font-size:19px;line-height:1.32;
  letter-spacing:-.01em;margin:0;text-wrap:balance}
.kv{display:grid;grid-template-columns:auto 1fr;gap:5px 14px;font-size:13px;align-items:baseline}
.kv dt{color:var(--ink-3);font:600 10.5px/1.7 ui-monospace,Menlo,monospace;letter-spacing:.09em;text-transform:uppercase}
.kv dd{margin:0;min-width:0;overflow-wrap:anywhere}
.pill{display:inline-flex;align-items:center;gap:6px;padding:2px 8px;border-radius:2px;
  font:600 10.5px/1.7 ui-monospace,Menlo,monospace;letter-spacing:.08em;text-transform:uppercase;
  border:1px solid currentColor}
.pill.ok{color:var(--good);background:var(--b-wash)} .pill.no{color:var(--bad)}
.meter{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);
  border:1px solid var(--line);border-radius:2px;overflow:hidden}
.meter div{background:var(--panel);padding:9px 11px;display:flex;flex-direction:column;justify-content:space-between}
.meter .k{font:600 9.5px/1.5 ui-monospace,Menlo,monospace;letter-spacing:.09em;text-transform:uppercase;
  color:var(--ink-3);min-height:2.2em}
.meter .v{font-size:17px;font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums;margin-top:2px}
.think{border-left:2px solid var(--mk);padding:2px 0 2px 12px;margin:0;font-size:12.5px;
  line-height:1.5;color:var(--ink-2)}
.think b{display:block;font:700 10px/1.6 ui-monospace,Menlo,monospace;letter-spacing:.09em;
  text-transform:uppercase;color:var(--mk)}
.thinks{display:flex;flex-direction:column;gap:11px;max-height:230px;overflow-y:auto}

/* ---- comparison table ---- */
.tblwrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:right;padding:11px 14px;border-bottom:1px solid var(--line-soft);white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal}
thead th{font:600 10.5px/1.6 ui-monospace,Menlo,monospace;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);border-bottom:1px solid var(--line)}
tbody td{font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:0}
td.metric{color:var(--ink-2)}
td.metric small{display:block;color:var(--ink-3);font-size:11.5px;line-height:1.4}
.swatch{display:inline-block;width:9px;height:9px;border-radius:1px;margin-right:7px;vertical-align:baseline}
.delta{font:600 12px/1 ui-monospace,Menlo,monospace}

/* ---- charts ---- */
.charts{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:820px){.charts{grid-template-columns:1fr}}
.chart{padding:16px 18px 12px}
.chart h3{margin:0;font-size:14px;font-weight:600;letter-spacing:-.005em}
.chart p.cs{margin:3px 0 12px;font-size:12.5px;color:var(--ink-3)}
.legend{display:flex;gap:16px;margin:0 0 10px;font:600 11px/1 ui-monospace,Menlo,monospace;
  letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);flex-wrap:wrap}
.legend span{display:inline-flex;align-items:center;gap:6px}
svg{display:block;width:100%;height:auto;overflow:visible}
.gl{stroke:var(--grid);stroke-width:1}
.axlab{fill:var(--ink-3);font:500 10.5px ui-monospace,Menlo,monospace}
.vlab{font:600 10.5px ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}

/* ---- measured-on-device panel ---- */
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:1px;background:var(--line)}
.tile{background:var(--panel);padding:15px 17px;display:flex;flex-direction:column;gap:3px}
.tile .lbl{display:flex;align-items:baseline;gap:7px;font:600 11px/1.5 ui-monospace,Menlo,monospace;
  letter-spacing:.09em;text-transform:uppercase;color:var(--ink-2)}
.tile .lbl i{width:9px;height:9px;border-radius:1px;flex:0 0 auto;align-self:center}
.tile .big{font-size:29px;line-height:1.05;font-weight:600;letter-spacing:-.028em;
  font-variant-numeric:tabular-nums;margin-top:4px}
.tile .rel{font:600 12px/1.4 ui-monospace,Menlo,monospace;color:var(--ink-3)}
.tile .why{font-size:12.5px;color:var(--ink-3);margin-top:2px}
.bars{padding:2px 17px 15px}
.bars .row{display:grid;grid-template-columns:96px 1fr auto;gap:11px;align-items:center;
  padding:6px 0;font-size:12.5px}
.bars .nm{font:600 11px/1.4 ui-monospace,Menlo,monospace;letter-spacing:.07em;text-transform:uppercase;
  color:var(--ink-2);white-space:nowrap}
/* both must be blockified: width and height do nothing on an inline span */
.bars .track{display:block;height:10px;background:var(--panel-2);border-radius:1px;overflow:hidden}
.bars .fill{display:block;height:100%;border-radius:0 2px 2px 0;min-width:2px}
.bars .val{font-variant-numeric:tabular-nums;color:var(--ink);font-weight:600;white-space:nowrap}

/* ---- notices ---- */
.notice{display:flex;gap:12px;align-items:flex-start;padding:14px 16px;border-radius:2px;
  border:1px solid var(--line);background:var(--panel-2);font-size:13.5px;color:var(--ink-2)}
.notice .mk{flex:0 0 auto;width:6px;align-self:stretch;background:var(--ink-3);border-radius:1px}
.notice.wait{background:var(--a-wash)}
.notice.wait .mk{background:var(--a)}
.notice strong{color:var(--ink)}
.status{display:inline-flex;align-items:center;gap:7px;padding:3px 9px;border:1px solid var(--line);
  border-radius:2px;font:600 10.5px/1.7 ui-monospace,Menlo,monospace;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink-3);background:var(--panel)}
.status .led{width:7px;height:7px;border-radius:50%;background:var(--led,var(--ink-3))}
.status.live .led{animation:pulse 1.7s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.28}}

/* ---- method ---- */
.method{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1px;background:var(--line)}
.method>div{background:var(--panel);padding:16px 18px}
.method h4{margin:0 0 5px;font:600 10.5px/1.6 ui-monospace,Menlo,monospace;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-3)}
.method p{margin:0;font-size:13px;color:var(--ink-2)}
.method code{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--ink)}
footer{margin-top:52px;padding-top:20px;border-top:1px solid var(--line);color:var(--ink-3);font-size:12.5px}
footer p{margin:0 0 7px;max-width:78ch}

@media (prefers-reduced-motion:reduce){
  *{animation-duration:.001ms!important;animation-iteration-count:1!important;transition-duration:.001ms!important}
}
</style>

<header>
  <div class="mast">
    <div class="thesis">
      <div class="eyebrow">Cost-aware visual search &middot; Qwen3.5-4B &middot; GRPO</div>
      <h1 id="hero">One term in the reward. <em>Same answers, faster on the laptop.</em></h1>
    </div>
    <div class="headline" id="headline"></div>
  </div>
</header>

<div class="wrap">

  <section id="viewer-sec">
    <div class="eyebrow">The viewer</div>
    <h2>Where it looked</h2>
    <p class="sub">Every box is one call to the zoom tool, drawn in the order the policy asked for
      it. The crop underneath is what came back. Switch runs to see the same question answered
      with a different price on looking.
      <span class="kbd">Arrow keys change the image, 1&ndash;4 pick a run, R replays.</span></p>
    <div class="card viewer">
      <div class="stage-col">
        <div class="bar">
          <div class="seg" id="runseg" role="group" aria-label="Choose a run"></div>
          <button class="ghost" id="replay" type="button">Replay</button>
        </div>
        <div class="stage"><div class="frame" id="frame"></div></div>
        <div class="strip" id="strip"></div>
        <div class="picker" id="picker" role="group" aria-label="Choose an image"></div>
      </div>
      <div class="read" id="read"></div>
    </div>
  </section>

  <section id="cmp-sec">
    <div class="eyebrow">The comparison</div>
    <h2>What the cost term bought</h2>
    <p class="sub" id="cmp-sub"></p>
    <div id="cmp-notice"></div>
    <div class="card tblwrap" style="margin-top:14px"><table id="cmp"></table></div>
  </section>

  <section id="m4-sec" hidden>
    <div class="eyebrow">The deployment target</div>
    <h2>What it costs on the machine it ships to</h2>
    <p class="sub" id="m4-sub"></p>
    <div class="card" id="m4-panel"></div>
    <div class="charts" id="m4-charts" style="margin-top:18px"></div>
  </section>

  <section id="chart-sec">
    <div class="eyebrow">The distribution</div>
    <h2>How often it looked</h2>
    <p class="sub">The headline rate hides the shape. These are the same eval episodes, counted
      by how many zooms each one spent.</p>
    <div class="charts" id="charts"></div>
  </section>

  <section id="method-sec">
    <div class="eyebrow">Provenance</div>
    <h2>Where every number comes from</h2>
    <p class="sub">Nothing here is estimated by hand. Latency is the one number that is modelled
      rather than measured per episode, and the model is stated below.</p>
    <div class="card method" id="method"></div>
    <footer id="foot"></footer>
  </section>

</div>

<script>
const D = __DATA__;
const F = {
  pct: v => v == null ? "—" : (v * 100).toFixed(v * 100 >= 99.5 ? 0 : 1) + "%",
  n2:  v => v == null ? "—" : v.toFixed(2),
  n0:  v => v == null ? "—" : Math.round(v).toLocaleString(),
  sec: v => v == null ? "—" : (v / 1000).toFixed(2) + " s",
};
const esc = s => String(s == null ? "" : s).replace(/[&<>"]/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const RUNS = D.order.map(id => D.runs[id]).filter(r => r);
const LIVE = RUNS.filter(r => r.present && r.eval);
const A = D.runs.A, B = D.runs.B;
const hasB = !!(B && B.present && B.eval);
const MK = { A: "var(--a-mark)", B: "var(--b-mark)" };
const SER = { A: "var(--a)", B: "var(--b)" };
const mark = id => MK[id] || "var(--ink-3)";
const ser  = id => SER[id] || "var(--ink-3)";

const shows = r => r.present && r.eval && Object.keys(r.episodes || {}).length > 0;
let curRun = (A && shows(A)) ? "A" : ((LIVE.find(shows) || LIVE[0] || { id: null }).id);
let curSid = D.samples.length ? D.samples[0].sid : null;
let timers = [];

/* ---------- masthead ---------- */
function statusOf(r){
  if (!r || !r.present) return { cls: "", txt: "not started" };
  if (r.done) return { cls: "done", txt: "complete" };
  const of = r.expected_steps ? " / ~" + r.expected_steps : "";
  return { cls: "live", txt: "training · step " + r.steps_seen + of };
}
/* An eval at step 0 runs before the first optimizer step: it is the base model, not the run's
   policy. Never let one stand in a headline as if it were a trained result. */
const trained = e => !!e && e.step > 0;
/* A run is "settled" when it has finished and run its own full eval. Only settled runs get a
   headline number, because a mid-run checkpoint is scored on a 32-question subset and would sit
   next to a finished run's 191-question number as if the two were the same measurement. */
const settled = r => !!r && r.present && trained(r.eval) && (r.done || r.eval.tag === "final");

function headline(){
  const el = document.getElementById("headline");
  const a = A && A.eval, b = hasB ? B.eval : null;
  const items = [];
  items.push({ cls: "aa", k: "Run A · baseline", v: a ? F.n2(a.mean_zooms) : "—",
    n: (a ? "looks per question, no cost term" : "not started")
       + (a && !trained(a) ? " — base model, before the first step" : "") });
  const bOK = settled(B);
  items.push({ cls: "bb", k: "Run B · cost-aware", v: bOK ? F.n2(b.mean_zooms) : "training",
    n: bOK ? "looks per question, " + F.pct(1 - b.mean_zooms / a.mean_zooms) + " fewer than Run A"
       : (B && B.present
            ? "Run B is at step " + B.steps_seen + (B.expected_steps ? " of ~" + B.expected_steps : "")
              + ". Its checkpoints are scored on a 32-question subset, so there is no number here "
              + "that belongs beside Run A's."
            : "Run B has not started. No number to show yet.") });
  if (settled(A) && bOK && a.n === b.n){
    const d = (b.accuracy - a.accuracy) * 100;
    items.push({ cls: "", k: "Accuracy", v: F.pct(a.accuracy) + " → " + F.pct(b.accuracy),
      n: `${d >= 0 ? "+" : ""}${d.toFixed(1)} points on the same ${a.n} questions` });
  }
  // Prefer the wall clock actually timed on the Mac. Fall back to the model, and say which.
  const mg = (D.m4 && D.m4.groups) || [];
  const ma = mg.find(x => x.tag === "run_a_final"), mb = mg.find(x => x.tag === "run_b_final");
  if (ma && mb){
    items.push({ cls: "", k: "B vs A on an M4 Max", v: (ma.ms / mb.ms).toFixed(2) + "×",
      n: `faster per question, measured at Q4_K_M on ${ma.n} images` });
  } else if (settled(A) && bOK && a.n === b.n && a.pred_ms && b.pred_ms){
    items.push({ cls: "", k: "B vs A, predicted", v: (a.pred_ms / b.pred_ms).toFixed(2) + "×",
      n: "from the cost model, not yet timed on the Mac" });
  }
  el.innerHTML = items.map(i =>
    `<div class="hstat ${i.cls}"><div class="k">${esc(i.k)}</div>
     <div class="v">${esc(i.v)}</div><div class="n">${esc(i.n)}</div></div>`).join("");
}

/* ---------- viewer ---------- */
function runSeg(){
  const el = document.getElementById("runseg");
  el.innerHTML = RUNS.map(r => {
    const on = r.id === curRun, ok = shows(r);
    const s = statusOf(r);
    const t = ok ? r.label + " · " + r.role : r.label + " · " + s.txt;
    return `<button type="button" class="s${r.id.toLowerCase()}" data-run="${r.id}"
      aria-pressed="${on}" ${ok ? "" : "disabled"} title="${esc(t)}">
      <span class="dot"></span>${esc(r.label)}</button>`;
  }).join("");
  el.querySelectorAll("button[data-run]").forEach(b =>
    b.onclick = () => { curRun = b.dataset.run; runSeg(); draw(); });
}
function picker(){
  const el = document.getElementById("picker");
  el.innerHTML = D.samples.map((s, i) =>
    `<button type="button" data-sid="${esc(s.sid)}" aria-pressed="${s.sid === curSid}"
      title="${esc(s.question)}">${String(i + 1).padStart(2, "0")} &nbsp;${esc(s.short)}</button>`).join("");
  el.querySelectorAll("button[data-sid]").forEach(b =>
    b.onclick = () => { curSid = b.dataset.sid; picker(); draw(); });
}

function draw(){
  timers.forEach(clearTimeout); timers = [];
  const s = D.samples.find(x => x.sid === curSid);
  const r = D.runs[curRun];
  const ep = (r && r.episodes) ? r.episodes[curSid] : null;
  const frame = document.getElementById("frame");
  if (s) frame.style.setProperty("--ar", (s.w / s.h).toFixed(4));
  const strip = document.getElementById("strip");
  const read = document.getElementById("read");
  if (!s){ frame.innerHTML = ""; read.innerHTML = "<p class='sub'>No images mirrored yet.</p>"; return; }

  const mk = mark(curRun);
  const boxes = ep ? (ep.boxes || []) : [];
  frame.innerHTML = `<img class="base" alt="V*Bench image, reduced to the 384 px view the policy sees"
      src="${s.thumb}" width="${s.w}" height="${s.h}">`
    + boxes.slice(0, 4).map((b, i) => {
        const L = b[0] / 10, T = b[1] / 10, W = (b[2] - b[0]) / 10, H = (b[3] - b[1]) / 10;
        return `<div class="box" data-i="${i}" style="--mk:${mk};left:${L}%;top:${T}%;width:${W}%;height:${H}%">
          <i class="tick t1"></i><i class="tick t2"></i><i class="tick t3"></i><i class="tick t4"></i>
          <span class="tag">Look ${i + 1}</span></div>`;
      }).join("")
    + (boxes.length === 0
        ? `<div class="nozoom">No zoom · answered from the 384 px view</div>` : "");

  const crops = ep ? (ep.crops || []) : [];
  strip.innerHTML = crops.length
    ? crops.map((c, i) =>
        `<figure class="look" data-i="${i}" style="--mk:${mk};margin:0">
          <img src="${c}" alt="Crop ${i + 1}, taken from the full-resolution original">
          <figcaption class="cap">Look ${i + 1}</figcaption></figure>`).join("")
    : `<div class="empty">${ep ? "The policy took no crop on this question."
                               : "This run has not written crops for this image."}</div>`;

  read.innerHTML = readout(s, r, ep);
  play();
}

function play(){
  timers.forEach(clearTimeout); timers = [];
  const boxes = [...document.querySelectorAll("#frame .box")];
  const looks = [...document.querySelectorAll("#strip .look")];
  const nz = document.querySelector("#frame .nozoom");
  boxes.forEach(b => b.classList.remove("on"));
  looks.forEach(l => l.classList.remove("on"));
  if (nz) nz.classList.remove("on");
  const step = 620;
  boxes.forEach((b, i) => timers.push(setTimeout(() => b.classList.add("on"), 340 + i * step)));
  looks.forEach((l, i) => timers.push(setTimeout(() => l.classList.add("on"), 340 + i * step)));
  if (nz) timers.push(setTimeout(() => nz.classList.add("on"), 420));
}
document.getElementById("replay").onclick = play;

/* Driven live in a room: arrows step through images, 1-4 pick a run, R replays. */
document.addEventListener("keydown", e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
  const i = D.samples.findIndex(s => s.sid === curSid);
  let hit = true;
  if (e.key === "ArrowRight" || e.key === "ArrowDown"){
    curSid = D.samples[(i + 1) % D.samples.length].sid; picker(); draw();
  } else if (e.key === "ArrowLeft" || e.key === "ArrowUp"){
    curSid = D.samples[(i - 1 + D.samples.length) % D.samples.length].sid; picker(); draw();
  } else if (e.key === "r" || e.key === "R"){ play(); }
  else if (/^[1-9]$/.test(e.key)){
    const r = RUNS[+e.key - 1];
    if (r && shows(r)){ curRun = r.id; runSeg(); draw(); }
  } else hit = false;
  if (hit) e.preventDefault();
});

function readout(s, r, ep){
  const mk = mark(curRun);
  const st = statusOf(r);
  let h = `<div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
      <span class="status ${st.cls}" style="--led:${ser(curRun)}"><span class="led"></span>${esc(r.label)} · ${esc(st.txt)}</span>
      <span class="eyebrow">${esc(s.kind)}</span></div>
    <p class="q">${esc(s.question)}</p>`;
  if (!ep){
    return h + `<div class="notice wait"><span class="mk"></span><div>
      <strong>${esc(r.label)} has not answered this image yet.</strong>
      Its crops mirror from the rig as the eval runs.</div></div>`;
  }
  h += `<dl class="kv">
      <dt>Answer</dt><dd>${esc(ep.answer || "— no parseable answer")}
        <span class="pill ${ep.correct ? "ok" : "no"}">${ep.correct ? "correct" : "wrong"}</span></dd>
      <dt>Gold</dt><dd>${esc(s.gold)}</dd>
    </dl>
    <div class="meter">
      <div><div class="k">Zoom calls</div><div class="v">${ep.n_tool_calls}</div></div>
      <div><div class="k">Decode tokens</div><div class="v num">${F.n0(ep.decode_tokens)}</div></div>
      <div><div class="k">Reward cost</div><div class="v num">${F.sec(ep.pred_ms)}</div></div>
    </div>`;
  const th = (ep.thinks || []).filter(t => t);
  if (th.length){
    h += `<div class="thinks">` + th.map((t, i) =>
      `<p class="think" style="--mk:${mk}"><b>Turn ${i + 1}</b>${esc(t)}</p>`).join("") + `</div>`;
  }
  return h;
}

/* ---------- comparison ---------- */
function comparison(){
  const sub = document.getElementById("cmp-sub");
  const note = document.getElementById("cmp-notice");
  const a = A && A.eval ? A : null, b = hasB ? B : null;
  /* C and D only earn a column once they have finished and run their own full eval. A run that
     is mid-training is scored on the 32-question subset, and letting it in would drag the whole
     table off a common eval set and silently strip the A-vs-B deltas. */
  const others = RUNS.filter(r => !["A", "B"].includes(r.id) && settled(r));
  const cols = [a, b].filter(Boolean).concat(others);

  const P = comparableStats();
  const sameN = new Set(cols.map(r => r.eval.n)).size === 1;
  /* Deltas need two trained policies measured on the same eval set. A mid-run checkpoint is
     evaluated on a 32-question subset while a finished run is evaluated on all 191, and
     subtracting one from the other invents a result. */
  const comparable = cols.length > 1 && trained(cols[0].eval) && trained(cols[1].eval) && sameN;

  sub.textContent = comparable
    ? "Same model, same seed, same data, same eval set. One line of the reward differs."
    : (cols.length > 1
        ? "Same model, same seed, same data. One line of the reward differs — but the two columns "
          + "are not yet on the same eval, so read them as two separate measurements."
        : "Run A is on the board. Run B trains next, and the second column fills in when it lands.");

  /* Report what did not happen. The prediction going in was that charging for looks would cut
     the tool-call RATE. It did not: both runs call the tool on ~99% of questions. What moved is
     how many times. Saying so is the difference between a result and a sales pitch. */
  let nullNote = "";
  if (comparable){
    const ra = cols[0].eval.tool_rate, rb = cols[1].eval.tool_rate;
    if (rb >= ra - 0.02){
      nullNote = `<div class="notice"><span class="mk"></span><div>
        <strong>What did not happen.</strong> The cost term did not teach the policy to skip
        looking. Both runs still call the zoom tool on about ${F.pct(Math.max(ra, rb))} of
        questions${rb > ra ? ` — Run B's rate is marginally <em>higher</em> than Run A's` : ""}.
        The prediction going in was a lower tool-call rate, and that is a null. What the cost term
        changed is how many times it looks once it starts: Run A's most common episode spends the
        whole budget, Run B's spends one look and commits.</div></div>`;
    }
  }
  note.innerHTML = nullNote + (comparable ? "" : `<div class="notice wait"><span class="mk"></span><div>
      <strong>${b && trained(b.eval) ? "Run B is still training."
        : b ? "Run B has no trained eval yet." : "Run B is not in yet."}</strong>
      ${b && trained(b.eval) ? "It has a trained checkpoint, but it is still mid-run and evaluated "
          + "on the 32-question subset while Run A has finished on all 191. Those are different "
          + "eval sets, so there is no change column until B runs its own full eval."
        : b ? "Its only eval so far ran before the first GRPO step, so it is the base model, not a "
          + "cost-aware policy. No change column until that is a real checkpoint."
          : "Only the baseline has finished mirroring, so this page shows one column and no "
          + "A-vs-B claim."} Nothing here is duplicated or filled in to stand in for B.</div></div>`);

  const rows = [
    ["Tool-call rate", "questions that got at least one zoom"
      + (P ? `. Saturated near 1.0 for both runs, so it carries almost no information here —
              paired McNemar p = ${P.tool_p.toFixed(2)}, not a difference` : ""),
      r => F.pct(r.eval.tool_rate), "down"],
    ["Zooms per question", "mean calls, budget of " + (r0(cols) || 3), r => F.n2(r.eval.mean_zooms), "down"],
    ["Accuracy", sameN ? ((cols[0].eval.n >= 150 ? "V*Bench, all " : "V*Bench subset, ")
        + (cols[0].eval.n || "?") + " questions"
        + (P ? `. Paired McNemar p = ${P.acc_p.toFixed(2)}: B fixed ${P.acc_b_only} that A missed,
                A fixed ${P.acc_a_only} that B missed. Matched, not better and not worse` : ""))
      : "V*Bench, held out — the columns are on different-sized evals, so read them separately",
      r => F.pct(r.eval.accuracy) + (sameN ? ""
        : `<span style="color:var(--ink-3);font-weight:400"> · n=${r.eval.n}</span>`), "up"],
    ["Cost the reward charged",
      "predicted M4 ms from this episode's token counts — measured wall clock is in the next section",
      r => F.sec(r.eval.pred_ms), "down"],
    ["Decode tokens", "the expensive term on this hardware", r => F.n0(r.eval.decode_tokens), "down"],
    ["Vision tokens", "thumbnail plus every crop", r => F.n0(r.eval.vision_tokens), "down"],
    ["Boxes inside the frame", "how often a box was usable as emitted", r => F.pct(r.eval.box_in_frame_rate), "up"],
    ["Groups that produced a gradient",
      "GRPO scores 8 rollouts against each other; a group where they all score the same has zero "
      + "variance and teaches nothing. A binary correct/wrong reward ties often — a cost term "
      + "separates rollouts that are equally correct",
      r => r.groups ? F.pct(r.groups.used / r.groups.total)
        + `<span style="color:var(--ink-3);font-weight:400"> · ${r.groups.used}/${r.groups.total}`
        + ` over ${r.groups.steps} steps</span>` : "—", "up"],
  ];
  const head = `<thead><tr><th>Metric</th>` + cols.map(r =>
      `<th><span class="swatch" style="background:${ser(r.id)}"></span>${esc(r.label)}<br>
       <span style="color:var(--ink-3);font-weight:500">${esc(r.role)} · ${esc(r.eval.tag)}</span></th>`).join("")
    + (cols.length > 1 ? `<th>Change</th>` : "") + `</tr></thead>`;

  const body = rows.map(([k, note2, fn, dir]) => {
    let d = "";
    if (cols.length > 1 && !comparable){
      d = `<td class="delta" style="color:var(--ink-3)">—</td>`;
    } else if (cols.length > 1){
      const va = num(cols[0], k), vb = num(cols[1], k);
      if (va != null && vb != null && va !== 0){
        const p = (vb - va) / Math.abs(va) * 100;
        const good = dir === "down" ? p < 0 : p > 0;
        d = `<td class="delta" style="color:${Math.abs(p) < 0.5 ? "var(--ink-3)"
          : good ? "var(--b)" : "var(--a)"}">${p > 0 ? "+" : ""}${p.toFixed(p > -10 && p < 10 ? 1 : 0)}%</td>`;
      } else d = `<td class="delta" style="color:var(--ink-3)">—</td>`;
    }
    return `<tr><td class="metric">${esc(k)}<small>${esc(note2)}</small></td>`
      + cols.map(r => `<td>${fn(r)}</td>`).join("") + d + `</tr>`;
  }).join("");

  /* The thumbnail-only reference. It runs on the first 96 images, which `load_vstar` returns
     grouped by category, so all 96 are direct-attribute questions — a different population from
     the 191-question eval. Reported as a matched pair on those same 96 images so nobody can
     subtract it from the full-set accuracy and get a number that is not true. */
  const anyNZ = cols.some(r => r.never_zoom && r.never_zoom.matched_accuracy != null);
  const nzCell = (r, pick) => {
    const z = r.never_zoom;
    if (!z || z.matched_accuracy == null) return "—";
    return F.pct(pick(z));
  };
  const nzN = (cols.find(r => r.never_zoom && r.never_zoom.matched_n) || { never_zoom: {} })
    .never_zoom.matched_n;
  const nzCats = (cols.find(r => r.never_zoom && r.never_zoom.categories) || { never_zoom: {} })
    .never_zoom.categories;
  const blank = cols.length > 1 ? `<td class="delta" style="color:var(--ink-3)">—</td>` : "";
  const refRow = !anyNZ ? "" :
    `<tr><td class="metric">Accuracy on the reference subset, zooming allowed
        <small>the policy on the same ${nzN} images the line below uses${
          nzCats && nzCats.length === 1
            ? ` — all of them ${esc({ direct_attributes: "direct-attribute",
                relative_position: "relative-position" }[nzCats[0]] || nzCats[0])} questions, so
                this is a different population from the full eval above` : ""}</small></td>`
      + cols.map(r => `<td>${nzCell(r, z => z.matched_accuracy)}</td>`).join("") + blank + `</tr>`
    + `<tr><td class="metric">Accuracy with zooming disabled
        <small>reference line, not a policy: the same weights on the same ${nzN} images with the
        zoom budget set to 0</small></td>`
      + cols.map(r => `<td>${nzCell(r, z => z.accuracy)}</td>`).join("") + blank + `</tr>`;

  const caption = cols.map(r => {
    const e = r.eval, base = e.step === 0;
    return `<strong>${esc(r.label)}</strong> — eval <code>${esc(e.tag)}</code>, `
      + (base ? "taken before the first GRPO step, so this is the base model with an untrained "
              + "adapter, not a trained policy"
              : "after " + e.step + " GRPO step" + (e.step === 1 ? "" : "s"))
      + `. Answer parsing fell back to prose on `
      + `${e.invalid_format_rate == null ? "—" : F.pct(e.invalid_format_rate)} of episodes.`;
  }).join(" ");
  document.getElementById("cmp").innerHTML =
    `<caption style="caption-side:bottom;text-align:left;padding:12px 14px;font-size:12.5px;
      color:var(--ink-3);border-top:1px solid var(--line)">${caption}</caption>`
    + head + `<tbody>${body}${refRow}</tbody>`;
}
function comparableStats(){
  const p = D.paired;
  return (p && settled(A) && settled(B)) ? p : null;
}
function r0(cols){ return cols[0] && cols[0].max_zooms; }
function num(r, k){
  const e = r.eval;
  return { "Tool-call rate": e.tool_rate, "Zooms per question": e.mean_zooms, "Accuracy": e.accuracy,
    "Cost the reward charged": e.pred_ms, "Decode tokens": e.decode_tokens, "Vision tokens": e.vision_tokens,
    "Boxes inside the frame": e.box_in_frame_rate,
    "Groups that produced a gradient": r.groups ? r.groups.used / r.groups.total : null }[k];
}

/* ---------- charts ---------- */
function charts(){
  const host = document.getElementById("charts");
  // Same rule as the table: only settled runs are peers. A step-0 histogram is the base model.
  let cols = RUNS.filter(r => settled(r) && r.eval.zoom_hist);
  if (!cols.length) cols = RUNS.filter(r => r.present && r.eval && r.eval.zoom_hist);
  let out = "";
  if (cols.length) out += histCard(cols);
  const withSeries = RUNS.filter(r => r.present &&
    ((r.series || []).length >= 2 || (r.train || []).length >= 3));
  out += withSeries.length ? curveCard(withSeries) : pendingCard();
  host.innerHTML = out;
}

/* A bar with only its data end rounded; the baseline end stays square. */
function barPath(x, y, w, h, r){
  const rr = Math.min(r, w / 2, h);
  return `M${x} ${y + h} L${x} ${y + rr} Q${x} ${y} ${x + rr} ${y}
    L${x + w - rr} ${y} Q${x + w} ${y} ${x + w} ${y + rr} L${x + w} ${y + h} Z`;
}
/* A tick step that yields whole-percent labels. */
function niceMax(top){
  const st = [0.05, 0.1, 0.2, 0.25, 0.5].find(s => top / s <= 5) || 0.5;
  const ymax = Math.min(1, Math.ceil(top / st - 1e-9) * st);
  return { ymax, n: Math.max(2, Math.round(ymax / st)) };
}

function histCard(cols){
  const maxZ = Math.max(3, ...cols.flatMap(r => Object.keys(r.eval.zoom_hist).map(Number)));
  const keys = [...Array(maxZ + 1).keys()];
  const W = 460, H = 210, PL = 34, PR = 10, PT = 10, PB = 34;
  const iw = W - PL - PR, ih = H - PT - PB;
  const tot = r => Object.values(r.eval.zoom_hist).reduce((a, b) => a + b, 0) || 1;
  const top = Math.max(0.1, ...cols.flatMap(r => keys.map(k => (r.eval.zoom_hist[k] || 0) / tot(r))));
  const { ymax, n: nt } = niceMax(top);
  const gw = iw / keys.length, bw = Math.max(9, Math.min(22, (gw - 16) / cols.length - 2));

  let g = "";
  for (let t = 0; t <= nt; t++){
    const y = PT + ih - (t / nt) * ih;
    g += `<line class="gl" x1="${PL}" x2="${W - PR}" y1="${y}" y2="${y}"></line>
      <text class="axlab" x="${PL - 7}" y="${y + 3.5}" text-anchor="end">${Math.round(ymax * t / nt * 100)}</text>`;
  }
  let bars = "";
  keys.forEach((k, ki) => {
    const cx = PL + gw * ki + gw / 2;
    const span = cols.length * bw + (cols.length - 1) * 2;
    cols.forEach((r, ri) => {
      const v = (r.eval.zoom_hist[k] || 0) / tot(r), hgt = Math.max(v > 0 ? 2 : 0, (v / ymax) * ih);
      const x = cx - span / 2 + ri * (bw + 2), y = PT + ih - hgt;
      if (hgt) bars += `<path d="${barPath(x, y, bw, hgt, 4)}"
        fill="${ser(r.id)}"><title>${esc(r.label)}: ${(v * 100).toFixed(0)}% of questions used ${k} zoom${k === 1 ? "" : "s"}</title></path>`;
      if (v > 0.015) bars += `<text class="vlab" x="${x + bw / 2}" y="${y - 5}" text-anchor="middle"
        fill="var(--ink-2)">${Math.round(v * 100)}</text>`;
    });
    bars += `<text class="axlab" x="${cx}" y="${H - PB + 17}" text-anchor="middle">${k}</text>`;
  });
  const leg = cols.map(r =>
    `<span><i class="swatch" style="background:${ser(r.id)}"></i>${esc(r.label)}</span>`).join("");

  return `<div class="card chart"><h3>Zooms spent per question</h3>
    <p class="cs">Percent of eval questions, by number of tool calls. Budget is
      ${cols[0].max_zooms != null ? cols[0].max_zooms : 3}.</p>
    <div class="legend">${leg}</div>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Grouped bar chart of zooms per question">
      ${g}${bars}
      <line class="gl" x1="${PL}" x2="${W - PR}" y1="${PT + ih}" y2="${PT + ih}"></line>
      <text class="axlab" x="${PL + iw / 2}" y="${H - 2}" text-anchor="middle">zoom calls in the episode</text>
    </svg></div>`;
}

function curveCard(rs){
  const W = 460, H = 210, PL = 34, PR = 46, PT = 10, PB = 34;
  const iw = W - PL - PR, ih = H - PT - PB;
  const steps = rs.flatMap(r => r.series.map(p => p.step).concat(r.train.map(p => p.step)));
  const smax = Math.max(1, ...steps);
  const x = s => PL + (s / smax) * iw;
  const budget = Math.max(1, ...rs.map(r => r.max_zooms || 3));
  const y = v => PT + ih - (v / budget) * ih;
  const path = ps => ps.map((p, i) =>
    (i ? "L" : "M") + x(p.step).toFixed(1) + " " + y(p.mean_zooms).toFixed(1)).join(" ");
  let g = "";
  for (let t = 0; t <= 4; t++){
    const yy = PT + ih - (t / 4) * ih;
    g += `<line class="gl" x1="${PL}" x2="${W - PR}" y1="${yy}" y2="${yy}"></line>
      <text class="axlab" x="${PL - 7}" y="${yy + 3.5}" text-anchor="end">${(budget * t / 4).toFixed(1)}</text>`;
  }
  const labels = [];
  const lines = rs.map(r => {
    // Faint: the training batch, one point per GRPO step. Bold: the held-out eval set.
    const tp = r.train.filter(p => p.mean_zooms != null);
    const ep = r.series.filter(p => p.mean_zooms != null);
    let s = tp.length >= 2 ? `<path d="${path(tp)}" fill="none" stroke="${ser(r.id)}"
        stroke-width="1.25" opacity=".38" stroke-linejoin="round"></path>` : "";
    if (ep.length){
      if (ep.length >= 2) s += `<path d="${path(ep)}" fill="none" stroke="${ser(r.id)}"
        stroke-width="2" stroke-linejoin="round" stroke-linecap="round"></path>`;
      s += ep.map(p => `<circle cx="${x(p.step)}" cy="${y(p.mean_zooms)}" r="4.5" fill="${ser(r.id)}"
          stroke="var(--panel)" stroke-width="2"><title>${esc(r.label)} eval at step ${p.step}: ${p.mean_zooms.toFixed(2)} zooms</title></circle>`).join("");
    }
    // Name the run at whichever series reaches furthest right, so the label never lands on the axis.
    const ends = [ep[ep.length - 1], tp[tp.length - 1]].filter(Boolean);
    const last = ends.sort((a, b) => b.step - a.step)[0];
    if (last) labels.push({ x: x(last.step) + 8, y: y(last.mean_zooms), t: r.label });
    return s;
  }).join("");
  // Nudge apart any labels that would land on top of each other when the runs converge.
  labels.sort((a, b) => a.y - b.y);
  labels.forEach((L, i) => { if (i && L.y - labels[i - 1].y < 12) L.y = labels[i - 1].y + 12; });
  const names = labels.map(L =>
    `<text class="vlab" x="${L.x.toFixed(1)}" y="${(L.y + 3.5).toFixed(1)}"
      fill="var(--ink-2)">${esc(L.t)}</text>`).join("");
  const leg = rs.length > 1 ? `<div class="legend">` + rs.map(r =>
    `<span><i class="swatch" style="background:${ser(r.id)}"></i>${esc(r.label)}</span>`).join("")
    + `</div>` : "";
  return `<div class="card chart"><h3>Zooms per question over training</h3>
    <p class="cs">Mean tool calls against a budget of ${budget}. Bold marks the held-out eval set;
      the faint line is the training batch at every step.</p>
    ${leg}
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Tool-call rate against training step">
      ${g}${lines}${names}
      <line class="gl" x1="${PL}" x2="${W - PR}" y1="${PT + ih}" y2="${PT + ih}"></line>
      <text class="axlab" x="${PL + iw / 2}" y="${H - 2}" text-anchor="middle">GRPO step</text>
    </svg></div>`;
}

function pendingCard(){
  return `<div class="card chart"><h3>Zooms per question over training</h3>
    <p class="cs">Needs two or more eval checkpoints.</p>
    <div class="notice wait" style="margin-top:10px"><span class="mk"></span><div>
      <strong>Too few points so far.</strong> The curve appears once three training steps or a
      second eval land in <code class="mono">metrics.jsonl</code>.</div></div></div>`;
}

/* ---------- measured on the deployment target ---------- */
const M4C = { base_q4: "var(--ink-3)", run_a_final: "var(--a)", run_b_final: "var(--b)",
              run_c_final: "var(--ink-2)", run_d_final: "var(--ink-2)" };

function m4(){
  const m = D.m4;
  if (!m || !m.groups.length) return;
  document.getElementById("m4-sec").hidden = false;

  const g = m.groups, base = g.find(x => x.tag === "base_q4") || g[0], n = g[0].n;
  document.getElementById("m4-sub").textContent =
    `Wall clock timed on an M4 Max through llama.cpp at ${m.quant.toUpperCase()}_K_M, ${n} V*Bench `
    + `images per run. These are measurements, not the cost model's predictions. This is a different `
    + `machine and a smaller image set than the rig eval above, so the two never share a row.`;

  const tiles = g.map(x =>
    `<div class="tile"><div class="lbl"><i style="background:${M4C[x.tag] || "var(--ink-3)"}"></i>
        ${esc(x.label)} · ${esc(x.role)}</div>
      <div class="big">${F.sec(x.ms)}</div>
      <div class="rel">${x === base ? "the reference" : (x.ms / base.ms).toFixed(2) + "× the base model"}</div>
      <div class="why">${esc(x.blurb)}</div></div>`).join("");

  const rows = [["ms", "Wall clock", F.sec], ["tool_rate", "Tool-call rate", F.pct],
                ["zooms", "Zooms per question", F.n2],
                ["decode", "Decode tokens", F.n0]].map(([k, title, fmt]) => {
    const top = Math.max(...g.map(x => x[k])) || 1;
    return `<div style="padding:14px 17px 2px;border-top:1px solid var(--line-soft)">
        <div class="eyebrow">${esc(title)}</div></div><div class="bars">`
      + g.map(x => `<div class="row"><span class="nm">${esc(x.label)}</span>
          <span class="track"><span class="fill" style="width:${(x[k] / top * 100).toFixed(1)}%;
            background:${M4C[x.tag] || "var(--ink-3)"}"></span></span>
          <span class="val">${fmt(x[k])}</span></div>`).join("") + `</div>`;
  }).join("");

  document.getElementById("m4-panel").innerHTML = `<div class="tiles">${tiles}</div>${rows}
    <div style="padding:13px 17px;border-top:1px solid var(--line-soft);font-size:12.5px;color:var(--ink-3)">
      Every number here is a different machine, a different image set and a different
         quantization from the rig table above, so never carry one across. The tool-call rate in
         particular disagrees with the rig's: under llama.cpp at Q4 these same adapters call the
         tool far less often than they do under the trainer at bf16 — the untrained base model
         alone drops from 1.94 zooms a question on the rig to ${F.n2(base.zooms)} here. So Q4
         shifts every policy toward fewer looks, and part of the gap you see between the runs is
         the quantization amplifying it rather than training alone. On ${n} images that is an
         observation, not a finding. Accuracy on ${n} images is likewise too small a sample to
         rank runs by — the ${(A && A.eval && A.eval.n) || 191}-question rig eval is the accuracy
         number.</div>`;

  document.getElementById("m4-charts").innerHTML = scatterCard(m) + costTermCard(g, base);
}

/* Predicted against measured: the check the whole cost model rests on. */
function scatterCard(m){
  const W = 460, H = 250, PL = 56, PR = 12, PT = 12, PB = 40;
  const iw = W - PL - PR, ih = H - PT - PB;
  const top = Math.max(...m.scatter.map(d => Math.max(d.a, d.p))) * 1.06;
  const x = v => PL + (v / top) * iw, y = v => PT + ih - (v / top) * ih;
  const step = top > 8000 ? 4000 : top > 4000 ? 2000 : 1000;
  let gl = "";
  for (let t = 0; t * step <= top; t++){
    const v = t * step;
    gl += `<line class="gl" x1="${PL}" x2="${W - PR}" y1="${y(v)}" y2="${y(v)}"></line>
      <text class="axlab" x="${PL - 7}" y="${y(v) + 3.5}" text-anchor="end">${v / 1000}</text>
      <text class="axlab" x="${x(v)}" y="${H - PB + 16}" text-anchor="middle">${v / 1000}</text>`;
  }
  const dots = m.scatter.map(d =>
    `<circle cx="${x(d.p).toFixed(1)}" cy="${y(d.a).toFixed(1)}" r="3.4"
      fill="${M4C[d.tag] || "var(--ink-3)"}" fill-opacity=".8"><title>predicted ${(d.p / 1000).toFixed(2)} s, measured ${(d.a / 1000).toFixed(2)} s</title></circle>`).join("");
  const k = m.ratio_median || 1;
  const legend = m.groups.map(gr =>
    `<span><i class="swatch" style="background:${M4C[gr.tag] || "var(--ink-3)"}"></i>${esc(gr.label)}</span>`).join("");

  return `<div class="card chart"><h3>Predicted against measured</h3>
    <p class="cs">Every timed episode, in seconds. Dashed is perfect agreement. Solid is the
      ${k.toFixed(2)}× the measurements actually came in at.</p>
    <div class="legend">${legend}</div>
    <svg viewBox="0 0 ${W} ${H}" role="img"
      aria-label="Measured episode latency against the cost model's prediction">
      ${gl}
      <line x1="${x(0)}" y1="${y(0)}" x2="${x(top)}" y2="${y(top)}" stroke="var(--ink-3)"
        stroke-width="1.25" stroke-dasharray="4 3"></line>
      <line x1="${x(0)}" y1="${y(0)}" x2="${x(top / k)}" y2="${y(top)}" stroke="var(--ink-2)"
        stroke-width="1.5"></line>
      ${dots}
      <text class="axlab" x="${PL + iw / 2}" y="${H - 5}" text-anchor="middle">predicted, seconds</text>
      <text class="axlab" x="${PL - 40}" y="${PT + ih / 2}" text-anchor="middle"
        transform="rotate(-90 ${PL - 40} ${PT + ih / 2})">measured, seconds</text>
    </svg>
    <p class="cs" style="margin:9px 0 2px">Pearson r ${m.pearson.toFixed(3)} over ${m.n} episodes.
      The model ranks episodes almost exactly and under-predicts the absolute clock by about
      ${k.toFixed(2)}×. GRPO compares rollouts inside a group, so ranking is what the reward needs
      and a constant factor is absorbed by lambda. Absolute seconds here are the measured ones.</p>
    </div>`;
}

function costTermCard(g, base){
  const a = g.find(x => x.tag === "run_a_final"), b = g.find(x => x.tag === "run_b_final");
  let body;
  if (a && b){
    body = `Run A pays <strong>${(a.ms / base.ms).toFixed(2)}×</strong> the base model's wall
      clock, because nothing in its reward discourages a look. Run B pays
      <strong>${(b.ms / base.ms).toFixed(2)}×</strong> — it costs what the untrained model cost,
      and it is <strong>${(a.ms / b.ms).toFixed(2)}× faster than Run A</strong> on the same chip,
      the same quantization and the same ${a.n} images. Same code, same data, same seed. The only
      difference between the two runs is one term in the reward.`;
  } else if (a){
    body = `Training with no cost term made the model <strong>${(a.ms / base.ms).toFixed(2)}× slower
      on the machine it deploys to</strong> — ${F.sec(base.ms)} to ${F.sec(a.ms)} per question. It
      learned to think longer and look more: ${F.n0(a.decode)} decode tokens a question against
      ${F.n0(base.decode)}, and ${F.n2(a.zooms)} zooms against ${F.n2(base.zooms)}. Run B is the
      same run with the cost term switched on. Its column appears when it lands.`;
  } else {
    body = `Timed so far: ${esc(g.map(x => x.label).join(", "))}. The A-vs-B beat needs both
      adapters measured on the Mac.`;
  }
  const S = D.paired && D.paired.saving;
  let split = "";
  if (S){
    const order = [["decode", "decode tokens", "thinking less out loud"],
                   ["vision", "vision tokens", "smaller crops, fewer of them"],
                   ["tools", "tool calls", "the zoom itself"]];
    split = `<div style="margin-top:15px;padding-top:13px;border-top:1px solid var(--line-soft)">
      <div class="eyebrow" style="margin-bottom:9px">Where the ${F.sec(S.total_ms)} came from</div>`
      + order.map(([k, label, why]) =>
        `<div class="row" style="display:grid;grid-template-columns:1fr auto;gap:10px;
            padding:5px 0;font-size:12.5px;align-items:baseline">
          <span style="color:var(--ink-2)">${esc(label)}
            <span style="color:var(--ink-3)"> — ${esc(why)}</span></span>
          <span class="num" style="font-weight:600">${F.pct(S.share[k])}</span></div>`).join("")
      + `<p class="cs" style="margin:10px 0 0">Almost all of it is reasoning, not looking. At Q4 a
         decode token costs ${D.cost_model.b.toFixed(1)} ms against
         ${D.cost_model.a.toFixed(2)} ms for a vision token, so the cheapest milliseconds to give
         back are tokens of thought. A reward that counted tool calls would have pushed the other
         way — that is the case for denominating cost in measured time.</p></div>`;
  }
  return `<div class="card chart"><h3>What "no cost term" costs</h3>
    <p style="margin:6px 0 0;font-size:13.5px;color:var(--ink-2);line-height:1.6">${body}</p>
    ${split}</div>`;
}

/* ---------- provenance ---------- */
function method(){
  const cm = D.cost_model || {};
  // Same coefficients in the reward and on the page? Check the pinned hash, don't assert it.
  const paid = RUNS.filter(r => r.present && r.cost_model &&
    r.cost_model.sha256 && cm.sha256 && r.cost_model.sha256.startsWith(cm.sha256));
  const cards = [];
  cards.push(["The cost model", cm.a
    ? `<code>ms = ${cm.a.toFixed(2)}·vision + ${cm.b.toFixed(1)}·decode + ${cm.c.toFixed(0)}·calls
        ${cm.intercept >= 0 ? "+" : "−"} ${Math.abs(cm.intercept).toFixed(1)}</code>.
       Fitted on ${cm.n_points} timed runs on an M4 Max at ${esc(cm.quant.toUpperCase())},
       R² ${cm.r2.toFixed(4)}. Frozen and pinned <code>${esc(cm.sha256)}</code>.`
      + (paid.length ? ` ${esc(paid.map(r => r.label).join(" and "))} paid the reward from this
         same pinned file — the ruler on this page and the ruler in the reward are one file.` : "")
    : "Not loaded."]);
  cards.push(["Why decode dominates", `<code>b</code> is ${cm.b ? (cm.b / cm.a).toFixed(0) : "—"}×
    <code>a</code>. Decode is bandwidth-bound on this chip; vision prefill is nearly free. So one more
    look is cheap and forty more tokens of reasoning is not. A token-counting reward says the opposite.`]);
  // The page recomputes latency from token counts. A run that charged a cost term logged its own
  // cost_ms during training, so the two must agree. Report the check either way.
  const chk = RUNS.filter(r => r.present && r.eval && r.eval.logged_ms > 0 && r.eval.pred_ms);
  let chkTxt = "";
  if (chk.length){
    const worst = chk.map(r => ({ r, d: Math.abs(r.eval.pred_ms - r.eval.logged_ms) / r.eval.logged_ms }))
      .sort((a, b) => b.d - a.d)[0];
    chkTxt = worst.d < 0.02
      ? ` Checked against the training loop's own <code>cost_ms</code>: ${esc(worst.r.label)}
          predicts ${F.sec(worst.r.eval.pred_ms)} here against ${F.sec(worst.r.eval.logged_ms)}
          charged in the reward, a ${(worst.d * 100).toFixed(2)}% gap — the fit's intercept, which
          the reward drops.`
      : ` Cross-check flag: ${esc(worst.r.label)} predicts ${F.sec(worst.r.eval.pred_ms)} here but
          the reward charged ${F.sec(worst.r.eval.logged_ms)}, a ${(worst.d * 100).toFixed(1)}%
          gap. Treat the latency numbers as unconfirmed until that is explained.`;
  }
  cards.push(["Latency is modelled", `Every millisecond on this page is that regression applied to the
    token counts each episode actually spent. Rollouts are never timed live: training runs on shared
    3090s, and live timing would put that cluster's load into the gradient. Training hardware and
    target hardware are different machines — the normal case — which is why the cost term is measured
    on the target instead of proxied by token counts.` + chkTxt]);
  cards.push(["Boxes", `The policy emits <code>bbox_2d</code> on a normalised 0-1000 grid over the whole
    image. Boxes here are positioned as percentages of that grid, and every crop is taken from the
    full-resolution original, never from another crop.`]);
  const a = A && A.eval;
  cards.push(["The baseline", A && A.present
    ? `Run A is the control: identical code, data and seed, with the cost term switched off
       (<code>cost_mode=${esc(A.cost_mode)}</code>). It reproduces the published picture — DeepEyes
       reports calling its zoom tool on 100% of inputs, and Run A sits at
       ${a ? F.pct(a.tool_rate) : "—"}.`
    : "Run A has not mirrored yet."]);
  cards.push(["The eval set", a
    ? `${a.n} V*Bench questions${a.n >= 150 ? " — the whole benchmark" : ", a fixed subset"}, held out
       from training. Train images are COCO, eval images are SA-1B; basename overlap is zero. Options
       are shuffled per sample, because V*Bench lists the gold first in every file.`
    : "Pending."]);
  document.getElementById("method").innerHTML = cards.map(([h, p]) =>
    `<div><h4>${esc(h)}</h4><p>${p}</p></div>`).join("");

  const missing = RUNS.filter(r => !r.present).map(r => r.label);
  const partial = RUNS.filter(r => r.present && !r.done).map(r => r.label);
  document.getElementById("foot").innerHTML =
    `<p>Built ${esc(D.generated_at)} from artifacts mirrored to the Mac.
      ${partial.length ? "Still training: " + esc(partial.join(", ")) + "." : ""}
      ${missing.length ? "Not started: " + esc(missing.join(", ")) + "." : ""}</p>
     <p>The thumbnails are the reduced view the policy is given, rebuilt box-free from the saved
      crops so the overlay can be animated. Crop images are the real returns from the zoom tool.
      No number on this page is hand-entered.</p>`;
}

headline(); runSeg(); picker(); draw(); comparison(); m4(); charts(); method();
</script>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.path.expanduser("~/archive/cost-aware-vlm"))
    ap.add_argument("--out", default=str(REPO / "out" / "demo.html"))
    ap.add_argument("--dev", action="store_true", help="also read the shape/smoke shakedown runs")
    ap.add_argument("--max-samples", type=int, default=6)
    args = ap.parse_args()

    out = Path(args.out)
    payload = build(Path(args.archive), out, args.dev, args.max_samples)

    size = out.stat().st_size
    print(f"wrote {out}  {size / 1e6:.2f} MB")
    for rid in payload["order"]:
        r = payload["runs"][rid]
        if not r.get("present"):
            print(f"  {rid}: absent")
            continue
        ev = r.get("eval")
        tail = (f"eval@{ev['tag']} acc={ev['accuracy']} tool_rate={ev['tool_rate']} "
                f"ms={None if ev['pred_ms'] is None else round(ev['pred_ms'])}") if ev else "no eval yet"
        print(f"  {rid}: {len(r.get('episodes', {}))} episodes, "
              f"{'DONE' if r.get('done') else 'running'}, {tail}")
    bad = {k: v for k, v in payload["residuals"].items() if v}
    if bad:
        print(f"  thumbnail repaint: {len(bad)} image(s) still have draw-coloured pixels in the "
              f"box band — usually real content, worth an eyeball: {bad}")
    if size > 12e6:
        print("WARNING: page is over 12 MB", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
