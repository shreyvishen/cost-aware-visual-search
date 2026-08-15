#!/usr/bin/env python3
"""Time Qwen3.5-VL-4B on the M4 Max and append every point to m4_measurements.csv.

Runs one measurement at a time. Never run two of these at once -- concurrent
Metal work poisons every number in the file.

Resumable: each row is appended and flushed the moment it lands, and a restart
skips (quant, config_id, rep) triples the CSV already has. A crash costs at most
one point.

Timing source: `llama-server`, not `llama-mtmd-cli`. Same libllama + libmtmd +
Metal backend and the same GGUFs, but this build's mtmd-cli prints no
`llama_perf` block, so it gives no way to separate prefill from decode. The
server returns an authoritative `timings` object per request:
`prompt_ms` (image encode + prefill) and `predicted_ms` (decode). It also holds
the model in memory, so 80 measurements cost one model load instead of 80.
The response variable is `prompt_ms + predicted_ms` -- the generation phases,
never process startup.

Usage:
    python3 measure.py --quant q4 [--reps 4] [--port 8199]
"""

import argparse
import base64
import csv
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vision_tokens import vision_tokens_for  # noqa: E402

CSV_PATH = os.path.join(HERE, "m4_measurements.csv")
RAW_DIR = os.path.join(HERE, "raw")

LLAMA_BIN = os.path.expanduser("~/code/forks/llama.cpp/build/bin/llama-server")
MODEL_DIR = os.path.expanduser("~/assets/models/qwen3.5-4b-gguf")
MMPROJ = os.path.join(MODEL_DIR, "mmproj-F16.gguf")
IMG_DIR = ("/private/tmp/claude-501/-Users-shreyvishen-code-personal-yc-expo-"
           "cost-aware-visual-search/c876d371-9d24-4700-b54a-11ba6542da6c/"
           "scratchpad/img")

QUANTS = {
    "q4": os.path.join(MODEL_DIR, "Qwen3.5-4B-Q4_K_M.gguf"),
    "q8": os.path.join(MODEL_DIR, "Qwen3.5-4B-Q8_0.gguf"),
    "bf16": os.path.join(MODEL_DIR, "Qwen3.5-4B-BF16.gguf"),
}

PROMPT = "Describe the image."

# (config_id, base_px, gen_tokens, n_crops, crop_px)
#
# A reduced but balanced slice of the GOAL sec.18 grid. The full grid is
# 5 px x 4 gen x 4 crops x 3 reps = 240 runs per quant, which does not fit three
# quants in the time available. These 20 points keep vision_tokens,
# decode_tokens and tool_calls independently varying so a, b and c stay
# identifiable: tool_calls=4 appears at 320 and at 2048 vision tokens,
# tool_calls=1 at 128 and at 1088, and every gen length appears with and without
# crops. fit_report.md lists what was dropped.
GRID = [
    # crops = 0: base image size x generation length
    ("c01", 256, 32, 0, 0),
    ("c02", 256, 512, 0, 0),
    ("c03", 384, 128, 0, 0),
    ("c04", 512, 32, 0, 0),
    ("c05", 512, 256, 0, 0),
    ("c06", 768, 128, 0, 0),
    ("c07", 768, 512, 0, 0),
    ("c08", 1024, 32, 0, 0),
    ("c09", 1024, 256, 0, 0),
    # crops > 0: crop count and crop size vary independently of the base image
    ("c10", 256, 128, 1, 256),
    ("c11", 256, 32, 4, 256),
    ("c12", 384, 256, 2, 384),
    ("c13", 512, 32, 1, 512),
    ("c14", 512, 512, 2, 256),
    ("c15", 512, 128, 4, 384),
    ("c16", 768, 32, 2, 512),
    ("c17", 768, 256, 4, 256),
    ("c18", 1024, 128, 1, 256),
    ("c19", 1024, 512, 2, 384),
    ("c20", 1024, 32, 4, 512),
]

# Held-out points. Every one is a cell of the GOAL sec.18 grid that the fit
# never saw, so scoring the frozen coefficients against these is a real test of
# the reduced design, not a re-read of the training residuals. `fit.py` excludes
# any config_id starting with "v".
VALIDATION = [
    ("v01", 384, 32, 1, 256),
    ("v02", 512, 128, 0, 0),
    ("v03", 768, 256, 1, 384),
    ("v04", 1024, 512, 4, 256),
    ("v05", 256, 256, 2, 512),
    ("v06", 384, 512, 4, 512),
    ("v07", 1024, 128, 2, 256),
    ("v08", 768, 32, 0, 0),
]

FIELDS = [
    "ts", "quant", "config_id", "rep", "is_warmup",
    "base_px", "gen_tokens", "n_crops", "crop_px",
    "vision_tokens", "decode_tokens", "tool_calls",
    "prompt_n", "prompt_ms", "predicted_n", "predicted_ms",
    "total_ms", "client_wall_ms", "cache_n",
]


def b64_image(path):
    with open(path, "rb") as fh:
        return "data:image/jpeg;base64," + base64.b64encode(fh.read()).decode()


def wait_healthy(port, proc, timeout_s=300):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early, rc={proc.returncode}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=3
            ) as r:
                if json.load(r).get("status") == "ok":
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError("llama-server did not become healthy in time")


def start_server(quant, port, log_path):
    cmd = [
        LLAMA_BIN, "-m", QUANTS[quant], "--mmproj", MMPROJ,
        "-ngl", "99", "--host", "127.0.0.1", "--port", str(port),
        "-c", "32768", "-np", "1", "--no-webui",
    ]
    log = open(log_path, "ab", buffering=0)
    log.write(f"\n=== {datetime.now(timezone.utc).isoformat()} {' '.join(cmd)}\n"
              .encode())
    proc = subprocess.Popen(cmd, stdout=log, stderr=log,
                            preexec_fn=os.setsid)
    wait_healthy(port, proc)
    return proc, log


def stop_server(proc, log):
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    try:
        log.close()
    except Exception:
        pass


def build_messages(images):
    """One image per user turn -- the shape the training conversation has.

    Stuffing every image into a single user message DOES NOT WORK in this
    llama-server build: it silently drops images (1 and 2 images both report
    prompt_n=80, and prompt_ms does not move). Verified 2026-08-15. One image
    per user turn is exactly linear instead: each extra image adds 64 vision
    tokens (at 256 px) plus a constant 17 text tokens for the turn wrapper.
    Measured prompt_n for 1/2/3/5 images of 256 px: 80 / 161 / 242 / 404.

    The short "ok" assistant turn keeps the replayed context minimal, so `c`
    measures the marginal cost of one more crop -- image encode plus turn
    wrapper -- and not an arbitrary amount of replayed reasoning text.
    """
    msgs = []
    for i, url in enumerate(images):
        msgs.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": PROMPT},
        ]})
        if i < len(images) - 1:
            msgs.append({"role": "assistant", "content": "ok"})
    return msgs


TEXT_OVERHEAD_FIRST = 16   # tokens the first image's turn adds
TEXT_OVERHEAD_EXTRA = 17   # tokens each additional image's turn adds


def one_run(port, images, gen_tokens):
    """One timed generation. Returns the server's timings dict + client wall."""
    body = {
        "messages": build_messages(images),
        "n_predict": gen_tokens,
        "temperature": 0,
        "cache_prompt": False,   # every run re-prefills; no free lunch from reuse
        "ignore_eos": True,      # exactly gen_tokens decode steps, always
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        resp = json.load(r)
    wall = (time.perf_counter() - t0) * 1000.0
    return resp["timings"], wall


def load_done():
    done = set()
    if not os.path.exists(CSV_PATH):
        return done
    with open(CSV_PATH, newline="") as fh:
        for row in csv.DictReader(fh):
            done.add((row["quant"], row["config_id"], int(row["rep"])))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", required=True, choices=sorted(QUANTS))
    ap.add_argument("--reps", type=int, default=4,
                    help="reps per config; rep 0 is a discarded warm-up")
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--validation", action="store_true",
                    help="run the held-out points instead of the fit grid")
    args = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    done = load_done()
    grid = VALIDATION if args.validation else GRID
    todo = [(c, r) for c in grid for r in range(args.reps)
            if (args.quant, c[0], r) not in done]
    if not todo:
        print(f"[{args.quant}] nothing to do, all points present")
        return

    new_file = not os.path.exists(CSV_PATH)
    csv_fh = open(CSV_PATH, "a", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()
        csv_fh.flush()
        os.fsync(csv_fh.fileno())

    cache = {}

    def images_for(base_px, n_crops, crop_px):
        paths = [f"{IMG_DIR}/img_{base_px}.jpg"]
        for i in range(1, n_crops + 1):
            paths.append(f"{IMG_DIR}/crop{i}_{crop_px}.jpg")
        out = []
        for p in paths:
            if p not in cache:
                cache[p] = b64_image(p)
            out.append(cache[p])
        return out

    log_path = os.path.join(RAW_DIR, f"server_{args.quant}.log")
    jsonl_path = os.path.join(RAW_DIR, f"timings_{args.quant}.jsonl")
    proc = log = None
    t_start = time.time()
    try:
        print(f"[{args.quant}] starting server ...", flush=True)
        proc, log = start_server(args.quant, args.port, log_path)

        # One untimed warm-up so Metal shader compilation and the first-touch
        # cost never land in a recorded row.
        one_run(args.port, images_for(512, 0, 0), 8)
        print(f"[{args.quant}] warm, {len(todo)} points to run", flush=True)

        for idx, ((cid, base_px, gen, n_crops, crop_px), rep) in enumerate(todo):
            imgs = images_for(base_px, n_crops, crop_px)
            t, wall = one_run(args.port, imgs, gen)
            vt = vision_tokens_for(base_px, base_px)
            vt += n_crops * (vision_tokens_for(crop_px, crop_px) if n_crops else 0)
            row = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "quant": args.quant, "config_id": cid, "rep": rep,
                "is_warmup": 1 if rep == 0 else 0,
                "base_px": base_px, "gen_tokens": gen,
                "n_crops": n_crops, "crop_px": crop_px,
                "vision_tokens": vt,
                "decode_tokens": t["predicted_n"],
                "tool_calls": n_crops,
                "prompt_n": t["prompt_n"],
                "prompt_ms": round(t["prompt_ms"], 3),
                "predicted_n": t["predicted_n"],
                "predicted_ms": round(t["predicted_ms"], 3),
                "total_ms": round(t["prompt_ms"] + t["predicted_ms"], 3),
                "client_wall_ms": round(wall, 3),
                "cache_n": t.get("cache_n", 0),
            }
            writer.writerow(row)
            csv_fh.flush()
            os.fsync(csv_fh.fileno())
            with open(jsonl_path, "a") as jf:
                jf.write(json.dumps({"row": row, "timings": t}) + "\n")
                jf.flush()

            # prompt_n must equal our vision count plus the known per-turn text
            # overhead. If it ever does not, the reward units are wrong.
            expect = vt + TEXT_OVERHEAD_FIRST + TEXT_OVERHEAD_EXTRA * n_crops
            flag = "" if t["prompt_n"] == expect else f"  !! prompt_n != {expect}"
            print(f"[{args.quant}] {idx + 1}/{len(todo)} {cid} rep{rep} "
                  f"vt={vt} dec={t['predicted_n']} tc={n_crops} "
                  f"total={row['total_ms']}ms{flag}", flush=True)
    finally:
        stop_server(proc, log)
        csv_fh.close()
        print(f"[{args.quant}] done in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
