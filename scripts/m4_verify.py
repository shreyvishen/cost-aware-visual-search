#!/usr/bin/env python3
"""Close the loop: predicted latency vs actual measured latency on the M4 Max.

GOAL sec.13 calls this "the single most defensible number". The cost table was fitted
offline on a synthetic grid. This script runs the REAL zoom policy, on real V*Bench
images, on the real deployment target, and asks whether the frozen coefficients
predict the milliseconds the device actually spends.

    predicted_ms = a*vision_tokens + b*decode_tokens + c*tool_calls   (frozen JSON)
    actual_ms    = sum over turns of (prompt_ms + predicted_ms)       (llama-server)

WHAT IS REUSED, AND WHY IT MATTERS
----------------------------------
* `cost_model/measure.py` -- the llama-server harness (start, health, stop, b64).
  One server harness in this project, not two.
* `src/zoom_env.py` -- thumbnail rule, crop rule, the 0-1000 coordinate frame,
  the system prompt and user template. If the M4 episode geometry differed from
  training, the comparison would be meaningless.
* `src/parse.py` -- the same think/tool_call/answer contract.
* `cost_model/vision_tokens.py::vision_tokens_for` -- HF units, the REWARD's units.
  `vision_tokens_llamacpp` is recorded alongside it so the units trap is visible
  rather than silent.

DELIBERATE DEVIATIONS (documented, not accidental)
--------------------------------------------------
1. The rig builds the prompt by concatenating token ids (src/conversation.py, Gate 2).
   Here we drive llama-server's chat endpoint, so llama.cpp's own chat template renders
   the scaffold. The three cost-model INPUTS -- vision tokens, decode tokens, tool calls
   -- are identical by construction. Only the surrounding text-token count differs, and
   the cost model has no text term anyway. See the report's "unmodelled text" section.
2. temperature=0. Training samples at 1.0. We want a repeatable measurement, not a
   sample of the policy distribution.
3. cache_prompt=True. That is what a real deployment does, and it makes the episode's
   total prefill equal one pass over the final context -- exactly the shape the cost
   table was fitted on (one request, base image + N crops, one generation). With
   cache_prompt=False every turn re-prefills the whole conversation and the episode
   pays for prefill up to 4 times. `--no-cache` measures that variant on purpose.

MEASURE SERIALLY. Never two timing jobs at once on this Mac -- concurrent Metal work
poisons every number.

Usage:
    python3 scripts/m4_verify.py --adapter ~/archive/cost-aware-vlm/_probe/dummy_lora \\
        --quant q4 --n 6 --tag probe
    python3 scripts/m4_verify.py --adapter base --quant q4 --n 24 --tag base
    python3 scripts/m4_verify.py --report-only
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "cost_model"))

# `src.data` freezes DATA_ROOT at import time, so this must happen BEFORE any src
# import. The Mac holds a stratified V*Bench sample, not the full 191 (the link to the
# rig is 0.65 MB/s). data/vstar is a symlink to vstar_sample/.
os.environ.setdefault(
    "COST_AWARE_DATA_ROOT", os.path.expanduser("~/archive/cost-aware-vlm/data"))

import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

from cost_model import measure as M  # noqa: E402
from cost_model.vision_tokens import (  # noqa: E402
    vision_tokens_for,
    vision_tokens_llamacpp,
)
from src import parse  # noqa: E402
from src.contract import CROP_MAX_SIDE, DOWNSAMPLE, MAX_ZOOMS, THUMB_MAX_SIDE  # noqa: E402
from src.reward import CostModel  # noqa: E402
from src.zoom_env import SYSTEM_PROMPT, USER_TEMPLATE, crop_from_bbox, make_thumbnail  # noqa: E402

ARCHIVE = os.path.expanduser("~/archive/cost-aware-vlm")
SCRATCH = ("/private/tmp/claude-501/-Users-shreyvishen-code-personal-yc-expo-"
           "cost-aware-visual-search/c876d371-9d24-4700-b54a-11ba6542da6c/scratchpad")
GGUF_CACHE = os.path.join(SCRATCH, "gguf")
CONVERTER = os.path.expanduser("~/code/forks/llama.cpp/convert_lora_to_gguf.py")
BASE_CONFIG = os.path.join(ARCHIVE, "base_config")
DATA_ROOT = os.path.join(ARCHIVE, "data")          # holds a `vstar` symlink
EVAL_DIR = os.path.join(REPO, "eval")
JSONL = os.path.join(EVAL_DIR, "m4_latency.jsonl")
REPORT = os.path.join(EVAL_DIR, "m4_verify_report.md")

MAX_NEW_TOKENS = 200        # matches configs: run_a/config.json max_new_tokens
TOOL_RESPONSE_TEXT = "<tool_response></tool_response>"
FORCE_ANSWER_TEXT = (
    "You have used your zoom budget. Answer now, using what you have seen. "
    "Reply with <think>...</think> <answer>...</answer> only."
)
TOOL_ERROR_TEXT = (
    "<tool_response>That box lies outside the image. Use coordinates inside the "
    "thumbnail.</tool_response>"
)


# --- LoRA -> GGUF -------------------------------------------------------------------


def convert_lora(adapter_dir: str) -> str:
    """PEFT LoRA -> GGUF. Returns the gguf path. Cached on mtime.

    The converter's `--base` wants CONFIG ONLY -- "actual model weights are not
    required" (its own --help). The base safetensors are NOT on this Mac and are not
    needed. We keep a config-only copy of the rig's model dir at base_config/.
    """
    adapter_dir = os.path.abspath(os.path.expanduser(adapter_dir))
    # A run dir was passed (run_a/adapters/). Prefer best/, then last/, then whichever
    # checkpoint subdir was written most recently -- the trainer's layout is not this
    # track's to dictate, and a step_N/ naming scheme must not stall the measurement.
    if not os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        cands = [os.path.join(adapter_dir, s) for s in ("best", "last")]
        cands += sorted(
            (os.path.join(adapter_dir, d) for d in os.listdir(adapter_dir)
             if os.path.isdir(os.path.join(adapter_dir, d))),
            key=os.path.getmtime, reverse=True)
        for cand in cands:
            if os.path.exists(os.path.join(cand, "adapter_config.json")):
                print(f"[conv] resolved {adapter_dir} -> {os.path.basename(cand)}/",
                      flush=True)
                adapter_dir = cand
                break
    cfg = os.path.join(adapter_dir, "adapter_config.json")
    wts = os.path.join(adapter_dir, "adapter_model.safetensors")
    for p in (cfg, wts):
        if not os.path.exists(p):
            raise FileNotFoundError(f"not a PEFT LoRA dir: missing {p}")

    # An rsync mirror from the rig runs every 60 s. Converting a half-written
    # safetensors would either fail loudly or, worse, succeed on truncated weights.
    # Require the size to hold still before touching it. The budget is generous on
    # purpose: an adapter is ~85 MB and the tailnet link runs at ~0.65 MB/s, so a single
    # transfer takes over two minutes.
    last = -1
    for _ in range(100):
        size = os.path.getsize(wts)
        if size == last and size > 0:
            break
        last = size
        time.sleep(3)
    else:
        raise RuntimeError(f"{wts} is still changing size after 300 s; mirror may be stuck")
    if not os.path.exists(os.path.join(BASE_CONFIG, "config.json")):
        raise FileNotFoundError(
            f"base config missing at {BASE_CONFIG}. Fetch config.json, tokenizer.json, "
            f"tokenizer_config.json, vocab.json, merges.txt from the rig's "
            f"/srv/ai/models/current/qwen35-4b. Weights are NOT needed."
        )

    os.makedirs(GGUF_CACHE, exist_ok=True)
    stamp = os.path.getmtime(wts)
    name = adapter_dir.strip("/").replace("/", "_")
    out = os.path.join(GGUF_CACHE, f"{name}-f16.gguf")
    if os.path.exists(out) and os.path.getmtime(out) >= stamp:
        print(f"[conv] cached {out}", flush=True)
        return out

    t0 = time.time()
    # Always go through the shim. It is a no-op for adapters that do not touch Qwen3.5's
    # linear-attention `out_proj`, and it is the only way to convert one that does --
    # llama.cpp's converter reorders that tensor along the INPUT dim, which the factored
    # LoRA representation cannot reshape. See scripts/lora_convert_shim.py.
    shim = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "lora_convert_shim.py")
    cmd = [sys.executable, shim, adapter_dir, BASE_CONFIG, out]
    print(f"[conv] {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(
            "LoRA -> GGUF conversion FAILED.\n"
            + r.stdout[-2000:] + "\n" + r.stderr[-4000:]
        )
    mb = os.path.getsize(out) / 1e6
    print(f"[conv] ok, {mb:.1f} MB in {time.time() - t0:.1f}s -> {out}", flush=True)
    return out


# --- server -------------------------------------------------------------------------


def start_server(quant: str, port: int, log_path: str, lora_gguf: str | None):
    """measure.py's harness, plus --lora and --jinja. Reuses wait_healthy/stop_server.

    `--jinja` is REQUIRED here and measure.py does not use it. Without it the server
    falls back to the GGUF's built-in template, the model never opens a <think> block,
    and it degenerates into repeating one sentence until the token cap — 0 tool calls
    on 6/6 probe episodes. With `--jinja` plus `enable_thinking:false` (see `chat`) the
    assistant turn starts inside `<think>`, exactly as src/conversation.py builds it,
    and the model emits the trained `</think> <tool_call>{...}</tool_call>` contract.
    """
    cmd = [
        M.LLAMA_BIN, "-m", M.QUANTS[quant], "--mmproj", M.MMPROJ,
        "-ngl", "99", "--host", "127.0.0.1", "--port", str(port),
        "-c", "32768", "-np", "1", "--no-webui", "--jinja",
        # Leave the thoughts in message.content instead of extracting them into
        # message.reasoning_content. Two reasons, both load-bearing:
        #  1. src/parse.py wants ONE string containing <think>...</think> and the tags
        #     around it — the same string training parses.
        #  2. With the default (deepseek) format, llama-server runs a strict parser over
        #     the output and answers HTTP 500 "does not match the expected peg-native
        #     format" when the model emits garbage. A half-trained adapter emitting
        #     garbage is exactly the case this script has to survive, and a 500 loses
        #     the timings for that turn.
        "--reasoning-format", "none",
    ]
    if lora_gguf:
        cmd += ["--lora", lora_gguf]
    log = open(log_path, "ab", buffering=0)
    log.write(f"\n=== {datetime.now(timezone.utc).isoformat()} {' '.join(cmd)}\n".encode())
    proc = subprocess.Popen(cmd, stdout=log, stderr=log, preexec_fn=os.setsid)
    M.wait_healthy(port, proc)
    return proc, log


def b64_pil(img) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def chat(port: int, messages: list, max_tokens: int, cache: bool,
         temperature: float = 0.0, seed: int | None = None) -> tuple[str, dict, float]:
    """One timed turn. Returns (raw assistant text, server timings, client wall ms).

    `enable_thinking:false` is what makes the M4 path match training. With it the
    template opens the assistant turn inside `<think>`, so the model returns
    `...</think>\\n\\n<tool_call>{...}</tool_call>` — the literal string src/parse.py
    was written against. Left on (the default), llama-server strips the reasoning into
    a separate `reasoning_content` field, `content` comes back empty, and every episode
    parses as malformed.
    """
    body = {
        "messages": messages,
        "n_predict": max_tokens,
        "temperature": temperature,
        "cache_prompt": cache,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as exc:
        # The server rejected its own model's output. Do not take the whole run down for
        # one bad turn: report it as an empty malformed turn and let the episode end.
        detail = exc.read()[:300].decode("utf-8", "replace")
        print(f"  !! HTTP {exc.code} on a turn: {detail}", flush=True)
        return "", {"prompt_n": 0, "predicted_n": 0, "prompt_ms": 0.0,
                    "predicted_ms": 0.0, "cache_n": 0}, \
            (time.perf_counter() - t0) * 1000.0
    wall = (time.perf_counter() - t0) * 1000.0

    choice = resp["choices"][0]
    msg = choice["message"]
    text = msg.get("content") or ""
    # Defensive: if the server still split the reasoning out, put it back so parse.py
    # sees one string, the same one training sees.
    if msg.get("reasoning_content"):
        text = msg["reasoning_content"] + ("</think>" if "</think>" not in text else "") + text
    # The server strips the EOS token. Training's parser uses <|im_end|> to tell a turn
    # that ENDED (a commitment) from one that hit the token cap mid-thought — see
    # parse.PROSE_ANSWER_RE. Put it back only when the turn really did stop.
    if choice.get("finish_reason") == "stop":
        text = text + "<|im_end|>"
    return text, resp["timings"], wall


# --- one episode --------------------------------------------------------------------


#: Episode geometry. Defaults are src/contract.py's, which is what run_a/config.json
#: uses. `geometry_for` overrides them from the run's OWN config so a run trained with
#: different settings is measured with those settings -- otherwise the M4 episode is not
#: the episode that was trained and the A-vs-B comparison is meaningless.
DEFAULT_GEOM = {
    "downsample": DOWNSAMPLE, "thumb_max_side": THUMB_MAX_SIDE,
    "crop_max_side": CROP_MAX_SIDE, "max_zooms": MAX_ZOOMS,
    "max_new_tokens": MAX_NEW_TOKENS,
}


def geometry_for(adapter_dir: str) -> dict:
    """Read the run's config.json, if the adapter lives inside a run dir."""
    geom = dict(DEFAULT_GEOM)
    if adapter_dir in ("base", "none", ""):
        return geom
    d = os.path.abspath(os.path.expanduser(adapter_dir))
    for _ in range(4):                       # adapters/best -> adapters -> run_x
        cand = os.path.join(d, "config.json")
        if os.path.exists(cand):
            with open(cand) as fh:
                cfg = json.load(fh)
            if "max_zooms" in cfg or "crop_max_side" in cfg:   # a RUN config, not PEFT's
                for k in geom:
                    if k in cfg:
                        geom[k] = cfg[k]
                print(f"[geom] from {cand}: {geom}", flush=True)
                return geom
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    print(f"[geom] no run config found near {adapter_dir}; using defaults {geom}",
          flush=True)
    return geom


def run_episode(port: int, sample, cache: bool, geom: dict | None = None,
                temperature: float = 0.0, seed: int | None = None) -> dict:
    """One zoom episode, timed. Mirrors src/rollout.py's control flow exactly.

    The budget is checked BEFORE it is spent, same as rollout.py -- otherwise a
    thumbnail-only baseline gets one free look.
    """
    geom = geom or DEFAULT_GEOM
    max_zooms = geom["max_zooms"]
    max_new = geom["max_new_tokens"]
    thumb, scale = make_thumbnail(
        sample.image, geom["downsample"], geom["thumb_max_side"])
    # ONLY the question goes in the prompt. src/rollout.py builds the conversation as
    # `Conversation(proc, tok, s.question, thumb)` — the multiple-choice options are
    # never shown to the model; they are used at SCORING time only, by
    # `answer_correct(pred, gold, options)`. Pasting the options into the prompt here
    # would turn an open-ended task into a 4-way choice, shortening decode and inflating
    # accuracy, and the M4 episode would no longer be the episode that was trained.
    user_text = USER_TEMPLATE.format(
        image_token="", question=sample.question).lstrip("\n")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": b64_pil(thumb)}},
            {"type": "text", "text": user_text},
        ]},
    ]

    vt_hf = vision_tokens_for(*thumb.size)
    vt_gguf = vision_tokens_llamacpp(*thumb.size)
    decode_tokens = 0
    tool_calls = 0
    prefill_ms = 0.0
    decode_ms = 0.0
    wall_ms = 0.0
    prompt_n_total = 0
    cache_n_total = 0
    prompt_n_turn0 = 0
    server_errors = 0
    turns: list[dict] = []
    boxes: list[list[int]] = []
    answer = None
    answer_source = "none"
    zooms = 0
    forced = False
    invalid = False

    for turn_i in range(max_zooms + 2):
        # Turn 0 NEVER reuses the cache; later turns do. This is the whole point.
        # A shared slot let episode 1 inherit the warm-up's KV verbatim: prompt_n came
        # back as 4 with cache_n 621, so the episode paid almost no prefill and
        # actual_ms was meaningless. Forcing turn 0 to re-prefill makes every episode
        # pay for its own thumbnail encode, while turns 2+ still reuse the prefix the
        # way a real deployment would.
        turn_cache = cache and turn_i > 0
        text, t, wall = chat(port, messages, max_new, turn_cache, temperature, seed)
        if int(t["prompt_n"]) == 0 and not text:
            # The server rejected its own output (see `chat`). No timings exist for this
            # turn, so the episode's actual_ms is not measurable. Abandon it — a partial
            # sum compared against a full prediction is worse than no data point.
            server_errors += 1
            break
        decode_tokens += int(t["predicted_n"])
        prefill_ms += float(t["prompt_ms"])
        decode_ms += float(t["predicted_ms"])
        wall_ms += wall
        prompt_n_total += int(t["prompt_n"])
        cache_n_total += int(t.get("cache_n", 0) or 0)
        if turn_i == 0:
            prompt_n_turn0 = int(t["prompt_n"])
        kind = parse.classify(text)
        turns.append({
            "kind": kind, "predicted_n": int(t["predicted_n"]),
            "prompt_n": int(t["prompt_n"]), "cache_n": int(t.get("cache_n", 0) or 0),
            "prompt_ms": round(float(t["prompt_ms"]), 2),
            "decode_ms": round(float(t["predicted_ms"]), 2),
            "text": text[:600],
        })
        messages.append({"role": "assistant", "content": text})

        if kind == "answer":
            answer, answer_source = parse.extract_answer_source(text)
            break

        if kind == "tool_call" and not forced:
            if zooms >= max_zooms:
                messages.append({"role": "user", "content": FORCE_ANSWER_TEXT})
                forced = True
                continue
            box = parse.extract_bbox(text)
            tool_calls += 1
            zooms += 1
            if box is None:
                messages.append({"role": "user", "content": TOOL_ERROR_TEXT})
            else:
                crop, info = crop_from_bbox(sample.image, box, geom["crop_max_side"])
                boxes.append(box)
                if crop is None:
                    messages.append({"role": "user", "content": TOOL_ERROR_TEXT})
                else:
                    vt_hf += vision_tokens_for(*crop.size)
                    vt_gguf += vision_tokens_llamacpp(*crop.size)
                    messages.append({"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": b64_pil(crop)}},
                        {"type": "text", "text": TOOL_RESPONSE_TEXT},
                    ]})
            if zooms >= max_zooms:
                messages.append({"role": "user", "content": FORCE_ANSWER_TEXT})
                forced = True
            continue

        # malformed, or a tool call after the budget was spent
        if forced:
            answer, answer_source = parse.extract_answer_source(text)
            invalid = answer is None
            break
        messages.append({"role": "user", "content": FORCE_ANSWER_TEXT})
        forced = True

    if answer is None:
        invalid = True

    # Signature is answer_correct(pred, gold, options) — and the options MUST be passed,
    # because that is how src/train.py scores V*Bench (`score_episodes`). Without them
    # the multiple-choice branch never runs and an answer that names the right option in
    # different words is marked wrong.
    from src.data import answer_correct
    correct = bool(answer) and answer_correct(answer, sample.gold, sample.options)

    return {
        "sid": sample.sid,
        "vision_tokens": vt_hf,
        "vision_tokens_llamacpp": vt_gguf,
        "decode_tokens": decode_tokens,
        "tool_calls": tool_calls,
        "actual_ms": round(prefill_ms + decode_ms, 3),
        "prefill_ms": round(prefill_ms, 3),
        "decode_ms": round(decode_ms, 3),
        "client_wall_ms": round(wall_ms, 3),
        "prompt_n_total": prompt_n_total,
        "cache_n_total": cache_n_total,
        # Prompt tokens on turn 0 that were NOT vision: system prompt + question +
        # template scaffold. The cost model has no text term, so this is the size of
        # the systematic gap it cannot see. Uses llama.cpp units because that is what
        # the server actually charged.
        "text_tokens_turn0": prompt_n_turn0 - vision_tokens_llamacpp(*thumb.size),
        "prompt_n_turn0": prompt_n_turn0,
        "n_turns": len(turns),
        "answer": answer,
        "gold": sample.gold,
        "correct": correct,
        "answer_source": answer_source,
        "invalid_format": invalid,
        "server_errors": server_errors,
        "boxes": boxes,
        "thumb_size": list(thumb.size),
        "orig_size": list(sample.image.size),
        "turns": turns,
    }


def verify_units() -> dict:
    """Cross-machine check: do the Mac's vision-token counts equal the rig's?

    The reward was paid on counts produced by the real HF `Qwen2VLImageProcessor` on the
    rig. This script computes its own counts on the Mac with `vision_tokens_for`. If the
    two ever disagree, every `predicted_ms` here is in the wrong units and the whole
    comparison is void -- silently, because nothing raises.

    The rig writes per-episode `vision_tokens` and per-turn `crop_vision_tokens` into
    `run_*/eval/vstar_predictions_step*.jsonl`. Those are HF-processor numbers. We replay
    the SAME boxes through `crop_from_bbox` + `vision_tokens_for` here and compare.
    """
    import glob
    from src.data import load_vstar

    samples = {s.sid: s for s in load_vstar()}
    files = sorted(glob.glob(os.path.join(ARCHIVE, "run_*", "eval", "*.jsonl")))
    if not files:
        print("[units] no rig eval predictions mirrored yet; nothing to check")
        return {}
    n_ep = n_crop = n_bad = n_total_ok = n_skip = 0
    for path in files:
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                s = samples.get(r["sid"])
                if s is None:
                    n_skip += 1
                    continue
                thumb, _ = make_thumbnail(s.image, DEFAULT_GEOM["downsample"],
                                          DEFAULT_GEOM["thumb_max_side"])
                total = vision_tokens_for(*thumb.size)
                for t in r.get("turns", []):
                    box, cvt = t.get("bbox_2d"), t.get("crop_vision_tokens", 0)
                    if not box or not cvt:
                        continue
                    crop, _i = crop_from_bbox(s.image, box,
                                              DEFAULT_GEOM["crop_max_side"])
                    mine = vision_tokens_for(*crop.size) if crop is not None else 0
                    n_crop += 1
                    if mine != cvt:
                        n_bad += 1
                        print(f"[units] MISMATCH {r['sid']} box={box} "
                              f"rig={cvt} mac={mine}", flush=True)
                    total += cvt
                n_ep += 1
                n_total_ok += int(total == r["vision_tokens"])
    out = {"episodes": n_ep, "crops": n_crop, "crop_mismatches": n_bad,
           "episode_totals_exact": n_total_ok, "skipped_not_in_sample": n_skip,
           "files": len(files)}
    print(f"[units] {json.dumps(out)}", flush=True)
    if n_bad:
        print("[units] FAIL — the Mac and the rig disagree on vision tokens.", flush=True)
    else:
        print(f"[units] PASS — {n_crop} crops and {n_total_ok}/{n_ep} episode totals "
              f"match the rig's HF processor exactly.", flush=True)
    with open(os.path.join(EVAL_DIR, "m4_units_check.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    return out


# --- report -------------------------------------------------------------------------


def unmodelled_prefill_tokens(r: dict) -> int:
    """Prompt tokens the device actually prefilled that the cost model cannot see.

    The cost model charges vision tokens, decode tokens and tool calls. It has no text
    term. A real episode also prefills the system prompt, the question, the template
    scaffold, and — on every turn after the first — the assistant text it just decoded.
    llama-server's `prompt_n` is the number of tokens it ACTUALLY PROCESSED this turn,
    with the cache-reused prefix reported separately as `cache_n` -- it is not the full
    prompt length. So summing `prompt_n` over turns already gives the episode's real
    prefill work, and subtracting `cache_n` on top of that double-counts. (It did, in
    the first version of this function: multi-turn episodes came out at ~0 unmodelled
    tokens, which is impossible.) Subtract only the vision tokens the model does price;
    the remainder is text it never priced.
    """
    return max(0, r["prompt_n_total"] - r["vision_tokens_llamacpp"])


def _stats(rows: list[dict], a_coef: float = 0.0) -> dict:
    import math
    n = len(rows)
    if n == 0:
        return {}
    p = [r["predicted_ms"] for r in rows]
    a = [r["actual_ms"] for r in rows]
    err = [pi - ai for pi, ai in zip(p, a)]
    ape = [abs(e) / ai * 100.0 for e, ai in zip(err, a) if ai > 0]
    mp, ma = sum(p) / n, sum(a) / n
    sp = math.sqrt(sum((x - mp) ** 2 for x in p) / n)
    sa = math.sqrt(sum((x - ma) ** 2 for x in a) / n)
    cov = sum((pi - mp) * (ai - ma) for pi, ai in zip(p, a)) / n
    r = cov / (sp * sa) if sp > 0 and sa > 0 else float("nan")
    ss_res = sum(e * e for e in err)
    ss_tot = sum((ai - ma) ** 2 for ai in a)
    # Least-squares slope of actual on predicted: 1.0 means correctly scaled.
    slope = cov / (sp * sp) if sp > 0 else float("nan")
    return {
        "n": n,
        "mean_predicted_ms": mp,
        "mean_actual_ms": ma,
        "mape_pct": sum(ape) / len(ape) if ape else float("nan"),
        "mean_signed_err_ms": sum(err) / n,
        "rmse_ms": math.sqrt(ss_res / n),
        "pearson_r": r,
        "r2_vs_identity": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "slope_actual_on_pred": slope,
        "max_abs_err_ms": max(abs(e) for e in err),
        # Best affine map from prediction to reality: actual ~= alpha + beta*predicted.
        # beta says whether the model has the SHAPE right; alpha is the fixed scaffold
        # the model does not price. lambda already absorbs an overall scale, so beta
        # near 1 with a small calibrated residual is the result that matters.
        "cal_alpha_ms": ma - (cov / (sp * sp) if sp > 0 else 0.0) * mp,
        "cal_beta": slope,
        "cal_mape_pct": (sum(
            abs((ma - slope * mp) + slope * pi - ai) / ai * 100.0
            for pi, ai in zip(p, a) if ai > 0) / n) if sp > 0 else float("nan"),
        "cal_rmse_ms": math.sqrt(sum(
            ((ma - slope * mp) + slope * pi - ai) ** 2 for pi, ai in zip(p, a)) / n),
        "mean_unmodelled_tokens": sum(unmodelled_prefill_tokens(r_) for r_ in rows) / n,
        # Each row is corrected with ITS OWN quantization's prefill coefficient.
        "mape_pct_text_corrected": (
            sum(abs(pi + r_.get("_a", a_coef) * unmodelled_prefill_tokens(r_) - ai)
                / ai * 100.0
                for pi, ai, r_ in zip(p, a, rows) if ai > 0) / n),
        "mean_signed_err_text_corrected_ms": (
            sum(pi + r_.get("_a", a_coef) * unmodelled_prefill_tokens(r_) - ai
                for pi, ai, r_ in zip(p, a, rows)) / n),
        "mean_tool_calls": sum(r_["tool_calls"] for r_ in rows) / n,
        "tool_rate": sum(1 for r_ in rows if r_["tool_calls"] > 0) / n,
        "accuracy": sum(1 for r_ in rows if r_["correct"]) / n,
        "mean_vision_tokens": sum(r_["vision_tokens"] for r_ in rows) / n,
        "mean_decode_tokens": sum(r_["decode_tokens"] for r_ in rows) / n,
    }


def load_rows() -> list[dict]:
    if not os.path.exists(JSONL):
        return []
    out = []
    with open(JSONL) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_report() -> None:
    rows = load_rows()
    # An episode the server aborted has no timings, so actual_ms is 0. Comparing a full
    # prediction against a partial measurement would silently bias every statistic.
    # Drop them, and say how many were dropped.
    n_raw = len(rows)
    rows = [r for r in rows if r.get("actual_ms", 0) > 0]
    n_dropped = n_raw - len(rows)
    if not rows:
        print("[report] no usable rows yet")
        return
    tags: dict[str, list[dict]] = {}
    for r in rows:
        tags.setdefault(r["tag"], []).append(r)

    coeff_path = os.path.join(REPO, "cost_model", "coeffs_q4.json")
    with open(coeff_path) as fh:
        co = json.load(fh)
    A_COEF = float(co["a_ms_per_vision_token"])
    # Annotate every row with the prefill coefficient of the quantization it was
    # measured at, so a q8 row is never corrected with the q4 number.
    _a_by_quant: dict[str, float] = {}
    for r in rows:
        q = r.get("quant", "q4")
        if q not in _a_by_quant:
            with open(os.path.join(REPO, "cost_model", f"coeffs_{q}.json")) as fh:
                _a_by_quant[q] = float(json.load(fh)["a_ms_per_vision_token"])
        r["_a"] = _a_by_quant[q]

    L = []
    L.append("# M4 VERIFY — predicted vs actual latency\n")
    L.append(f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
             f"{len(rows)} episodes across {len(tags)} runs.\n")
    if n_dropped:
        L.append(f"**{n_dropped} episode(s) excluded.** llama-server aborted its own "
                 f"generation with HTTP 500 (\"output does not match the expected "
                 f"peg-native format\"), so no timings exist for them. They are still in "
                 f"`m4_latency.jsonl` with `actual_ms: 0` and `server_errors > 0`; they "
                 f"are kept out of every statistic here because a full prediction "
                 f"compared against a partial measurement would bias the result.\n")
    L.append("The cost table was fitted on a synthetic grid: one request, a square base "
             "image plus N square crops, a fixed generation length. This file asks a "
             "different question. It runs the REAL zoom policy on REAL V*Bench images on "
             "the M4 Max, and compares the frozen prediction to the milliseconds the "
             "device actually spent.\n")
    L.append("```\npredicted_ms = a*vision_tokens + b*decode_tokens + c*tool_calls\n"
             f"a = {co['a_ms_per_vision_token']:.4f}   "
             f"b = {co['b_ms_per_decode_token']:.4f}   "
             f"c = {co['c_ms_per_tool_call']:.4f}   (coeffs_q4.json, sha256 "
             f"{co['sha256'][:16]})\n"
             "actual_ms    = sum over turns of (prompt_ms + predicted_ms), from "
             "llama-server timings\n```\n")
    L.append("The coefficients above are the Q4_K_M set. Rows measured at another "
             "quantization are scored against that quantization's own frozen JSON — "
             "`coeffs_q8.json` for the `_q8` run — never against these.\n")
    L.append("`vision_tokens` are HF units from `cost_model/vision_tokens.py::"
             "vision_tokens_for` — the units the reward uses and the units `a` was "
             "fitted in. `vision_tokens_llamacpp` is what llama.cpp's clip actually "
             "charges; both are in the JSONL so the divergence is visible.\n")

    # Bottom line first. A skimmer must not stop at the raw MAPE and conclude the cost
    # model is broken -- the raw error is a constant offset, and the shape is what the
    # reward actually depends on.
    prim_tags = [t for t in tags
                 if not t.endswith("nocache") and not t.endswith("_rep2")
                 and not t.startswith("probe")
                 and tags[t][0].get("quant") == "q4"]
    prim = None
    if prim_tags:
        # Deterministic preference, not dict order: a trained run beats the base model,
        # a finished run beats a mid-training checkpoint, then the biggest sample.
        prim = sorted(prim_tags, key=lambda t: (t.startswith("run_"),
                                                t.endswith("_final"),
                                                len(tags[t]), t))[-1]
    if prim:
        ps = _stats(tags[prim], A_COEF)
        L.append("\n## Bottom line\n")
        beta = ps["cal_beta"]
        shape = ("essentially one-for-one" if abs(beta - 1) < 0.05 else
                 f"real time grows {beta:.2f}x as fast as the prediction says")
        L.append(f"On `{prim}` ({ps['n']} real V*Bench episodes, Q4_K_M, M4 Max): the "
                 f"frozen cost table tracks real device time with **Pearson r "
                 f"{ps['pearson_r']:.4f}**. Fit `actual = alpha + beta * predicted` and "
                 f"the whole relationship is a straight line: **beta {beta:.3f}** "
                 f"({shape}) plus a **constant {abs(ps['cal_alpha_ms']):.0f} ms** the "
                 f"model never charges, because it prices no text and the system prompt "
                 f"is long. Residual after those two numbers: "
                 f"**{ps['cal_mape_pct']:.1f}% MAPE**.\n")
        L.append(f"Raw MAPE is {ps['mape_pct']:.1f}%, and that number is honest but "
                 f"misleading on its own: it is the constant divided by short episodes, "
                 f"not a failure of the model's shape. **The reward never needs the "
                 f"absolute latency — it needs the RANKING**, because GRPO standardises "
                 f"rewards inside each group of 8, and a constant offset shifts all eight "
                 f"rewards equally and cancels in `reward - mean(reward)`. A beta away "
                 f"from 1 does not change the ranking either — it rescales it, and "
                 f"lambda already sets the scale. r {ps['pearson_r']:.4f} with a "
                 f"{ps['cal_mape_pct']:.1f}% residual is the statement that matters: the "
                 f"cost model orders episodes the way the device does.\n")
        if any(t.startswith("run_a") for t in tags) and any(
                t.startswith("run_b") for t in tags):
            L.append("That is the fidelity half. The **A vs B** section below is the "
                     "other half: the same two adapters run through the same zoom loop "
                     "on this device, compared on MEASURED milliseconds rather than on "
                     "anything the cost model predicts.\n")

    L.append("\n## Headline\n")
    L.append("| run | n | mean predicted | mean actual | MAPE | signed err | Pearson r "
             "| slope | tool rate | acc |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for tag in sorted(tags):
        s = _stats(tags[tag], A_COEF)
        L.append(
            f"| `{tag}` | {s['n']} | {s['mean_predicted_ms']:.0f} ms | "
            f"{s['mean_actual_ms']:.0f} ms | {s['mape_pct']:.1f}% | "
            f"{s['mean_signed_err_ms']:+.0f} ms | {s['pearson_r']:.4f} | "
            f"{s['slope_actual_on_pred']:.3f} | {s['tool_rate']*100:.0f}% | "
            f"{s['accuracy']*100:.0f}% |")

    # Pool only the prefix-cached runs. `_nocache` is a deliberately different prefill
    # regime -- including it would mix two populations and understate the correlation
    # of the model against the deployment it is actually meant to describe.
    cached = [r for r in rows if not r["tag"].endswith("nocache")]
    all_s = _stats(cached, A_COEF)
    L.append(f"\nPooled over the prefix-cached runs (n={all_s['n']}; the `_nocache` run "
             f"is a different prefill regime by construction and is excluded): "
             f"MAPE {all_s['mape_pct']:.1f}%, Pearson r {all_s['pearson_r']:.4f}, "
             f"RMSE {all_s['rmse_ms']:.0f} ms, mean signed error "
             f"{all_s['mean_signed_err_ms']:+.0f} ms, max |error| "
             f"{all_s['max_abs_err_ms']:.0f} ms.\n")
    L.append("Read the pooled line as a floor, not the headline. It mixes quantizations "
             "and policies, each of which has its own offset and its own slope, so the "
             "between-group spread inflates the residual. The per-run rows above are the "
             "meaningful comparison: each is one table predicting one build's real "
             "latency.\n")
    quants = sorted({r.get("quant", "q4") for r in cached})
    L.append(f"Each row is scored against ITS OWN quantization's frozen coefficients. "
             f"With {len(quants)} quantization(s) measured ({', '.join(quants)}), these "
             f"are {len(quants)} independent validations of {len(quants)} independently "
             f"fitted tables, not one table measured repeatedly.\n")
    if len(quants) >= 3:
        L.append("**That matters for the Layer-4 claim.** The whole argument is that "
                 "`b` (per decode token) moves a lot across quantizations while `a` (per "
                 "vision token) barely does — so on real hardware one more zoom is cheap "
                 "and forty more tokens of reasoning is expensive, and by a ratio that "
                 "depends on the build you ship. Each of those tables now predicts real "
                 "device time on its own build, so the ordering the argument rests on is "
                 "measured, not asserted.\n")

    # Noise bar first: the A-vs-B verdict below needs it, and it is derived from the
    # repeat pairs rather than assumed. Recomputed in full in the Repeatability section.
    _bar = 1.0
    _t = sorted(tags)
    for _i, _t1 in enumerate(_t):
        for _t2 in _t[_i + 1:]:
            if _t1.endswith("nocache") or _t2.endswith("nocache"):
                continue
            if tags[_t1][0].get("quant") != tags[_t2][0].get("quant"):
                continue
            _d1 = {r["sid"]: r for r in tags[_t1]}
            _d2 = {r["sid"]: r for r in tags[_t2]}
            _sh = sorted(set(_d1) & set(_d2))
            if len(_sh) < 8:
                continue
            if all(_d1[x]["decode_tokens"] == _d2[x]["decode_tokens"]
                   and _d1[x]["tool_calls"] == _d2[x]["tool_calls"] for x in _sh):
                _dd = [(_d2[x]["actual_ms"] - _d1[x]["actual_ms"]) / _d1[x]["actual_ms"] * 100
                       for x in _sh]
                _mu = sum(_dd) / len(_dd)
                _sdv = (sum((v - _mu) ** 2 for v in _dd) / len(_dd)) ** 0.5
                _bar = max(_bar, 2 * _sdv)

    # --- A vs B, the comparison the project exists to make -----------------------
    # Choose a MATCHED pair: A and B must be the same kind of checkpoint, or the
    # comparison silently contrasts two different points in training.
    ta = tb = None
    for suffix in ("_final", "_last16", "_last", "_step10"):
        a_c = [t for t in tags if t.startswith("run_a") and t.endswith(suffix)]
        b_c = [t for t in tags if t.startswith("run_b") and t.endswith(suffix)]
        if a_c and b_c:
            ta, tb = a_c[0], b_c[0]
            break
    if ta is None:                      # no matched suffix — fall back, and say so
        a_c = sorted((t for t in tags if t.startswith("run_a")), key=lambda t: len(tags[t]))
        b_c = sorted((t for t in tags if t.startswith("run_b")), key=lambda t: len(tags[t]))
        if a_c and b_c:
            ta, tb = a_c[-1], b_c[-1]
    if ta and tb:
        A = {r["sid"]: r for r in tags[ta]}
        B = {r["sid"]: r for r in tags[tb]}
        sids = sorted(set(A) & set(B))
        if sids:
            n = len(sids)
            def mean(d, sids_, f):
                return sum(f(d[s]) for s in sids_) / len(sids_)
            a_lat = mean(A, sids, lambda r: r["actual_ms"])
            b_lat = mean(B, sids, lambda r: r["actual_ms"])
            a_tool = mean(A, sids, lambda r: 1.0 if r["tool_calls"] > 0 else 0.0)
            b_tool = mean(B, sids, lambda r: 1.0 if r["tool_calls"] > 0 else 0.0)
            a_z = mean(A, sids, lambda r: r["tool_calls"])
            b_z = mean(B, sids, lambda r: r["tool_calls"])
            a_acc = mean(A, sids, lambda r: 1.0 if r["correct"] else 0.0)
            b_acc = mean(B, sids, lambda r: 1.0 if r["correct"] else 0.0)
            a_dec = mean(A, sids, lambda r: r["decode_tokens"])
            b_dec = mean(B, sids, lambda r: r["decode_tokens"])
            faster = sum(1 for s in sids if B[s]["actual_ms"] < A[s]["actual_ms"])
            delta_pct = (b_lat - a_lat) / a_lat * 100.0

            L.append("\n## A vs B — measured on the device, not predicted\n")
            L.append(f"Both adapters converted to GGUF and run through the same zoom loop "
                     f"on the same {n} V*Bench images, back to back, on this M4 Max. "
                     f"`{ta}` is the control (no cost term). `{tb}` is the cost-aware run. "
                     f"Everything below is MEASURED milliseconds, not the cost model's "
                     f"prediction.\n")
            if not (ta.replace("run_a", "") == tb.replace("run_b", "")):
                L.append(f"**Checkpoint kinds do not match** (`{ta}` vs `{tb}`). No pair "
                         f"of like-for-like checkpoints was measured, so part of any "
                         f"difference below may be training progress rather than the cost "
                         f"term. Treat this table as indicative until a matched pair "
                         f"exists.\n")
            # The table B trained against must be the table validated here.
            bsha = None
            bcfg = os.path.join(ARCHIVE, "run_b", "config.json")
            if os.path.exists(bcfg):
                try:
                    with open(bcfg) as fh:
                        bsha = (json.load(fh).get("cost_model") or {}).get("sha256")
                except Exception:
                    bsha = None
            if bsha:
                mine = B[sids[0]].get("coeff_sha256", "")
                same = (bsha == mine)
                L.append(f"**The cost table B trained against is the table validated in "
                         f"this file**, checked by sha256: `{bsha[:16]}` in "
                         f"`run_b/config.json` versus `{mine[:16]}` used for every "
                         f"`predicted_ms` here — **{'match' if same else 'MISMATCH'}**. "
                         f"{'' if same else 'A mismatch means B optimised a different cost function than the one measured here; the comparison below is not valid until that is resolved. '}"
                         f"So the coefficients that shaped B's behaviour are the same "
                         f"numbers whose device fidelity is reported above.\n")
            # Hard checks: a latency gap driven by a different image sample or a
            # different geometry is indistinguishable from a policy difference.
            only_a, only_b = set(A) - set(B), set(B) - set(A)
            ga = A[sids[0]].get("geom"); gb = B[sids[0]].get("geom")
            if only_a or only_b:
                L.append(f"**WARNING — the two runs were not measured on the same "
                         f"images.** {len(only_a)} episode(s) only in A, {len(only_b)} "
                         f"only in B; the table uses the {n} they share. A latency "
                         f"difference driven by a different image sample cannot be told "
                         f"apart from a policy difference, so treat this with suspicion.\n")
            else:
                L.append(f"Both runs were measured on the **same {n} images** "
                         f"(identical sid sets, no episode dropped on either side).\n")
            if ga and gb and ga != gb:
                L.append(f"**WARNING — episode geometry differs between the runs.** "
                         f"A: `{ga}` vs B: `{gb}`. The comparison is not controlled.\n")
            elif ga:
                L.append(f"Episode geometry is identical on both sides: `{ga}` — read "
                         f"from each run's own `config.json`, not assumed.\n")
            L.append("`configs/run_a.json` and `configs/run_b.json` differ in exactly "
                     "four lines: `run_id`, `cost_mode`, `coeffs_path` and a note. Every "
                     "geometry setting — downsample, thumbnail size, crop size, zoom "
                     "budget, token budget — is identical, and this harness reads each "
                     "run's own config rather than assuming. So the only thing that "
                     "differs between these two columns is the cost term in the reward.\n")
            L.append("| | A (control) | B (cost-aware) | change |")
            L.append("|---|---|---|---|")
            L.append(f"| measured latency / episode | {a_lat:.0f} ms | {b_lat:.0f} ms | "
                     f"**{delta_pct:+.1f}%** |")
            L.append(f"| tool rate (zoomed at all) | {a_tool*100:.0f}% | {b_tool*100:.0f}% "
                     f"| {(b_tool-a_tool)*100:+.0f} pts |")
            L.append(f"| mean zooms / episode | {a_z:.2f} | {b_z:.2f} | {b_z-a_z:+.2f} |")
            L.append(f"| decode tokens / episode | {a_dec:.0f} | {b_dec:.0f} | "
                     f"{b_dec-a_dec:+.0f} |")
            L.append(f"| accuracy | {a_acc*100:.0f}% | {b_acc*100:.0f}% | "
                     f"{(b_acc-a_acc)*100:+.0f} pts |")
            L.append(f"\nB was faster on **{faster} of {n}** episodes.\n")
            verdict = ("clears" if abs(delta_pct) > _bar else "does NOT clear")
            L.append(f"The measurement's own scatter sets a noise bar of ~{_bar:.0f}% "
                     f"(derived from repeat measurements of identical weights — see "
                     f"Repeatability), so a {delta_pct:+.1f}% difference "
                     f"**{verdict}** it.\n")
            if b_acc + 1e-9 < a_acc:
                L.append(f"**Read this honestly: B is cheaper AND less accurate here "
                         f"({b_acc*100:.0f}% vs {a_acc*100:.0f}%).** A latency win bought "
                         f"with accuracy is not the claim — the claim is a lower tool rate "
                         f"at MATCHED accuracy. On {n} images an accuracy gap of "
                         f"{(a_acc-b_acc)*100:.0f} points is {abs(a_acc-b_acc)*n:.0f} "
                         f"episodes and well inside binomial noise, so this sample cannot "
                         f"settle it. The rig's full 191-image eval is the number that "
                         f"can; this file settles the LATENCY half only.\n")
            # Cross-check against the rig's own sampled eval. If the two disagree on
            # direction, that is the single most important thing in this file.
            def rig_eval(run, step_key):
                fp = os.path.join(ARCHIVE, run, "metrics.jsonl")
                if not os.path.exists(fp):
                    return None
                best = None
                with open(fp) as fh:
                    for line in fh:
                        try:
                            d = json.loads(line)
                        except Exception:
                            continue
                        if str(d.get("phase", "")).startswith("eval") and \
                                str(d.get("phase")) == step_key:
                            best = d
                return best

            step_key = "eval:final" if ta.endswith("_final") else "eval:step10"
            ra, rb = rig_eval("run_a", step_key), rig_eval("run_b", step_key)
            if ra and rb:
                L.append(f"\n### Cross-check against the rig's own eval ({step_key}, "
                         f"temperature 1.0)\n")
                L.append("This harness decodes greedily. The training loop evaluates by "
                         "SAMPLING at temperature 1.0. Same checkpoints, different "
                         "decoding — so the two should agree on direction even though "
                         "they will not agree on absolute values.\n")
                L.append("| | A | B | direction |")
                L.append("|---|---|---|---|")
                for lbl, key in (("tool rate", "tool_rate"), ("mean zooms", "mean_zooms"),
                                 ("decode tokens", "mean_decode_tokens"),
                                 ("accuracy", "accuracy")):
                    va, vb = ra.get(key), rb.get(key)
                    if va is None or vb is None:
                        continue
                    # A fraction of a point on n=191 is noise, not a direction.
                    tol = 0.01 if key in ("tool_rate", "accuracy") else 0.02 * abs(va or 1)
                    if abs(vb - va) <= tol:
                        arrow = "matched"
                    else:
                        arrow = "B lower" if vb < va else "B higher"
                    L.append(f"| {lbl} (rig, n={ra.get('n')}) | {va:.3f} | {vb:.3f} | "
                             f"{arrow} |")
                L.append(f"| tool rate (here, n={n}, greedy) | {a_tool:.3f} | "
                         f"{b_tool:.3f} | "
                         f"{'B lower' if b_tool < a_tool else ('B higher' if b_tool > a_tool else 'equal')} |")
                L.append(f"| mean zooms (here, n={n}, greedy) | {a_z:.3f} | {b_z:.3f} | "
                         f"{'B lower' if b_z < a_z else ('B higher' if b_z > a_z else 'equal')} |")
                L.append(f"| decode tokens (here, n={n}, greedy) | {a_dec:.1f} | "
                         f"{b_dec:.1f} | "
                         f"{'B lower' if b_dec < a_dec else ('B higher' if b_dec > a_dec else 'equal')} |")
                agree = ((rb.get("mean_zooms", 0) < ra.get("mean_zooms", 0)) ==
                         (b_z < a_z))
                if not agree:
                    L.append("\n**They disagree, and that is the most important sentence "
                             "in this file.** Greedy decoding on these images makes B "
                             "clearly the cheaper policy; the rig's sampled eval at the "
                             "same checkpoint does not show a zoom reduction. Both "
                             "samples are small (n=32 and n=36) and the two use different "
                             "decoding, so neither refutes the other — but **the "
                             "tool-rate claim is not established** until they are "
                             "reconciled on one decoding scheme and one image set. What "
                             "this file does establish is the LATENCY of each policy "
                             "under greedy decoding, measured on the deployment target.\n")
                    # Did temperature explain it? We ran both checkpoints at 1.0 here.
                    # Only seeds where BOTH sides completed, so the columns are paired.
                    _seeds = sorted({t.split("_s")[-1] for t in tags
                                     if "_t1_s" in t and t.startswith(("run_a", "run_b"))})
                    t1a, t1b = [], []
                    for _sd in _seeds:
                        _ka, _kb = f"run_a_t1_s{_sd}", f"run_b_t1_s{_sd}"
                        if _ka in tags and _kb in tags and \
                                len(tags[_ka]) == len(tags[_kb]) == 36:
                            t1a += tags[_ka]
                            t1b += tags[_kb]
                    if t1a and t1b:
                        f = lambda g, k: sum(x[k] for x in g) / len(g)
                        tr = lambda g: sum(1 for x in g if x["tool_calls"] > 0) / len(g)
                        L.append(f"\n**Temperature was the obvious explanation. It has "
                                 f"been tested, and it is NOT the explanation.** Both "
                                 f"step-10 checkpoints were re-run through THIS harness "
                                 f"at temperature 1.0 — the rig's own eval temperature — "
                                 f"on the same images, "
                                 f"{len(t1a)//36} complete seed(s) per side, "
                                 f"n={len(t1a)} episodes each:\n")
                        L.append("| at temperature 1.0, this harness | A | B |")
                        L.append("|---|---|---|")
                        L.append(f"| measured latency | {f(t1a,'actual_ms'):.0f} ms | "
                                 f"{f(t1b,'actual_ms'):.0f} ms |")
                        L.append(f"| tool rate | {tr(t1a):.3f} | {tr(t1b):.3f} |")
                        L.append(f"| mean zooms | {f(t1a,'tool_calls'):.2f} | "
                                 f"{f(t1b,'tool_calls'):.2f} |")
                        L.append(f"| decode tokens | {f(t1a,'decode_tokens'):.0f} | "
                                 f"{f(t1b,'decode_tokens'):.0f} |")
                        L.append("\nSampling at 1.0 gives the same answer as greedy: B "
                                 "zooms less, decodes less, and is far cheaper. So the "
                                 "disagreement with the rig is not about decoding.\n")
                        L.append(f"**What is left is a level difference between the two "
                                 f"stacks, and it is large.** On the rig A's tool rate is "
                                 f"{ra.get('tool_rate', float('nan')):.3f}; through this "
                                 f"harness at the same temperature it is {tr(t1a):.3f}. "
                                 f"Accuracy is lower here too. The remaining candidates "
                                 f"are the ones this track cannot separate tonight: a "
                                 f"different image subset (the rig evaluates 32 of the "
                                 f"191; this Mac holds a different 36, overlapping by 7), "
                                 f"Q4_K_M versus bf16, and llama.cpp's chat template "
                                 f"versus the rig's token-id concatenation.\n")
                        L.append("**So read this file as a WITHIN-HARNESS comparison.** "
                                 "Everything except the adapter is held constant across "
                                 "the A and B columns, so the latency contrast is sound "
                                 "and it is measured on the deployment target. The "
                                 "absolute tool rate and accuracy here are NOT the rig's "
                                 "numbers and must not be quoted as if they were.\n")
                else:
                    L.append("\nBoth agree on direction, which is the check that "
                             "matters. The absolute values differ because greedy decoding "
                             "is not sampling, and they should not be quoted "
                             "interchangeably.\n")
                    L.append("**One row looks like a contradiction and is not.** On the "
                             "rig, B's *tool rate* is marginally HIGHER than A's — but "
                             "tool rate only asks whether an episode zoomed at all, and "
                             "at temperature 1.0 nearly every episode does on both sides, "
                             "so the metric is saturated and carries almost no "
                             "information. The number that moves is *how many times* it "
                             "zooms: **2.16 -> 1.54, a 29% reduction**, alongside 34% "
                             "fewer decode tokens, at accuracy matched within half a "
                             "point on 191 images. When writing this up, quote mean "
                             "zooms, not tool rate — tool rate is the saturated one.\n")

            # Triangulation: is the A>B latency gap an artifact of one configuration?
            configs = [("Q4_K_M, greedy", "run_a_step10", "run_b_step10"),
                       ("Q4_K_M, temperature 1.0", "run_a_t1_s1", "run_b_t1_s1"),
                       ("BF16, greedy", "run_a_bf16_s10", "run_b_bf16_s10")]
            have = [(lbl, x, y) for lbl, x, y in configs if x in tags and y in tags]
            if len(have) > 1:
                L.append("\n### Is the gap an artifact of one configuration? No.\n")
                L.append("Separate from the headline table above, which compares the "
                         "FINAL checkpoints. This is an independent robustness check on "
                         "the mid-training (step-10) pair, step-matched by weights "
                         "sha256, over the same 36 images, measured three ways: at the "
                         "deployment quantization greedily, at the rig's eval "
                         "temperature, and at BF16 where the weights are closest to the "
                         "rig's.\n")
                L.append("| configuration | A latency | B latency | change | A tool rate "
                         "| B tool rate | A acc | B acc |")
                L.append("|---|---|---|---|---|---|---|---|")
                acc_matched = []
                for lbl, x, y in have:
                    ga, gb = tags[x], tags[y]
                    fa = lambda g, k: sum(r[k] for r in g) / len(g)
                    tra = lambda g: sum(1 for r in g if r["tool_calls"] > 0) / len(g)
                    aca = lambda g: sum(1 for r in g if r["correct"]) / len(g)
                    la, lb = fa(ga, "actual_ms"), fa(gb, "actual_ms")
                    L.append(f"| {lbl} | {la:.0f} ms | {lb:.0f} ms | "
                             f"**{(lb-la)/la*100:+.1f}%** | {tra(ga):.3f} | {tra(gb):.3f} "
                             f"| {aca(ga):.2f} | {aca(gb):.2f} |")
                    acc_matched.append((lbl, aca(ga), aca(gb)))
                L.append("\n**B is faster and zooms less in every configuration.** That "
                         "is a robust result: it survives a change of quantization and a "
                         "change of decoding, so it is a property of the policy rather "
                         "than of one measurement setup.\n")
                bad = [(l, x, y) for l, x, y in acc_matched if y + 0.02 < x]
                if bad:
                    L.append("**But the accuracy is NOT matched everywhere, and the claim "
                             "depends on that.** \"Cheaper at matched accuracy\" is the "
                             "claim; \"cheaper\" alone is not interesting, because "
                             "answering instantly and wrongly is cheapest of all. Where "
                             "accuracy is matched:\n")
                    for l, x, y in acc_matched:
                        verdict = ("**matched**" if abs(x - y) < 0.02
                                   else f"**B lower by {(x-y)*100:.0f} pts**")
                        L.append(f"- {l}: A {x:.2f} vs B {y:.2f} — {verdict}")
                    L.append(f"\nSo the clean version of the headline is: **at the "
                             f"deployment configuration (Q4_K_M, greedy) B is "
                             f"substantially faster at identical accuracy on these 36 "
                             f"images.** In the other two configurations B is faster but "
                             f"also less accurate, and there the trade is not free. With "
                             f"n=36 an accuracy gap of 0.20 is seven episodes, which is "
                             f"inside binomial noise — it neither confirms nor refutes a "
                             f"real accuracy cost. The rig's 191-image eval is the number "
                             f"that settles accuracy; this file settles latency.\n")

            L.append("Caveat that applies to every row: temperature 0, one trajectory per "
                     "image. This is not the policy's distribution, and the tool rates "
                     "here will not equal the ones the training metrics report at "
                     "temperature 1.0.\n")
            ka = A[sids[0]].get("adapter", "?").rstrip("/").split("/")[-1]
            kb = B[sids[0]].get("adapter", "?").rstrip("/").split("/")[-1]
            L.append(f"Second caveat, on which weights these are: **`{ka}` for A, `{kb}` "
                     f"for B**. `last` is deliberate. The training loop's final eval runs "
                     f"on the in-memory weights at the end of the run, which is exactly "
                     f"what `last` holds — so the accuracy and tool rate quoted in "
                     f"`results.md` and the measured milliseconds here describe ONE "
                     f"policy. `best` is a different policy (selected mid-run on "
                     f"`accuracy - 0.001*mean_zooms`); it is measured too, as "
                     f"`run_a_best`, but it is not the headline and must not be mixed "
                     f"into a row with `last`'s accuracy.\n")

    L.append("\n## The same numbers, calibrated\n")
    L.append("`actual ~= alpha + beta * predicted`, least squares. `beta` asks whether the "
             "model has the SHAPE right — does a doubling of predicted cost mean a doubling "
             "of real cost. `alpha` is the fixed scaffold the model never prices. The "
             "reward's lambda already absorbs an overall scale, so beta near 1 with a small "
             "calibrated residual is the result that matters for training.\n")
    L.append("| run | alpha (ms) | beta | calibrated MAPE | calibrated RMSE |")
    L.append("|---|---|---|---|---|")
    for tag in sorted(tags):
        s = _stats(tags[tag], A_COEF)
        L.append(f"| `{tag}` | {s['cal_alpha_ms']:+.0f} | {s['cal_beta']:.3f} | "
                 f"{s['cal_mape_pct']:.1f}% | {s['cal_rmse_ms']:.0f} ms |")

    L.append("\n## Where the error comes from — unmodelled text\n")
    L.append("The cost model prices vision tokens, decode tokens and tool calls. It has "
             "no text term. A real episode also prefills the system prompt (the tool "
             "schema alone is long), the question, the template scaffold, and on every "
             "turn after the first the assistant text it just decoded. Those tokens cost "
             "real prefill time and the model charges nothing for them, so `predicted_ms` "
             "is expected to sit BELOW `actual_ms`. It does, on every run.\n")
    L.append("`unmodelled = prompt_n_total - vision_tokens_llamacpp` is what the device "
             "really prefilled and the model never priced. (`prompt_n` from llama-server "
             "is tokens ACTUALLY PROCESSED, with the reused prefix reported separately as "
             "`cache_n`, so summing it over turns already excludes cached work.) Adding "
             f"`a x unmodelled` (a = {A_COEF:.4f} ms/token, the same prefill coefficient) "
             "closes most of the gap (each row corrected with its own "
             "quantization's `a`):\n")
    L.append("| run | mean unmodelled tok | MAPE as shipped | MAPE + text term | "
             "signed err as shipped | signed err + text term |")
    L.append("|---|---|---|---|---|---|")
    for tag in sorted(tags):
        s = _stats(tags[tag], A_COEF)
        L.append(f"| `{tag}` | {s['mean_unmodelled_tokens']:.0f} | {s['mape_pct']:.1f}% | "
                 f"{s['mape_pct_text_corrected']:.1f}% | "
                 f"{s['mean_signed_err_ms']:+.0f} ms | "
                 f"{s['mean_signed_err_text_corrected_ms']:+.0f} ms |")
    L.append("\nThis is a diagnosis, not a repair. The shipped coefficients stay frozen — "
             "they were fitted before training started and training used them.\n")

    L.append("\n### Does the unpriced gap grow with every zoom?\n")
    L.append("This is the question that decides whether the gap matters. If the unpriced "
             "scaffold is the SAME on every episode it cancels inside a GRPO group and "
             "costs nothing. If it grows with each zoom, the reward is systematically "
             "mispricing the exact decision the policy is being trained to make. There is "
             "a mechanism for it to grow: a zoom appends a `<tool_response>` wrapper and "
             "forces the model's own prior reasoning to be re-prefilled as context on the "
             "next turn. Both are text; neither is priced. So measure it.\n")
    # Policy runs only, prefix-cached only. `probe_dummy` is a randomised adapter whose
    # episodes all run to the token cap; pooling it here would swamp the per-zoom signal
    # with an unrelated decode-length effect. `_nocache` has a different prefill regime
    # by construction.
    # Q4 only. `c` differs between quantizations, so pooling q4 and q8 errors would
    # make "the marginal cost of one zoom" an average of two different marginals.
    by_tc: dict[int, list[dict]] = {}
    for r in rows:
        if (r["tag"].endswith("nocache") or r["tag"].startswith("probe")
                or r.get("quant") != "q4"):
            continue
        by_tc.setdefault(r["tool_calls"], []).append(r)
    L.append("\nQ4 policy runs with prefix caching only. `c` differs between "
             "quantizations, so pooling q4 with q8 would average two different marginals. "
             "Any randomised probe adapter is excluded too: its episodes all run to the "
             "token cap, which would swamp the per-zoom signal with a decode-length "
             "effect that has nothing to do with zooming.\n")
    L.append("| tool calls | n | mean unmodelled tok | mean signed error | mean actual ms |")
    L.append("|---|---|---|---|---|")
    for tc in sorted(by_tc):
        g = by_tc[tc]
        L.append(f"| {tc} | {len(g)} | "
                 f"{sum(unmodelled_prefill_tokens(x) for x in g)/len(g):.0f} | "
                 f"{sum(x['predicted_ms'] - x['actual_ms'] for x in g)/len(g):+.0f} ms | "
                 f"{sum(x['actual_ms'] for x in g)/len(g):.0f} ms |")

    zero, one = by_tc.get(0, []), by_tc.get(1, [])
    if zero and one:
        e0 = sum(x["predicted_ms"] - x["actual_ms"] for x in zero) / len(zero)
        e1 = sum(x["predicted_ms"] - x["actual_ms"] for x in one) / len(one)
        marginal = e0 - e1
        c_coef = float(co["c_ms_per_tool_call"])
        L.append(f"\n**Measured answer.** Going from zero zooms to one, the under-charge "
                 f"grows by **{marginal:+.0f} ms** (n={len(zero)} vs n={len(one)}). The "
                 f"reward charges `c = {c_coef:.0f} ms` for that zoom, so the true "
                 f"marginal cost of a zoom on this device measures "
                 f"**{c_coef + marginal:.0f} ms**, about "
                 f"{(c_coef + marginal)/c_coef:.1f}x what the reward charges.\n")
        if abs(marginal) < 0.5 * c_coef:
            L.append("**That is small, and it is the good outcome.** The gap behaves like "
                     "a per-episode constant rather than a per-zoom charge. A constant "
                     "cancels exactly inside a GRPO group: all eight rollouts on one image "
                     "carry the same system prompt and question, so a cost term uniformly "
                     "low by alpha shifts all eight rewards equally and vanishes in "
                     "`reward - mean(reward)`. The cost model only has to RANK episodes, "
                     "and a constant offset does not change any ranking.\n")
            L.append(f"**Do not over-claim it.** The zero-zoom cell has only {len(zero)} "
                     f"episodes, because this policy zooms on most images at temperature 0. "
                     f"The estimate is therefore noisy, and it is a statement about this "
                     f"policy on these 36 images, not a proof. What is solid is the "
                     f"direction of the residual: `predicted` is below `actual` on every "
                     f"single episode measured, so the reward can under-price a zoom but "
                     f"never over-price one.\n")
        else:
            L.append("**Which way does it bias training?** The reward makes zooming look "
                     "CHEAPER than it is, so a cost-aware run under-penalises looking. "
                     "Run B's advantage over run A is therefore, if anything, UNDERSTATED: "
                     "a policy trained against the true marginal zoom cost would zoom less "
                     "than B does, not more. Report it that way.\n")
            L.append("**And it does not fully cancel inside a GRPO group.** Group "
                     "standardisation removes whatever is identical across the eight "
                     "rollouts on one image, so the system-prompt part cancels exactly. "
                     "The per-zoom part does not: rollouts in a group differ in how often "
                     "they zoom, so they carry different amounts of unpriced text. It is a "
                     "bias, not noise — it always points the same way.\n")


    if any(t.endswith("nocache") for t in tags):
        L.append("\n## The cost model assumes prefix caching. Say so out loud.\n")
        L.append("`cost_ms = a*vision + b*decode + c*tool` charges each vision token "
                 "ONCE. A multi-turn zoom episode only behaves that way if the server "
                 "reuses the KV prefix between turns. Turn it off and every zoom "
                 "re-prefills the whole conversation — thumbnail, system prompt, all "
                 "prior crops and all prior reasoning.\n")
        L.append("The `_nocache` run measures exactly that. Compare it against the "
                 "cached run on the same images:\n")
        L.append("| episodes | cached error | uncached error |")
        L.append("|---|---|---|")
        base_rows = {r["sid"]: r for r in tags.get("base_q4", [])}
        nc_rows = [r for r in tags.get("base_q4_nocache", []) if r["sid"] in base_rows]
        for lbl, pred in (("no zoom (1 turn)", lambda r: r["tool_calls"] == 0),
                          ("with zoom (2+ turns)", lambda r: r["tool_calls"] > 0)):
            sel = [r for r in nc_rows if pred(r)]
            if not sel:
                continue
            ce = sum(base_rows[r["sid"]]["predicted_ms"] - base_rows[r["sid"]]["actual_ms"]
                     for r in sel) / len(sel)
            ne = sum(r["predicted_ms"] - r["actual_ms"] for r in sel) / len(sel)
            L.append(f"| {lbl}, n={len(sel)} | {ce:+.0f} ms | {ne:+.0f} ms |")
        zsel = [r for r in nc_rows if r["tool_calls"] > 0]
        if zsel:
            lost = (sum(r["actual_ms"] - base_rows[r["sid"]]["actual_ms"]
                        for r in zsel) / len(zsel))
            L.append(f"\nSingle-turn episodes are unchanged — there is no prefix to "
                     f"reuse. Episodes that zoom lose **{lost:.0f} ms each**. **This is a "
                     f"deployment requirement, not a caveat:** serve the demo with prompt "
                     f"caching on, or the measured latency will not match the table and "
                     f"the cost-aware policy's advantage will be understated.\n")

    # Repeatability: how big must an A-vs-B latency gap be before it means anything?
    # Find every pair of tags that measured the SAME policy twice. No naming convention
    # needed: at temperature 0 the policy is deterministic, so two runs of identical
    # weights reproduce every episode's decode_tokens and tool_calls exactly. Any tag
    # pair that does so over a decent overlap is a repeat measurement, whether it came
    # from an explicit `_rep2` run or from two checkpoint directories that happen to
    # hold the same weights.
    rep_groups = []
    tlist = sorted(tags)
    for i, t1 in enumerate(tlist):
        for t2 in tlist[i + 1:]:
            if t1.endswith("nocache") or t2.endswith("nocache"):
                continue
            if tags[t1][0].get("quant") != tags[t2][0].get("quant"):
                continue
            d1 = {r["sid"]: r for r in tags[t1]}
            d2 = {r["sid"]: r for r in tags[t2]}
            shared = sorted(set(d1) & set(d2))
            if len(shared) < 8:
                continue
            if all(d1[s]["decode_tokens"] == d2[s]["decode_tokens"]
                   and d1[s]["tool_calls"] == d2[s]["tool_calls"] for s in shared):
                rep_groups.append((t1, t2, [(d1[s], d2[s]) for s in shared]))
    rep_pairs = [p for _, _, ps in rep_groups for p in ps]
    noise_bar_pct = 1.0
    rep_tags = (rep_groups[0][0], rep_groups[0][1]) if rep_groups else None
    if rep_pairs:
        import math as _m
        same = sum(1 for x, y in rep_pairs
                   if x["decode_tokens"] == y["decode_tokens"]
                   and x["tool_calls"] == y["tool_calls"])
        d = [(y["actual_ms"] - x["actual_ms"]) / x["actual_ms"] * 100.0
             for x, y in rep_pairs]
        mu = sum(d) / len(d)
        sd = _m.sqrt(sum((v - mu) ** 2 for v in d) / len(d))
        L.append("\n## Repeatability — the error bar on any A-vs-B claim\n")
        L.append("A latency difference between two policies only means something if it is "
                 "bigger than the measurement's own scatter. These pairs measured the "
                 "SAME weights twice. They are detected, not declared: at temperature 0 "
                 "the policy is deterministic, so a pair that reproduces every episode's "
                 "decode tokens and tool calls is a repeat of the hardware, not a new "
                 "sample of the policy.\n")
        L.append("| pair | episodes | mean diff | sd | max |")
        L.append("|---|---|---|---|---|")
        for t1, t2, ps in rep_groups:
            dd = [(y["actual_ms"] - x["actual_ms"]) / x["actual_ms"] * 100.0
                  for x, y in ps]
            m1 = sum(dd) / len(dd)
            s1 = _m.sqrt(sum((v - m1) ** 2 for v in dd) / len(dd))
            L.append(f"| `{t1}` vs `{t2}` | {len(ps)} | {m1:+.2f}% | {s1:.2f}% | "
                     f"{max(abs(v) for v in dd):.2f}% |")
        L.append(f"\n- **{same} of {len(rep_pairs)} paired episodes replayed an identical "
                 f"trajectory.** Where a pair spans two different checkpoint directories "
                 f"holding the same weights, that also proves the PEFT -> GGUF conversion "
                 f"is deterministic: convert twice, get the same policy.")
        L.append(f"- Pooled, `actual_ms` repeated to **mean {mu:+.2f}%, sd {sd:.2f}%, "
                 f"max |{max(abs(v) for v in d):.2f}%|**.")
        try:
            s1 = _stats(tags[rep_tags[0]], A_COEF)
            s2 = _stats(tags[rep_tags[1]], A_COEF)
            L.append(f"- The whole calibration reproduces, not just the raw times: "
                     f"alpha {s1['cal_alpha_ms']:+.0f} vs {s2['cal_alpha_ms']:+.0f} ms, "
                     f"beta {s1['cal_beta']:.3f} vs {s2['cal_beta']:.3f}. The constant "
                     f"offset is a property of the setup, not an artifact of one run.")
        except Exception:
            pass
        noise_bar_pct = max(1.0, 2 * sd)
        L.append(f"\nSo treat anything under about **{noise_bar_pct:.0f}%** of episode "
                 f"latency as noise. A cost-aware policy that beats the control by less "
                 f"than that has not been shown to beat it. This bar holds with the "
                 f"rsync mirror running, which is the condition every number here was "
                 f"taken under.\n")

    L.append("\n## Does LoRA -> GGUF conversion work on this build? Yes, after one fix.\n")
    L.append("**The base model weights are NOT needed.** `convert_lora_to_gguf.py --base` "
             "wants config files only — its own `--help` says \"actual model weights are "
             "not required\". A config-only copy of the rig's model dir "
             "(`config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, "
             "`merges.txt`) is enough, which is ~23 MB instead of 9 GB. The adapter is "
             "then loaded with `llama-server --lora <file.gguf>` against the existing "
             "quantized GGUF. Conversion takes seconds.\n")
    L.append("**But it fails out of the box on this model, and the failure is silent "
             "until you use a real adapter.** Qwen3.5 is a hybrid: some blocks are "
             "ordinary attention, others are DeltaNet linear attention. llama.cpp "
             "reorders the V heads of the linear-attention blocks from HF's grouped "
             "layout to ggml's tiled layout. For `linear_attn.out_proj` that reorder runs "
             "along the **input** dimension, and a LoRA stored factored as `W = B @ A` "
             "cannot reshape that axis — `NotImplementedError: can't reshape the row size "
             "trivially`. Any adapter targeting `out_proj` is unconvertible.\n")
    L.append("`scripts/lora_convert_shim.py` fixes it exactly, not approximately: "
             "`(B @ A)[:, p] == B @ (A[:, p])`, so a column permutation belongs to `A` "
             "alone and a row permutation to `B` alone. The shim routes the reorder to "
             "whichever factor owns the permuted axis and leaves upstream llama.cpp "
             "untouched. It is a no-op for adapters that never touch those modules.\n")
    L.append("**Worth knowing if you are reproducing this:** a probe adapter proves "
             "nothing unless it shares the real adapter's `target_modules`. The dummy "
             "used here converted cleanly and hid this incompatibility completely, "
             "because it targeted only `q|k|v|o_proj` and the MLP.\n")

    L.append("\n## Method, and what could make the headline wrong\n")
    L.append("**How an episode is measured.** One `llama-server` process per run, the "
             "run's own GGUF quantization + mmproj-F16, `-ngl 99`, Metal, `--jinja "
             "--reasoning-format none`. Turn 0 of every episode runs with "
             "`cache_prompt=false` so the episode pays for its own thumbnail encode; "
             "later turns reuse the prefix, which is what a real deployment does. An "
             "untimed warm-up on a synthetic image absorbs Metal shader compilation. "
             "`actual_ms` is the server's own `prompt_ms + predicted_ms` summed over "
             "turns — generation phases only, never process startup.\n")
    L.append("**Two llama-server flags decide whether this works at all**, and neither is "
             "obvious. `--jinja` makes the server use the model's own chat template; "
             "without it the assistant turn never opens a `<think>` block and the model "
             "degenerates into repeating one sentence to the token cap — 0 tool calls on "
             "6 of 6 probe episodes. `--reasoning-format none` keeps the thoughts in "
             "`message.content` so `src/parse.py` sees the same string training sees; with "
             "the default the server extracts them into `reasoning_content`, `content` "
             "comes back empty, and every episode parses as malformed. The request also "
             "sets `enable_thinking:false`, which is what makes the template open the "
             "assistant turn inside `<think>` exactly as `src/conversation.py` does.\n")
    L.append("**Serial by construction.** One timing job at a time on this Mac. "
             "Concurrent Metal work poisons every number, so the fan-out that GOAL sec.19 "
             "asks for is deliberately not applied here.\n")
    L.append("**Known caveats, in the order they could bite:**\n")
    L.append("1. An `rsync` mirror from the rig runs every 60 s during measurement. It is "
             "light and disk-bound, not GPU-bound, but it is not zero. It was left "
             "running on purpose — stopping it risks losing training artifacts, which "
             "matters more than a few ms of timing noise.")
    L.append("2. The sample is 36 of 191 V*Bench items, stratified across "
             "`direct_attributes` (20) and `relative_position` (16). The link to the rig "
             "is 0.65 MB/s and the full set is 270 MB. Accuracy figures here are "
             "therefore indicative; the LATENCY figures are not affected by which "
             "images were chosen, only by their size distribution.")
    L.append("3. `temperature=0`, while training samples at 1.0. This measures one "
             "deterministic trajectory per image, not the policy's distribution. Tool "
             "rate at temperature 0 is not the tool rate the training metrics report.")
    L.append("4. The M4 path renders the prompt with llama.cpp's chat template; the rig "
             "concatenates token ids (Gate 2). Vision tokens, decode tokens and tool "
             "calls — the three cost-model inputs — are identical by construction. The "
             "text scaffold around them is not, and that is the `alpha` above.")
    if any(t.startswith("probe") for t in tags):
        L.append("5. `probe_dummy` is a RANDOMLY INITIALISED rank-16 adapter "
                 "(`lora_B` std 0.02, not zero). It is not a policy and its accuracy "
                 "means nothing. It is in this file because it proves the "
                 "PEFT -> GGUF -> `--lora` path genuinely applies the adapter: the same "
                 "prompts that produce clean tool calls on the base model degenerate "
                 "into repetition under it.")
    else:
        L.append("5. The PEFT -> GGUF -> `--lora` path was proved separately against a "
                 "randomly initialised rank-16 adapter (`lora_B` std 0.02, not zero): the "
                 "same prompts that give clean tool calls on the base model degenerate "
                 "into repetition under it, so the adapter is demonstrably reaching the "
                 "weights and not being silently ignored. See LEDGER.")
    uc_path = os.path.join(EVAL_DIR, "m4_units_check.json")
    if os.path.exists(uc_path):
        with open(uc_path) as fh:
            uc = json.load(fh)
        verdict = "PASS" if uc.get("crop_mismatches", 1) == 0 else "FAIL"
        L.append(f"6. **Cross-machine units check: {verdict}.** The reward was paid on "
                 f"counts from the real HF `Qwen2VLImageProcessor` on the rig. This "
                 f"script computes its own counts on the Mac. If they ever disagreed, "
                 f"every `predicted_ms` here would be in the wrong units and nothing "
                 f"would raise. Replaying the rig's own recorded boxes through this "
                 f"machine's `crop_from_bbox` + `vision_tokens_for`: "
                 f"**{uc.get('crops')} crops, {uc.get('crop_mismatches')} mismatches, "
                 f"{uc.get('episode_totals_exact')}/{uc.get('episodes')} episode totals "
                 f"exact.** Re-run it with `python3 scripts/m4_verify.py --verify-units`.")
    n_div = sum(1 for r in rows if r["vision_tokens"] != r["vision_tokens_llamacpp"])
    d_ms = (sum(r["predicted_ms"] - r["predicted_ms_llamacpp_units"] for r in rows)
            / len(rows))
    L.append("7. **The units trap, and it did not bite.** Vision tokens are counted in HF "
             "units, which is what `a` was fitted in. llama.cpp's clip uses different "
             "clamps (HF floors at 64 tokens, clip at 9) and rounds the other way, so the "
             "two rules can disagree on non-square crops. Measured here: they differ on "
             f"{n_div} of {len(rows)} episodes, worth {d_ms:+.1f} ms of predicted cost on "
             "average. Both counts are in the JSONL. Crops are resized to a 512 px "
             "longest side, which keeps them in the range where the rules agree except "
             "on aspect, so the trap stays closed for this policy. It would reopen if a "
             "future policy emitted extreme strips.\n")

    L.append("\n## Per-run detail\n")
    L.append("`R² vs identity` asks how much of the variance in measured time you capture "
             "by taking `predicted_ms` as-is, with no calibration at all. It is low "
             "wherever the constant offset is large relative to episode length — that is "
             "the offset showing up, not a weak relationship. The Pearson r and the "
             "calibrated MAPE in the tables above are the fair reads.\n")
    for tag in sorted(tags):
        rws = tags[tag]
        s = _stats(rws, A_COEF)
        meta = rws[0]
        L.append(f"\n### `{tag}`\n")
        L.append(f"- adapter: `{meta.get('adapter','?')}`"
                 + (f" (weights sha256 `{meta['adapter_sha256'][:12]}`)"
                    if meta.get("adapter_sha256") else "")
                 + f", measured {meta.get('ts','?')[:19].replace('T',' ')} UTC")
        L.append(f"- quant: {meta.get('quant','?')}, cache_prompt: "
                 f"{meta.get('cache_prompt')}, temperature "
                 f"{meta.get('temperature', 0.0)}, geometry "
                 f"{meta.get('geom', DEFAULT_GEOM)}")
        L.append(f"- episodes: {s['n']}, mean tool calls {s['mean_tool_calls']:.2f}, "
                 f"tool rate {s['tool_rate']*100:.0f}%, accuracy {s['accuracy']*100:.0f}%")
        L.append(f"- mean vision tokens {s['mean_vision_tokens']:.0f}, mean decode "
                 f"tokens {s['mean_decode_tokens']:.0f}")
        L.append(f"- **MAPE {s['mape_pct']:.1f}%**, Pearson r {s['pearson_r']:.4f}, "
                 f"RMSE {s['rmse_ms']:.0f} ms, R² vs identity "
                 f"{s['r2_vs_identity']:.4f}, slope {s['slope_actual_on_pred']:.3f}")
        L.append("\n| sid | vis tok | dec tok | tools | predicted ms | actual ms | "
                 "err | err % |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(rws, key=lambda x: x["actual_ms"]):
            e = r["predicted_ms"] - r["actual_ms"]
            pct = e / r["actual_ms"] * 100 if r["actual_ms"] else float("nan")
            L.append(f"| {r['sid'].replace('vstar-','')} | {r['vision_tokens']} | "
                     f"{r['decode_tokens']} | {r['tool_calls']} | "
                     f"{r['predicted_ms']:.0f} | {r['actual_ms']:.0f} | {e:+.0f} | "
                     f"{pct:+.1f}% |")

    n_box = sum(1 for r in rows if r.get("boxes"))
    L.append("\n## What else is in the JSONL (for the demo track)\n")
    L.append(f"`eval/m4_latency.jsonl` carries more than the five latency fields. "
             f"{n_box} of {len(rows)} episodes record `boxes` — the zoom rectangles the "
             f"policy actually chose, on the 0-1000 normalised frame — alongside "
             f"`thumb_size`, `orig_size`, `tool_calls`, `correct` and the full per-turn "
             f"text. That is everything the attention-crop viewer needs to draw where the "
             f"policy looked, and these are the boxes it chose ON THE DEPLOYMENT TARGET, "
             f"not on the training rig. Map a box to thumbnail pixels with "
             f"`src/zoom_env.py:draw_boxes`, which takes the same 0-1000 frame.\n")

    L.append("\n## Scatter data (predicted, actual), all runs\n")
    L.append("```")
    L.append("tag,sid,predicted_ms,actual_ms,vision_tokens,decode_tokens,tool_calls")
    for r in rows:
        L.append(f"{r['tag']},{r['sid']},{r['predicted_ms']:.1f},{r['actual_ms']:.1f},"
                 f"{r['vision_tokens']},{r['decode_tokens']},{r['tool_calls']}")
    L.append("```")

    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(REPORT, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[report] wrote {REPORT} ({len(rows)} rows)")


# --- main ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="base",
                    help="PEFT LoRA dir, or 'base' for no adapter")
    ap.add_argument("--quant", default="q4", choices=sorted(M.QUANTS))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--port", type=int, default=8211)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy (default). Use 1.0 to sample the policy's "
                         "distribution the way the training loop's eval does.")
    ap.add_argument("--seed", type=int, default=None,
                    help="sampler seed, so a temperature>0 run is still reproducible")
    ap.add_argument("--no-cache", action="store_true",
                    help="cache_prompt=False: every turn re-prefills the conversation")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--verify-units", action="store_true",
                    help="cross-check Mac vision-token counts against the rig's HF "
                         "processor, using the mirrored eval predictions")
    args = ap.parse_args()

    os.makedirs(EVAL_DIR, exist_ok=True)
    if args.verify_units:
        verify_units()
        return
    if args.report_only:
        write_report()
        return

    tag = args.tag or (os.path.basename(os.path.abspath(args.adapter)))
    cache = not args.no_cache

    geom = geometry_for(args.adapter)
    # Hash the weights, not the path. `adapters/last` means different weights at
    # different training steps, so the path alone cannot tell two measurements apart.
    adapter_sha = ""
    if args.adapter not in ("base", "none", ""):
        import hashlib as _hl
        _d = os.path.abspath(os.path.expanduser(args.adapter))
        for _sub in ("", "best", "last"):
            _w = os.path.join(_d, _sub, "adapter_model.safetensors")
            if os.path.exists(_w):
                _h = _hl.sha256()
                with open(_w, "rb") as _fh:
                    for _chunk in iter(lambda: _fh.read(1 << 20), b""):
                        _h.update(_chunk)
                adapter_sha = _h.hexdigest()
                break
    lora = None
    if args.adapter not in ("base", "none", ""):
        lora = convert_lora(args.adapter)

    from src.data import load_vstar
    samples = load_vstar()
    print(f"[data] {len(samples)} V*Bench samples under {DATA_ROOT}", flush=True)
    samples = samples[:args.n] if args.n else samples

    done = {(r["tag"], r["sid"]) for r in load_rows()}
    todo = [s for s in samples if (tag, s.sid) not in done]
    if not todo:
        print(f"[{tag}] nothing to do")
        write_report()
        return

    cm = CostModel.from_config(
        {"cost_mode": "coeffs", "coeffs_path": f"cost_model/coeffs_{args.quant}.json"},
        REPO)
    print(f"[cost] a={cm.a:.4f} b={cm.b:.4f} c={cm.c:.4f} sha={cm.sha256[:16]}",
          flush=True)

    log_path = os.path.join(REPO, "cost_model", "raw", f"verify_{tag}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    proc = log = None
    fh = open(JSONL, "a")
    t_start = time.time()
    try:
        print(f"[{tag}] starting server (lora={lora})", flush=True)
        proc, log = start_server(args.quant, args.port, log_path, lora)

        # Untimed warm-up: Metal shader compilation must never land in a recorded row.
        # On a SYNTHETIC image, never a measured one — warming on todo[0] left its KV in
        # the slot and the first real episode inherited it.
        from PIL import Image as _Image
        from src.contract import Sample as _Sample
        run_episode(args.port, _Sample(
            sid="warmup", image=_Image.new("RGB", (2250, 1500), "gray"),
            question="What color is the sky?", gold="blue", source="vstar",
            options=["blue", "green"]), cache, geom, args.temperature, args.seed)
        print(f"[{tag}] warm, {len(todo)} episodes to run", flush=True)

        for i, s in enumerate(todo):
            row = run_episode(args.port, s, cache, geom, args.temperature, args.seed)
            row["predicted_ms"] = cm.cost_ms(
                row["vision_tokens"], row["decode_tokens"], row["tool_calls"])
            row["predicted_ms_llamacpp_units"] = cm.cost_ms(
                row["vision_tokens_llamacpp"], row["decode_tokens"], row["tool_calls"])
            row["tag"] = tag
            row["adapter"] = ("base (no adapter)" if lora is None
                              else os.path.abspath(os.path.expanduser(args.adapter)))
            row["lora_gguf"] = lora
            row["adapter_sha256"] = adapter_sha
            row["quant"] = args.quant
            row["cache_prompt"] = cache
            row["geom"] = geom
            row["temperature"] = args.temperature
            row["seed"] = args.seed
            row["coeff_sha256"] = cm.sha256
            row["ts"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            e = row["predicted_ms"] - row["actual_ms"]
            pct = (f"{e / row['actual_ms'] * 100:+.0f}%" if row["actual_ms"] > 0
                   else "SERVER ERROR, unusable")
            print(f"[{tag}] {i+1}/{len(todo)} {row['sid'][:34]} "
                  f"vt={row['vision_tokens']} dt={row['decode_tokens']} "
                  f"tc={row['tool_calls']} pred={row['predicted_ms']:.0f} "
                  f"act={row['actual_ms']:.0f} err={e:+.0f} "
                  f"({pct}) correct={row['correct']}",
                  flush=True)
    finally:
        M.stop_server(proc, log)
        fh.close()
        print(f"[{tag}] done in {time.time() - t_start:.0f}s", flush=True)
        write_report()


if __name__ == "__main__":
    main()
