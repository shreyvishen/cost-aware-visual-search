"""Live inference backend for the demo web app.

One public entry point, `run_episode(image_path, question, model, quant, downproject)`.
It runs a real zoom episode against a warm `llama-server` on this Mac and returns the
tokens, the measured milliseconds, the dollars and the boxes the model looked at.

WHAT IS REUSED, AND WHY IT MATTERS
----------------------------------
Every number the demo page prints has to be the same *kind* of number the offline
measurement produced, or the page is lying. So this module does not re-implement the
episode; it imports it.

* `scripts/m4_verify.py` — the server harness (`start_server`, with the load-bearing
  `--jinja` / `--reasoning-format none` flags), the timed turn (`chat`, with
  `enable_thinking:false` and `cache_prompt`), the image encoder (`b64_pil`), and the
  three literal tool/force strings the episode replies with.
* `src/zoom_env.py` — the thumbnail rule, `crop_from_bbox`, the 0-1000 coordinate
  frame, `SYSTEM_PROMPT` and `USER_TEMPLATE`.
* `src/parse.py` — the same think/tool_call/answer contract.
* `cost_model/vision_tokens.py` — `vision_tokens_for`, the reward's (HF) units.

The zoom loop below is a line-for-line mirror of `m4_verify.run_episode`, kept separate
only because the page needs two things that function does not return: the full `<think>`
text (it truncates to 600 chars) and the crop images themselves. `python3 app/infer.py
--selfcheck` runs the same image through BOTH loops at temperature 0 and fails loudly if
vision tokens, decode tokens, tool calls, boxes or prefill tokens differ. Run it after
touching either file.

BASE64 FIELDS ARE RAW, NOT DATA URIS
------------------------------------
`thumb_png_b64` and `crop_png_b64` are bare base64 of a PNG. The page must prepend
`data:image/png;base64,` itself.

CONCURRENCY
-----------
At most one `llama-server` process is alive, keyed by (model, quant). A different key
stops the old one and starts a new one — 10-40 s, so call `warmup()` first (or
`python3 app/warm.py --model b --quant q4`). One module-level lock serialises every
generation: two concurrent Metal jobs corrupt the timings this page exists to report.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import importlib.util
import io
import json
import os
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from PIL import Image, ImageOps  # noqa: E402


def _load_m4():
    """Import scripts/m4_verify.py by path — `scripts/` is not a package."""
    path = os.path.join(REPO, "scripts", "m4_verify.py")
    spec = importlib.util.spec_from_file_location("m4_verify", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m4_verify"] = mod
    spec.loader.exec_module(mod)
    return mod


M4 = _load_m4()
M = M4.M                       # cost_model.measure — QUANTS, wait_healthy, stop_server

from cost_model.vision_tokens import vision_tokens_for  # noqa: E402
from src import parse  # noqa: E402
from src.zoom_env import SYSTEM_PROMPT, USER_TEMPLATE, crop_from_bbox, make_thumbnail  # noqa: E402

# --- configuration -------------------------------------------------------------------

#: Not 8211. That is m4_verify's default and the benchmark queue owns it.
PORT = int(os.environ.get("DEMO_INFER_PORT", "8231"))

LORA_DIR = os.path.expanduser("~/assets/models/qwen3.5-4b-gguf/lora")
#: model id -> LoRA GGUF, or None for the untuned base.
MODELS = {
    "base": None,
    "a": os.path.join(LORA_DIR, "run_a.gguf"),
    "b": os.path.join(LORA_DIR, "run_b.gguf"),
}
QUANTS = ("q4", "q8", "bf16")

#: Hard cap on ONE generation, measured from the moment the server is healthy. Loading
#: a model is not inside this budget (bf16 alone takes ~40 s) — that is what warmup()
#: is for. On expiry the server is killed, which unblocks the in-flight HTTP call, and
#: the partial numbers collected so far are returned with `error` set.
REQUEST_TIMEOUT_S = float(os.environ.get("DEMO_REQUEST_TIMEOUT_S", "120"))
#: Separate budget for model load.
LOAD_TIMEOUT_S = float(os.environ.get("DEMO_LOAD_TIMEOUT_S", "300"))

MAX_ZOOMS = M4.DEFAULT_GEOM["max_zooms"]            # 3
MAX_NEW_TOKENS = M4.DEFAULT_GEOM["max_new_tokens"]  # 200

#: Blended API price the page quotes, per token.
USD_PREFILL = 0.03 / 1e6
USD_DECODE = 0.15 / 1e6

#: The page shows the image the model was handed. A full-resolution V*Bench frame is a
#: multi-megabyte PNG once base64'd, so the PREVIEW (never the model's input) is capped.
PREVIEW_MAX_SIDE = 1024

#: downproject=False is the expensive baseline: full-resolution image, one turn, no tool.
#: The zoom tool is removed from the system prompt rather than merely discouraged — left
#: in, the model calls it anyway and the "no tool" baseline stops being one. The last two
#: sentences are copied verbatim from USER_TEMPLATE so answer length stays comparable.
SYSTEM_PROMPT_DIRECT = "You are a helpful assistant."
USER_TEMPLATE_DIRECT = (
    "{question}\n"
    "Answer the question from the image above. "
    "Format strictly as:  <think>...</think>  <answer>...</answer> "
    "Make the answer as short as possible: yes, no, or a single word or phrase."
)

LOG_DIR = os.path.join(REPO, "app", "logs")

# --- server lifecycle ----------------------------------------------------------------

#: Serialises EVERYTHING: server swaps and generation alike. Reentrant because the
#: timeout path stops the server while already holding it.
_LOCK = threading.RLock()
#: {"key": (model, quant), "proc": Popen, "log": file, "port": int} or None.
_SERVER: dict | None = None


def _log(msg: str) -> None:
    print(f"[infer] {msg}", file=sys.stderr, flush=True)


def bench_queue_busy() -> str | None:
    """Is some OTHER llama-server running? Returns a description, or None.

    Two servers on one Metal GPU corrupt both sets of timings. The demo warns rather
    than blocks — a live audience beats a clean measurement — but `app/warm.py` waits.
    """
    ours = _SERVER["proc"].pid if _SERVER and _SERVER["proc"].poll() is None else -1
    try:
        out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        return None
    for line in out.splitlines():
        line = line.strip()
        if "llama-server" not in line or " grep " in line:
            continue
        pid = line.split(None, 1)[0]
        if not pid.isdigit() or int(pid) == ours:
            continue
        return f"pid {pid}: {line[:160]}"
    return None


def _server_alive() -> bool:
    return _SERVER is not None and _SERVER["proc"].poll() is None


def _stop_locked() -> None:
    global _SERVER
    if _SERVER is None:
        return
    _log(f"stopping server {_SERVER['key']} (pid {_SERVER['proc'].pid})")
    try:
        M.stop_server(_SERVER["proc"], _SERVER["log"])
    except Exception as exc:                                  # never raise on teardown
        _log(f"stop_server complained: {exc}")
    _SERVER = None


def _ensure_server(model: str, quant: str) -> int:
    """Return a healthy server's port, starting or swapping the process if needed."""
    global _SERVER
    key = (model, quant)
    if _server_alive() and _SERVER["key"] == key:
        return _SERVER["port"]
    if _SERVER is not None:
        _stop_locked()

    other = bench_queue_busy()
    if other:
        _log(f"WARNING another llama-server is running, timings may be contended -> {other}")

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"llama_{model}_{quant}.log")
    lora = MODELS[model]
    t0 = time.perf_counter()
    _log(f"starting {model}/{quant} on :{PORT} (lora={lora})")
    proc, log = M4.start_server(quant, PORT, log_path, lora)
    _SERVER = {"key": key, "proc": proc, "log": log, "port": PORT}
    _log(f"ready in {time.perf_counter() - t0:.1f}s")
    return PORT


def warmup(model: str = "base", quant: str = "q4") -> dict:
    """Load a model and run one throwaway episode so the first real request is fast.

    The warm-up runs on a SYNTHETIC image, never a real one: warming on a real sample
    leaves its KV in the server's single slot and the next episode inherits it, which
    is exactly how m4_verify once measured a 4-token prefill.
    """
    _validate(model, quant)
    with _LOCK:
        t0 = time.perf_counter()
        port = _ensure_server(model, quant)
        state = _new_state(model, quant, True)
        img = Image.new("RGB", (2250, 1500), "gray")
        try:
            _episode_zoom(port, img, "What color is the sky?", state,
                          time.perf_counter() + REQUEST_TIMEOUT_S)
        except Exception as exc:
            _log(f"warmup episode failed: {type(exc).__name__}: {exc}")
        secs = time.perf_counter() - t0
        _log(f"warm {model}/{quant} in {secs:.1f}s "
             f"(prefill {state['prefill_tokens']} tok, decode {state['decode_tokens']} tok)")
        return {"model": model, "quant": quant, "seconds": round(secs, 2),
                "port": port, "error": state["error"]}


def shutdown() -> None:
    """Stop the server, if any. Safe to call repeatedly."""
    with _LOCK:
        _stop_locked()


def server_status() -> dict:
    """What is loaded right now — handy for the page's status pill."""
    with _LOCK:
        if not _server_alive():
            return {"loaded": False, "model": None, "quant": None, "port": PORT}
        model, quant = _SERVER["key"]
        return {"loaded": True, "model": model, "quant": quant, "port": _SERVER["port"],
                "pid": _SERVER["proc"].pid}


atexit.register(shutdown)

# --- helpers -------------------------------------------------------------------------


def _validate(model: str, quant: str) -> None:
    if model not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}, got {model!r}")
    if quant not in QUANTS:
        raise ValueError(f"quant must be one of {list(QUANTS)}, got {quant!r}")


def _png_b64(img: Image.Image, max_side: int | None = None) -> str:
    """Bare base64 of a PNG. No `data:` prefix — the page adds it."""
    out = img.convert("RGB")
    if max_side and max(out.size) > max_side:
        r = max_side / float(max(out.size))
        out = out.resize((max(1, int(out.width * r)), max(1, int(out.height * r))),
                         Image.BICUBIC)
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _think_of(text: str) -> str:
    """The turn's reasoning. Falls back to whatever preceded the first tag."""
    got = parse.extract_think(text)
    if got:
        return got
    head = text.split("<tool_call>")[0].split("<answer>")[0]
    return head.replace("</think>", "").replace("<|im_end|>", "").strip()


def _new_state(model: str, quant: str, downproject: bool) -> dict:
    return {
        "answer": "", "turns": [], "boxes": [],
        "prefill_tokens": 0, "decode_tokens": 0, "vision_tokens": 0,
        "prefill_ms": 0.0, "decode_ms": 0.0, "total_ms": 0.0,
        "zooms": 0, "usd": 0.0, "thumb_png_b64": "",
        "model": model, "quant": quant, "downproject": downproject,
        "error": None,
    }


def _account(state: dict, timings: dict) -> None:
    """Fold one turn's server-reported timings into the running totals.

    llama-server's own `timings`, never wall clock — that is the number the offline
    measurement used, so it is the number the page must show.
    """
    state["prefill_tokens"] += int(timings["prompt_n"])
    state["decode_tokens"] += int(timings["predicted_n"])
    state["prefill_ms"] += float(timings["prompt_ms"])
    state["decode_ms"] += float(timings["predicted_ms"])
    state["total_ms"] = state["prefill_ms"] + state["decode_ms"]
    state["usd"] = (state["prefill_tokens"] * USD_PREFILL
                    + state["decode_tokens"] * USD_DECODE)


def _finish(state: dict) -> dict:
    for k in ("prefill_ms", "decode_ms", "total_ms"):
        state[k] = round(state[k], 2)
    state["usd"] = round(state["usd"], 8)
    return state


# --- the two episodes ----------------------------------------------------------------


def _episode_zoom(port: int, image: Image.Image, question: str, state: dict,
                  deadline: float) -> None:
    """downproject=True. A mirror of m4_verify.run_episode — see --selfcheck.

    The budget is checked BEFORE it is spent, same as src/rollout.py, otherwise a
    thumbnail-only baseline gets one free look.
    """
    geom = M4.DEFAULT_GEOM
    thumb, _scale = make_thumbnail(image, geom["downsample"], geom["thumb_max_side"])
    state["thumb_png_b64"] = _png_b64(thumb)
    state["vision_tokens"] = vision_tokens_for(*thumb.size)

    user_text = USER_TEMPLATE.format(image_token="", question=question).lstrip("\n")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": M4.b64_pil(thumb)}},
            {"type": "text", "text": user_text},
        ]},
    ]

    answer = None
    forced = False
    for turn_i in range(geom["max_zooms"] + 2):
        if time.perf_counter() > deadline:
            state["error"] = state["error"] or "out of time before the next turn"
            break
        # Turn 0 never reuses the cache, so every episode pays for its own image
        # encode; later turns do, the way a real deployment would.
        text, t, _wall = M4.chat(port, messages, geom["max_new_tokens"], turn_i > 0, 0.0)
        if int(t["prompt_n"]) == 0 and not text:
            state["error"] = "llama-server rejected its own output on a turn"
            break
        _account(state, t)
        kind = parse.classify(text)
        turn = {"think": _think_of(text), "bbox_2d": None, "crop_png_b64": None}
        state["turns"].append(turn)
        messages.append({"role": "assistant", "content": text})

        if kind == "answer":
            answer, _src = parse.extract_answer_source(text)
            break

        if kind == "tool_call" and not forced:
            if state["zooms"] >= geom["max_zooms"]:
                messages.append({"role": "user", "content": M4.FORCE_ANSWER_TEXT})
                forced = True
                continue
            box = parse.extract_bbox(text)
            state["zooms"] += 1
            turn["bbox_2d"] = list(box) if box else None
            if box is None:
                messages.append({"role": "user", "content": M4.TOOL_ERROR_TEXT})
            else:
                crop, _info = crop_from_bbox(image, box, geom["crop_max_side"])
                state["boxes"].append(list(box))
                if crop is None:
                    messages.append({"role": "user", "content": M4.TOOL_ERROR_TEXT})
                else:
                    turn["crop_png_b64"] = _png_b64(crop)
                    state["vision_tokens"] += vision_tokens_for(*crop.size)
                    messages.append({"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": M4.b64_pil(crop)}},
                        {"type": "text", "text": M4.TOOL_RESPONSE_TEXT},
                    ]})
            if state["zooms"] >= geom["max_zooms"]:
                messages.append({"role": "user", "content": M4.FORCE_ANSWER_TEXT})
                forced = True
            continue

        # malformed, or a tool call after the budget was spent
        if forced:
            answer, _src = parse.extract_answer_source(text)
            break
        messages.append({"role": "user", "content": M4.FORCE_ANSWER_TEXT})
        forced = True

    _set_answer(state, answer)


def _episode_direct(port: int, image: Image.Image, question: str, state: dict,
                    deadline: float) -> None:
    """downproject=False. Full-resolution image, one turn, no tool, answer now.

    This is the expensive baseline the page compares against, so it really does send
    the original pixels: nothing is resized on the way in. llama.cpp's own clip caps
    the vision tower at 4096 tokens, which is the deployment's real ceiling.
    """
    state["thumb_png_b64"] = _png_b64(image, PREVIEW_MAX_SIDE)
    state["vision_tokens"] = vision_tokens_for(*image.size)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_DIRECT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": M4.b64_pil(image)}},
            {"type": "text", "text": USER_TEMPLATE_DIRECT.format(question=question)},
        ]},
    ]
    if time.perf_counter() > deadline:
        state["error"] = "out of time before the only turn"
        return
    text, t, _wall = M4.chat(port, messages, MAX_NEW_TOKENS, False, 0.0)
    if int(t["prompt_n"]) == 0 and not text:
        state["error"] = "llama-server rejected its own output"
        return
    _account(state, t)
    state["turns"].append(
        {"think": _think_of(text), "bbox_2d": None, "crop_png_b64": None})
    answer, _src = parse.extract_answer_source(text)
    _set_answer(state, answer)


def _set_answer(state: dict, answer: str | None) -> None:
    if answer is None:
        state["answer"] = ""
        if state["error"] is None:
            state["error"] = "the model did not produce a parseable answer"
    else:
        state["answer"] = answer.strip()


# --- public entry point --------------------------------------------------------------


def run_episode(image_path: str, question: str, model: str = "base",
                quant: str = "q4", downproject: bool = True) -> dict:
    """Run one episode and return the page's payload. Never raises.

    On any failure the same dict comes back with `error` set and whatever partial
    numbers were collected before it went wrong.
    """
    state = _new_state(model, quant, downproject)
    try:
        _validate(model, quant)
    except ValueError as exc:
        state["error"] = str(exc)
        return _finish(state)

    lora = MODELS[model]
    if lora and not os.path.exists(lora):
        state["error"] = f"missing LoRA GGUF for model {model!r}: {lora}"
        return _finish(state)

    try:
        with Image.open(image_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
    except Exception as exc:
        state["error"] = f"cannot read image {image_path!r}: {type(exc).__name__}: {exc}"
        return _finish(state)

    question = (question or "").strip()
    if not question:
        state["error"] = "empty question"
        return _finish(state)

    with _LOCK:
        try:
            port = _ensure_server(model, quant)
        except Exception as exc:
            _stop_locked()
            state["error"] = f"server would not start: {type(exc).__name__}: {exc}"
            return _finish(state)

        episode = _episode_zoom if downproject else _episode_direct
        deadline = time.perf_counter() + REQUEST_TIMEOUT_S
        done = threading.Event()
        crashed: dict = {}

        def worker():
            try:
                episode(port, image, question, state, deadline)
            except BaseException as exc:                      # noqa: BLE001
                crashed["exc"] = exc
            finally:
                done.set()

        th = threading.Thread(target=worker, name="episode", daemon=True)
        th.start()
        if not done.wait(REQUEST_TIMEOUT_S + 5.0):
            # Killing the server is what unblocks the worker's in-flight HTTP call.
            # The next request pays a reload; a wedged demo page costs more.
            state["error"] = f"timed out after {REQUEST_TIMEOUT_S:.0f}s"
            _log(state["error"] + " — killing the server to unblock")
            _stop_locked()
            done.wait(20)
        elif "exc" in crashed:
            exc = crashed["exc"]
            state["error"] = f"{type(exc).__name__}: {exc}"
            _log(f"episode failed: {state['error']}")

    return _finish(state)


# --- drift gate ----------------------------------------------------------------------


def _selfcheck(image_path: str, question: str, model: str, quant: str) -> int:
    """Prove `_episode_zoom` is still the same episode `m4_verify.run_episode` measures.

    Same image, same question, temperature 0, both loops, one server. If vision tokens,
    decode tokens, tool calls, boxes or prefill tokens differ, the page's numbers have
    drifted away from the numbers in eval/m4_verify_report.md and this exits nonzero.
    """
    from src.contract import Sample

    with Image.open(image_path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")

    with _LOCK:
        port = _ensure_server(model, quant)
        warmup(model, quant)

        mine = _new_state(model, quant, True)
        _episode_zoom(port, image, question, mine, time.perf_counter() + 600)

        theirs = M4.run_episode(port, Sample(
            sid="selfcheck", image=image, question=question, gold="", source="vstar",
            options=None), True, M4.DEFAULT_GEOM, 0.0, None)

    checks = [
        ("vision_tokens", mine["vision_tokens"], theirs["vision_tokens"]),
        ("decode_tokens", mine["decode_tokens"], theirs["decode_tokens"]),
        ("prefill_tokens", mine["prefill_tokens"], theirs["prompt_n_total"]),
        ("zooms", mine["zooms"], theirs["tool_calls"]),
        ("boxes", mine["boxes"], theirs["boxes"]),
        ("n_turns", len(mine["turns"]), theirs["n_turns"]),
        ("answer", mine["answer"], (theirs["answer"] or "").strip()),
    ]
    bad = [c for c in checks if c[1] != c[2]]
    for name, a, b in checks:
        print(f"  {'FAIL' if (name, a, b) in bad else 'ok  '} {name}: infer={a!r} m4_verify={b!r}")
    if bad:
        print(f"SELFCHECK FAILED on {len(bad)} field(s): the demo episode has drifted "
              f"from the measured episode.")
        return 1
    print("SELFCHECK PASS — app/infer.py runs the episode m4_verify measured.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one demo episode from the CLI.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", default="What color is the sign?")
    ap.add_argument("--model", default="base", choices=sorted(MODELS))
    ap.add_argument("--quant", default="q4", choices=list(QUANTS))
    ap.add_argument("--no-downproject", action="store_true",
                    help="full-resolution image, one turn, no zoom tool")
    ap.add_argument("--selfcheck", action="store_true",
                    help="compare this episode against m4_verify.run_episode")
    ap.add_argument("--keep-warm", action="store_true",
                    help="leave the server running after the call")
    args = ap.parse_args()

    try:
        if args.selfcheck:
            return _selfcheck(args.image, args.question, args.model, args.quant)
        out = run_episode(args.image, args.question, args.model, args.quant,
                          not args.no_downproject)
        slim = dict(out)
        slim["thumb_png_b64"] = f"<{len(out['thumb_png_b64'])} b64 chars>"
        slim["turns"] = [
            {**t, "crop_png_b64": (None if t["crop_png_b64"] is None
                                   else f"<{len(t['crop_png_b64'])} b64 chars>")}
            for t in out["turns"]
        ]
        print(json.dumps(slim, indent=2))
        return 0 if out["error"] is None else 1
    finally:
        if not args.keep_warm:
            shutdown()


if __name__ == "__main__":
    sys.exit(main())
