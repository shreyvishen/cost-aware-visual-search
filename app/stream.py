"""Token-streaming version of the demo episode.

`app/infer.py` runs an episode and returns the finished dict. That is right for measurement
and wrong for a demo — the user stares at a spinner for four seconds and learns nothing about
what the model is doing. This yields the episode as it happens: weights loading, prefill
throughput, tokens as they arrive, the tool call, the crop it pulled back, the answer.

It reuses infer's server management, geometry and parsing rather than copying them. If the two
ever disagree, the demo stops matching the measured numbers on the rest of the page.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps

from app import infer
from src import parse
from src.zoom_env import SYSTEM_PROMPT, USER_TEMPLATE, crop_from_bbox, make_thumbnail

M4 = infer.M4
HARD_DEADLINE_S = 180.0


def _chat_stream(port: int, messages: list, max_tokens: int, cache: bool,
                 temperature: float = 0.0):
    """Yield ('delta', text) as tokens arrive, then ('end', full_text, timings).

    `enable_thinking: False` matters as much here as in the measured path: without it
    llama-server moves the reasoning into `reasoning_content`, `content` comes back empty,
    and every episode parses as malformed.
    """
    body = {
        "messages": messages,
        "n_predict": max_tokens,
        "temperature": temperature,
        "cache_prompt": cache,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    text_parts: list[str] = []
    timings = {"prompt_n": 0, "predicted_n": 0, "prompt_ms": 0.0, "predicted_ms": 0.0}
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("timings"):
                    timings = {k: chunk["timings"].get(k, timings[k]) for k in timings}
                for ch in chunk.get("choices", []):
                    piece = (ch.get("delta") or {}).get("content") or ""
                    if piece:
                        text_parts.append(piece)
                        yield ("delta", piece)
    except Exception as exc:  # a bad turn must not take the episode down
        yield ("end", "".join(text_parts), timings, f"{type(exc).__name__}: {exc}")
        return
    yield ("end", "".join(text_parts), timings, None)


def _prefill_event(timings: dict) -> dict:
    n, ms = int(timings.get("prompt_n") or 0), float(timings.get("prompt_ms") or 0.0)
    return {"type": "prefill", "tokens": n, "ms": round(ms, 1),
            "tok_per_s": round(n / (ms / 1000.0), 1) if ms > 0 else None}


def stream_episode(image_path: str, question: str, model: str = "b",
                   quant: str = "q4", downproject: bool = True):
    """Yield the episode as a sequence of JSON-serialisable events."""
    t_start = time.perf_counter()
    deadline = t_start + HARD_DEADLINE_S
    try:
        infer._validate(model, quant)
    except Exception as exc:
        yield {"type": "error", "msg": str(exc)}
        return

    warm = infer.server_status().get("key") == [model, quant] if hasattr(
        infer, "server_status") else False
    yield {"type": "status", "stage": "loading",
           "msg": f"loading {quant.upper()} weights" + ("" if downproject else ", full resolution")}

    t_load = time.perf_counter()
    with infer._LOCK:
        try:
            port = infer._ensure_server(model, quant)
        except Exception as exc:
            yield {"type": "error", "msg": f"could not start the model: {exc}"}
            return
        load_ms = round((time.perf_counter() - t_load) * 1000)
        yield {"type": "status", "stage": "ready", "load_ms": 0 if warm else load_ms}

        try:
            image = ImageOps.exif_transpose(Image.open(image_path).convert("RGB"))
        except Exception as exc:
            yield {"type": "error", "msg": f"could not read the image: {exc}"}
            return

        state = infer._new_state(model, quant, downproject)
        geom = M4.DEFAULT_GEOM

        # --- what the model is shown first -----------------------------------
        if downproject:
            shown, _scale = make_thumbnail(image, geom["downsample"], geom["thumb_max_side"])
        else:
            shown = image
        state["thumb_png_b64"] = infer._png_b64(shown, 640)
        from cost_model.vision_tokens import vision_tokens_for
        state["vision_tokens"] = vision_tokens_for(*shown.size)
        yield {"type": "image", "which": "thumb", "png_b64": state["thumb_png_b64"],
               "w": shown.size[0], "h": shown.size[1],
               "orig_w": image.size[0], "orig_h": image.size[1]}

        user_text = USER_TEMPLATE.format(image_token="", question=question).lstrip("\n")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": M4.b64_pil(shown)}},
                {"type": "text", "text": user_text},
            ]},
        ]

        answer, forced = None, False
        max_turns = (geom["max_zooms"] + 2) if downproject else 1

        for turn_i in range(max_turns):
            if time.perf_counter() > deadline:
                state["error"] = "out of time"
                break

            yield {"type": "turn_start", "index": turn_i}
            text, timings, err = "", None, None
            for ev in _chat_stream(port, messages, geom["max_new_tokens"], turn_i > 0, 0.0):
                if ev[0] == "delta":
                    yield {"type": "token", "text": ev[1]}
                else:
                    _, text, timings, err = ev
            if err:
                state["error"] = err
                break
            infer._account(state, timings)
            yield _prefill_event(timings)
            dn = int(timings.get("predicted_n") or 0)
            dms = float(timings.get("predicted_ms") or 0.0)
            yield {"type": "decode", "tokens": dn, "ms": round(dms, 1),
                   "tok_per_s": round(dn / (dms / 1000.0), 1) if dms > 0 else None}

            kind = parse.classify(text) if downproject else "answer"
            turn = {"think": infer._think_of(text), "bbox_2d": None, "crop_png_b64": None}
            state["turns"].append(turn)
            messages.append({"role": "assistant", "content": text})

            if kind == "answer":
                answer, _src = parse.extract_answer_source(text)
                yield {"type": "turn_end", "kind": "answer", "think": turn["think"],
                       "bbox_2d": None}
                break

            if kind == "tool_call" and not forced:
                if state["zooms"] >= geom["max_zooms"]:
                    messages.append({"role": "user", "content": M4.FORCE_ANSWER_TEXT})
                    forced = True
                    yield {"type": "turn_end", "kind": "budget_spent", "think": turn["think"],
                           "bbox_2d": None}
                    continue
                box = parse.extract_bbox(text)
                state["zooms"] += 1
                turn["bbox_2d"] = list(box) if box else None
                yield {"type": "turn_end", "kind": "tool_call", "think": turn["think"],
                       "bbox_2d": turn["bbox_2d"]}
                if box is None:
                    messages.append({"role": "user", "content": M4.TOOL_ERROR_TEXT})
                    continue
                crop, _info = crop_from_bbox(image, box, geom["crop_max_side"])
                state["boxes"].append(list(box))
                if crop is None:
                    messages.append({"role": "user", "content": M4.TOOL_ERROR_TEXT})
                    yield {"type": "crop", "png_b64": None, "bbox_2d": list(box),
                           "vision_tokens": 0, "note": "box fell outside the image"}
                    continue
                vt = vision_tokens_for(*crop.size)
                state["vision_tokens"] += vt
                b64 = infer._png_b64(crop, 512)
                turn["crop_png_b64"] = b64
                yield {"type": "crop", "png_b64": b64, "bbox_2d": list(box),
                       "vision_tokens": vt}
                messages.append({"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": M4.b64_pil(crop)}},
                    {"type": "text", "text": M4.TOOL_RESPONSE_TEXT},
                ]})
                # Same as infer.py: hitting the cap forces the answer on the next turn.
                if state["zooms"] >= geom["max_zooms"]:
                    messages.append({"role": "user", "content": M4.FORCE_ANSWER_TEXT})
                    forced = True
                continue

            # malformed, or it called the tool after the budget was spent
            yield {"type": "turn_end", "kind": "malformed", "think": turn["think"],
                   "bbox_2d": None}
            if forced:
                answer, _src = parse.extract_answer_source(text)
                break
            messages.append({"role": "user", "content": M4.FORCE_ANSWER_TEXT})
            forced = True

        infer._set_answer(state, answer)
        out = infer._finish(state)

    out["type"] = "done"
    out["total_ms"] = round((time.perf_counter() - t_start) * 1000)
    dms = out.get("decode_ms") or 0
    out["decode_tok_per_s"] = round((out.get("decode_tokens") or 0) / (dms / 1000.0), 1) \
        if dms else None
    out.pop("turns", None)  # already streamed, and it bloats the final frame
    yield out


if __name__ == "__main__":
    import sys

    img = sys.argv[1] if len(sys.argv) > 1 else str(
        next((Path.home() / "archive/cost-aware-vlm/vstar_full/direct_attributes").glob("*.jpg")))
    q = sys.argv[2] if len(sys.argv) > 2 else "What colour is the flag?"
    for e in stream_episode(img, q, model="b", quant="q4", downproject=True):
        t = e.get("type")
        if t == "token":
            print(e["text"], end="", flush=True)
        else:
            print(f"\n[{t}] " + json.dumps(
                {k: (v[:24] + "…" if isinstance(v, str) and len(v) > 24 else v)
                 for k, v in e.items() if k != "type"})[:220], flush=True)
