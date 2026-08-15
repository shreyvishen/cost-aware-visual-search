"""Parse the <think> / <tool_call> / <answer> contract (GOAL §18).

The model is asked for exactly this shape:

    <think>...</think> <tool_call>{"name":"image_zoom_in_tool",
      "arguments":{"bbox_2d":[x1,y1,x2,y2],"label":"..."}}</tool_call>
    ...tool result...
    <think>...</think> <answer>final</answer>

We parse leniently on the wrapper and strictly on the payload: a malformed bbox is a *failed*
tool call, not a crash. Nothing here raises.
"""
from __future__ import annotations

import json
import re

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
# Fallback: the model opened <answer> and hit the token cap before closing it.
OPEN_ANSWER_RE = re.compile(r"<answer>(.*)", re.S)
# Second fallback: the base model very often closes </think> and then answers in plain prose,
# ending the turn, without ever writing the tags (GOAL §8 predicted this). That IS an answer —
# it committed and stopped. Discarding it threw away ~31% of rollouts in the shape run.
# Guarded: the turn must have actually ENDED (<|im_end|>), so a run that merely hit the token
# cap mid-thought is not mistaken for a commitment.
PROSE_ANSWER_RE = re.compile(r"</think>\s*(.+?)\s*<\|im_end\|>", re.S)
BBOX_NUMS_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_think(text: str) -> str | None:
    m = THINK_RE.search(text)
    return m.group(1).strip() if m else None


def extract_answer(text: str) -> str | None:
    """Closed <answer> wins, then an unclosed one, then a finished prose turn."""
    ans, _ = extract_answer_source(text)
    return ans


def extract_answer_source(text: str) -> tuple[str | None, str]:
    """Same as extract_answer, but also says which rule fired, so the run can report how
    much it leaned on the prose fallback instead of hiding it."""
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).strip(), "tagged"
    m = OPEN_ANSWER_RE.search(text)
    if m:
        got = m.group(1).strip()
        if got:
            return got, "unclosed"
    if "<tool_call>" not in text:
        m = PROSE_ANSWER_RE.search(text)
        if m:
            got = m.group(1).strip()
            if got and "<think>" not in got:
                return got, "prose"
    return None, "none"


def extract_bbox(text: str) -> list[int] | None:
    """Return [x1, y1, x2, y2] in the frame of the image the model was shown, or None.

    Tries JSON first (the data's own wire format), then a bare-number fallback so a stray
    trailing comma does not cost us a rollout.
    """
    m = TOOL_RE.search(text)
    if not m:
        return None
    payload = m.group(1).strip()

    try:
        obj = json.loads(payload)
        args = obj.get("arguments", obj)
        box = args.get("bbox_2d") or args.get("bbox")
        if box is not None:
            return _coerce_box(box)
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    # Fallback: pull the first four numbers inside the bracket that follows "bbox".
    # Start after the '[' so the "2" in the key name "bbox_2d" is not read as a coordinate.
    tail = payload
    key = tail.find("bbox")
    if key >= 0:
        tail = tail[key:]
        br = tail.find("[")
        if br >= 0:
            tail = tail[br + 1:]
    nums = BBOX_NUMS_RE.findall(tail)
    if len(nums) >= 4:
        return _coerce_box(nums[:4])
    return None


def _coerce_box(box) -> list[int] | None:
    try:
        vals = [int(round(float(v))) for v in list(box)[:4]]
    except (TypeError, ValueError):
        return None
    if len(vals) != 4:
        return None
    return vals


def has_tool_call(text: str) -> bool:
    return TOOL_RE.search(text) is not None or "<tool_call>" in text


def classify(text: str) -> str:
    """'answer' | 'tool_call' | 'malformed'. Whichever tag appears FIRST wins, so a model
    that writes an answer and then keeps talking is taken at its first commitment."""
    a = ANSWER_RE.search(text) or OPEN_ANSWER_RE.search(text)
    t = TOOL_RE.search(text)
    if a and t:
        return "answer" if a.start() < t.start() else "tool_call"
    if a:
        return "answer"
    if t:
        return "tool_call"
    if extract_answer(text) is not None:
        return "answer"
    return "malformed"


if __name__ == "__main__":
    cases = [
        ('<think>look</think> <answer>red</answer>', "answer", None, "red"),
        (
            '<think>a</think><tool_call>{"name":"image_zoom_in_tool","arguments":'
            '{"bbox_2d":[10,20,30,40],"label":"sign"}}</tool_call>',
            "tool_call",
            [10, 20, 30, 40],
            None,
        ),
        # trailing comma -> json fails, number fallback saves it
        ('<tool_call>{"bbox_2d":[1,2,3,4,]}</tool_call>', "tool_call", [1, 2, 3, 4], None),
        # floats round
        ('<tool_call>{"arguments":{"bbox_2d":[1.4,2.6,3.0,4.0]}}</tool_call>', "tool_call", [1, 3, 3, 4], None),
        ('<think>hmm</think>', "malformed", None, None),
        ('<answer>blue', "answer", None, "blue"),
        # prose answer that ended the turn -> accepted
        ('reasoning</think>\n\nThe stripe is white.<|im_end|>', "answer", None,
         "The stripe is white."),
        # still thinking, hit the token cap -> NOT an answer
        ('reasoning</think>\n\nThe stripe is', "malformed", None, None),
        # prose that precedes a tool call is not an answer
        ('t</think>\n<tool_call>{"bbox_2d":[1,2,3,4]}</tool_call>', "tool_call",
         [1, 2, 3, 4], None),
    ]
    for text, want_cls, want_box, want_ans in cases:
        assert classify(text) == want_cls, (text, classify(text))
        assert extract_bbox(text) == want_box, (text, extract_bbox(text))
        assert extract_answer(text) == want_ans, (text, extract_answer(text))
    print(f"parse.py: {len(cases)} cases pass")
