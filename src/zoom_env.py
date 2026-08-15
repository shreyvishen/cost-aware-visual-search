"""The zoom environment: thumbnail, crop, and the coordinate frame.

The one rule that keeps this honest (GOAL §1, landmine 3): **every box is in the frame of the
thumbnail, and every crop is taken from the ORIGINAL image.** We never zoom into a zoom. The
thumbnail stays in context, so all four looks reference one frame and there is no nested
scale factor to get wrong.

Thumbnail rule (decided 2026-08-15 03:00, see LEDGER):

    scale = max(DOWNSAMPLE, longest_side / THUMB_MAX_SIDE)

Train images are COCO (640 px longest side); eval images are V*Bench (2250 px). A fixed pixel
size would destroy wildly different amounts of information on the two sets. A fixed ratio alone
would leave a 561 px thumbnail on eval, which is expensive and too easy. Taking the max of the
two gives "at least 4x destroyed, never more than 384 px of context" on both sets.
Gate 1 tunes DOWNSAMPLE, nothing else.
"""
from __future__ import annotations

from PIL import Image

from .contract import CROP_MAX_SIDE, DOWNSAMPLE, THUMB_MAX_SIDE

#: Smallest crop we will take from the original, in original pixels. A degenerate box gets
#: grown around its centre rather than rejected — the model still paid for the look.
MIN_CROP_PX = 48
#: Widest aspect ratio a crop may have before we pad it out. Extreme strips tokenize badly.
MAX_CROP_ASPECT = 4.0

#: Verbatim from the training parquet (GOAL §18 — "use the data's own protocol"). Do not
#: rewrite this; the model was function-calling trained against this exact shape.
SYSTEM_PROMPT = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}
</tool_call>"""

#: From the parquet user turn, plus two added sentences. Both additions are documented
#: deviations, measured before and after (see LEDGER 03:45):
#:
#:  1. THE COORDINATE FRAME. The parquet protocol never says what frame `bbox_2d` is in.
#:     Qwen emits boxes on its native normalised 0-1000 grid, so with a 160x120 thumbnail
#:     31 of 31 probe boxes landed outside the image and every single zoom returned nothing.
#:     We state the 0-1000 frame instead of fighting it. It is also the better frame: it does
#:     not change meaning when Gate 1 resizes the thumbnail.
#:  2. SHORT ANSWERS. Every gold in this corpus is a full sentence, but the scorer grades on
#:     polarity and content words. Asking for the short form costs nothing and saves decode
#:     tokens, which are the expensive term in the measured cost model.
USER_TEMPLATE = (
    "{image_token}\n{question}\n"
    "The image above is a reduced-resolution view of a larger image. To see fine detail, "
    "call image_zoom_in_tool with bbox_2d in NORMALISED coordinates on a 0-1000 grid: "
    "[x1, y1, x2, y2], where x=0 is the left edge and x=1000 the right edge, y=0 the top "
    "and y=1000 the bottom. The crop is taken from the full-resolution original. Always use "
    "this 0-1000 grid, never pixels.\n"
    "Think first, call **image_zoom_in_tool** if needed, then answer. "
    "Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  "
    "<answer>...</answer> "
    "Make the answer as short as possible: yes, no, or a single word or phrase."
)

#: The normalised grid the model works in.
COORD_SCALE = 1000

#: Appended when the zoom budget is spent, so the episode always terminates in an answer.
FORCE_ANSWER = (
    "You have used your zoom budget. Answer now, using what you have seen. "
    "Reply with <think>...</think> <answer>...</answer> only."
)


def make_thumbnail(image: Image.Image, downsample: int = DOWNSAMPLE,
                   max_side: int = THUMB_MAX_SIDE) -> tuple[Image.Image, float]:
    """Return (thumbnail, scale) where scale converts thumbnail px -> original px."""
    w, h = image.size
    scale = max(float(downsample), max(w, h) / float(max_side))
    tw, th = max(1, int(round(w / scale))), max(1, int(round(h / scale)))
    thumb = image.resize((tw, th), Image.BICUBIC)
    return thumb, scale


def crop_from_bbox(image: Image.Image, bbox_norm: list[int],
                   crop_max_side: int = CROP_MAX_SIDE) -> tuple[Image.Image | None, dict]:
    """Crop the ORIGINAL image using a box on the normalised 0-1000 grid.

    One frame for everything. The thumbnail's pixel size never enters the mapping, so Gate 1
    can resize the thumbnail without changing what a box means.

    Returns (crop, info). `crop` is None only when the box is unusable after clamping —
    a wasted-but-charged tool call, never an exception. `info` records the raw box and the
    fixed box so the rollout log shows exactly what the policy asked for.
    """
    ow, oh = image.size
    info: dict = {"bbox_norm_raw": list(bbox_norm), "clamped": False, "grown": False}

    x1, y1, x2, y2 = bbox_norm
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    info["in_frame"] = 0 <= x1 < x2 <= COORD_SCALE and 0 <= y1 < y2 <= COORD_SCALE
    if not info["in_frame"]:
        info["clamped"] = True
    x1 = max(0, min(x1, COORD_SCALE))
    x2 = max(0, min(x2, COORD_SCALE))
    y1 = max(0, min(y1, COORD_SCALE))
    y2 = max(0, min(y2, COORD_SCALE))
    if x2 <= x1 or y2 <= y1:
        info["reason"] = "degenerate after clamp"
        return None, info

    # Normalised grid -> original px, per axis so rounding cannot drift.
    sx, sy = ow / float(COORD_SCALE), oh / float(COORD_SCALE)
    ox1, oy1 = int(round(x1 * sx)), int(round(y1 * sy))
    ox2, oy2 = int(round(x2 * sx)), int(round(y2 * sy))

    ox1, oy1, ox2, oy2, grown = _grow_to_minimum(ox1, oy1, ox2, oy2, ow, oh)
    info["grown"] = grown
    info["bbox_orig"] = [ox1, oy1, ox2, oy2]

    crop = image.crop((ox1, oy1, ox2, oy2))
    cw, ch = crop.size
    if cw < 1 or ch < 1:
        info["reason"] = "empty crop"
        return None, info

    # Resize so the longest side is exactly crop_max_side. Up as well as down: a small crop
    # is the whole point of zooming, and a constant size keeps vision_tokens per look
    # constant, which is what the cost model assumes.
    r = crop_max_side / float(max(cw, ch))
    crop = crop.resize((max(1, int(round(cw * r))), max(1, int(round(ch * r)))), Image.BICUBIC)
    info["crop_size"] = list(crop.size)
    info["area_frac"] = ((ox2 - ox1) * (oy2 - oy1)) / float(ow * oh)
    return crop, info


def _grow_to_minimum(x1: int, y1: int, x2: int, y2: int, ow: int,
                     oh: int) -> tuple[int, int, int, int, bool]:
    """Grow a too-small or too-thin box around its centre, then shift it back inside."""
    grown = False
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = x2 - x1, y2 - y1

    if w < MIN_CROP_PX:
        w, grown = MIN_CROP_PX, True
    if h < MIN_CROP_PX:
        h, grown = MIN_CROP_PX, True
    if w > MAX_CROP_ASPECT * h:
        h, grown = w / MAX_CROP_ASPECT, True
    if h > MAX_CROP_ASPECT * w:
        w, grown = h / MAX_CROP_ASPECT, True

    w, h = min(w, ow), min(h, oh)
    nx1, ny1 = int(round(cx - w / 2)), int(round(cy - h / 2))
    nx2, ny2 = nx1 + int(round(w)), ny1 + int(round(h))
    if nx1 < 0:
        nx2, nx1 = nx2 - nx1, 0
    if ny1 < 0:
        ny2, ny1 = ny2 - ny1, 0
    if nx2 > ow:
        nx1, nx2 = max(0, nx1 - (nx2 - ow)), ow
    if ny2 > oh:
        ny1, ny2 = max(0, ny1 - (ny2 - oh)), oh
    return nx1, ny1, nx2, ny2, grown


def draw_boxes(thumb: Image.Image, boxes_norm: list[list[int]]) -> Image.Image:
    """Thumbnail with the chosen boxes drawn in order — the "where it looked" picture.

    Boxes arrive on the 0-1000 grid and are mapped to thumbnail pixels here.
    """
    from PIL import ImageDraw

    out = thumb.convert("RGB").copy()
    tw, th = out.size
    d = ImageDraw.Draw(out)
    colors = ["#ff3b30", "#ffcc00", "#34c759", "#0a84ff"]
    for i, b in enumerate(boxes_norm[:4]):
        x1, y1, x2, y2 = b
        px = [x1 * tw / COORD_SCALE, y1 * th / COORD_SCALE,
              x2 * tw / COORD_SCALE, y2 * th / COORD_SCALE]
        px = [max(0, min(px[0], tw - 1)), max(0, min(px[1], th - 1)),
              max(1, min(px[2], tw)), max(1, min(px[3], th))]
        if px[2] <= px[0] or px[3] <= px[1]:
            continue
        d.rectangle(px, outline=colors[i % len(colors)], width=2)
        d.text((px[0] + 2, px[1] + 2), str(i + 1), fill=colors[i % len(colors)])
    return out


if __name__ == "__main__":
    img = Image.new("RGB", (2250, 1500), "white")
    thumb, scale = make_thumbnail(img)
    assert thumb.size == (384, 256), thumb.size
    assert abs(scale - 2250 / 384) < 1e-6, scale

    small = Image.new("RGB", (640, 480), "white")
    t2, s2 = make_thumbnail(small)
    assert t2.size == (160, 120), t2.size
    assert s2 == 4.0, s2

    # the box the smoke test actually produced must now work
    crop, info = crop_from_bbox(img, [686, 175, 937, 317])
    assert crop is not None and info["in_frame"] is True, info
    assert info["bbox_orig"] == [1544, 262, 2108, 476], info["bbox_orig"]

    # a tiny box still yields a usable crop
    crop, info = crop_from_bbox(img, [500, 500, 502, 501])
    assert crop is not None and max(crop.size) == CROP_MAX_SIDE, (crop.size, info)
    assert info["grown"] is True

    # a partly out-of-frame box clamps rather than crashing
    crop, info = crop_from_bbox(img, [-50, -50, 200, 200])
    assert info["clamped"] is True and crop is not None

    # a fully out-of-frame box is a wasted look, not an exception
    crop, info = crop_from_bbox(img, [1200, 1200, 1400, 1400])
    assert crop is None, info

    # boxes stay inside the original
    crop, info = crop_from_bbox(img, [990, 990, 1000, 1000])
    x1, y1, x2, y2 = info["bbox_orig"]
    assert 0 <= x1 < x2 <= 2250 and 0 <= y1 < y2 <= 1500, info

    # the frame is independent of thumbnail size: same box, same crop, either thumbnail
    c1, i1 = crop_from_bbox(img, [200, 200, 400, 400])
    c2, i2 = crop_from_bbox(img.resize((900, 600)), [200, 200, 400, 400])
    assert i1["area_frac"] == i2["area_frac"], (i1, i2)
    print("zoom_env.py: all checks pass")
