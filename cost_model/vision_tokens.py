"""Vision-token accounting for Qwen3.5-VL-4B.

The reward is `cost_ms = a*vision_tokens + b*decode_tokens + c*tool_calls`.
`a` was fitted on the M4 Max against the counts `vision_tokens_for` returns.
The training env MUST import this function. If the rig's HF processor disagrees
with it, the reward is silently wrong and nothing raises.

TWO RULES, AND THEY ARE NOT THE SAME
------------------------------------
`vision_tokens_for`      -- what the HF processor on the rig reports. THIS IS
                            THE REWARD'S UNIT. Use it everywhere in training.
`vision_tokens_llamacpp` -- what llama.cpp's clip actually does on the Mac. Use
                            it only to predict demo latency.

Both use factor = patch_size * merge_size = 16 * 2 = 32 px per token per side.
They differ in three places:

  |             | HF (reward)          | llama.cpp (Mac demo) |
  |-------------|----------------------|----------------------|
  | min_pixels  | 65,536  (64 tokens)  | 9,216  (9 tokens)    |
  | max_pixels  | 16,777,216 (16384 t) | 4,194,304 (4096 t)   |
  | rounding    | banker's (Python)    | half-away (std::round)|

So a 100x100 crop is 64 tokens to the reward and 9 tokens to the Mac, and
1104x480 is 510 vs 525. **Every image size in the fitted grid -- 256, 384, 512,
768 and 1024 square -- lands on an identical count under both rules**, so `a` is
valid in HF units across the whole measured range. Outside it, prefer HF.

PROVENANCE (measured 2026-08-15, not guessed)
---------------------------------------------
HF: `Qwen2VLImageProcessor` from /srv/ai/models/current/qwen35-4b on the rig,
28 sizes, reading `image_grid_thw`. Its preprocessor_config.json carries
patch_size 16, merge_size 2, size.shortest_edge 65536, size.longest_edge
16777216.
llama.cpp: `llama-server` + `mmproj-F16.gguf` on the M4, 24 sizes, reading
`timings.prompt_n` minus the constant 16-token text prompt.
Both sets are asserted below. Run `python3 vision_tokens.py` on either machine.
"""

import math

PATCH_SIZE = 16
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE  # 32 px per vision token per side

# HF: from preprocessor_config.json size.shortest_edge / size.longest_edge.
HF_MIN_PIXELS = 65_536       # 64 tokens
HF_MAX_PIXELS = 16_777_216   # 16384 tokens

# llama.cpp clip, derived from measurement.
GGUF_MIN_PIXELS = 9_216      # 9 tokens
GGUF_MAX_PIXELS = 4_194_304  # 4096 tokens

TEXT_TOKEN_OVERHEAD_FIRST_TURN = 16
TEXT_TOKEN_OVERHEAD_PER_EXTRA_TURN = 17


def _ceil_by_factor(n, factor=FACTOR):
    return int(math.ceil(n / factor)) * factor


def _floor_by_factor(n, factor=FACTOR):
    return int(math.floor(n / factor)) * factor


def _smart_resize(width, height, min_pixels, max_pixels, round_fn):
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")
    w_bar = max(FACTOR, round_fn(width))
    h_bar = max(FACTOR, round_fn(height))
    if w_bar * h_bar > max_pixels:
        beta = math.sqrt((width * height) / max_pixels)
        w_bar = max(FACTOR, _floor_by_factor(width / beta))
        h_bar = max(FACTOR, _floor_by_factor(height / beta))
    elif w_bar * h_bar < min_pixels:
        beta = math.sqrt(min_pixels / (width * height))
        w_bar = _ceil_by_factor(width * beta)
        h_bar = _ceil_by_factor(height * beta)
    return w_bar, h_bar


def _round_bankers(n, factor=FACTOR):
    """transformers `round_by_factor`: Python round(), i.e. banker's rounding."""
    return int(round(n / factor)) * factor


def _round_half_away(n, factor=FACTOR):
    """C++ std::round: halves go away from zero."""
    return int(math.floor(n / factor + 0.5)) * factor


def smart_resize_hf(width, height):
    """Pixel size the HF processor feeds the vision tower."""
    return _smart_resize(width, height, HF_MIN_PIXELS, HF_MAX_PIXELS,
                         _round_bankers)


def smart_resize_llamacpp(width, height):
    """Pixel size llama.cpp's clip feeds the vision tower."""
    return _smart_resize(width, height, GGUF_MIN_PIXELS, GGUF_MAX_PIXELS,
                         _round_half_away)


def vision_tokens_for(width, height):
    """Vision tokens one image costs. HF units -- the reward's units."""
    w_bar, h_bar = smart_resize_hf(width, height)
    return (w_bar // FACTOR) * (h_bar // FACTOR)


def vision_tokens_llamacpp(width, height):
    """Vision tokens llama.cpp charges on the Mac. Demo latency only."""
    w_bar, h_bar = smart_resize_llamacpp(width, height)
    return (w_bar // FACTOR) * (h_bar // FACTOR)


# Every size fed through the real HF processor on the rig, 2026-08-15.
MEASURED_HF = {
    (256, 256): 64, (384, 384): 144, (512, 512): 256, (768, 768): 576,
    (1024, 1024): 1024, (300, 200): 70, (1000, 500): 496, (100, 100): 64,
    (64, 64): 64, (33, 33): 64, (2250, 1500): 3290, (1600, 1200): 1900,
    (50, 700): 90, (1104, 480): 510, (1136, 480): 540, (200, 40): 72,
    (40, 200): 72, (96, 96): 64, (160, 160): 64, (80, 80): 64, (48, 48): 64,
    (120, 80): 70, (3000, 2000): 5828, (4000, 3000): 11750, (128, 128): 64,
    (224, 224): 64, (255, 255): 64, (257, 257): 64,
}

# Every size fed through llama-server on the M4, 2026-08-15.
MEASURED_LLAMACPP = {
    (256, 256): 64, (384, 384): 144, (512, 512): 256, (768, 768): 576,
    (1024, 1024): 1024, (300, 200): 54, (1000, 500): 496, (100, 100): 9,
    (64, 64): 9, (33, 33): 9, (2250, 1500): 3290, (1600, 1200): 1900,
    (50, 700): 44, (1104, 480): 525, (1136, 480): 540, (200, 40): 14,
    (40, 200): 14, (96, 96): 9, (160, 160): 25, (80, 80): 9, (48, 48): 9,
    (120, 80): 12, (3000, 2000): 4056, (4000, 3000): 4015,
}

# Every image size used in the timing grid. Both rules must agree here, or the
# fitted `a` is not in the reward's units.
FIT_GRID_SIZES = [256, 384, 512, 768, 1024]


def _selftest():
    bad = [(w, h, want, vision_tokens_for(w, h))
           for (w, h), want in MEASURED_HF.items()
           if vision_tokens_for(w, h) != want]
    if bad:
        raise AssertionError(f"vision_tokens_for disagrees with HF: {bad}")

    bad = [(w, h, want, vision_tokens_llamacpp(w, h))
           for (w, h), want in MEASURED_LLAMACPP.items()
           if vision_tokens_llamacpp(w, h) != want]
    if bad:
        raise AssertionError(f"vision_tokens_llamacpp disagrees with the M4: {bad}")

    bad = [px for px in FIT_GRID_SIZES
           if vision_tokens_for(px, px) != vision_tokens_llamacpp(px, px)]
    if bad:
        raise AssertionError(f"HF and llama.cpp disagree inside the fit grid: {bad}")

    print(f"vision_tokens_for:      {len(MEASURED_HF)}/{len(MEASURED_HF)} HF points match")
    print(f"vision_tokens_llamacpp: {len(MEASURED_LLAMACPP)}/{len(MEASURED_LLAMACPP)} M4 points match")
    print(f"fit grid {FIT_GRID_SIZES}: both rules agree, so `a` is in HF units")


if __name__ == "__main__":
    _selftest()
