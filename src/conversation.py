"""Builds a multi-turn conversation by CONCATENATING token ids.

This is the module that defends against landmine 1 (GOAL §5). The wrong-but-natural design
renders the whole conversation to text each turn and re-tokenizes it. That produces slightly
different ids at the boundaries, and you then train on tokens the model never emitted.

Here, an episode's id stream is built once and only ever appended to:

    [ prompt segment ] [ sampled ids ] [ tool-result segment ] [ sampled ids ] ...
      tokenized once     kept verbatim   tokenized once          kept verbatim

Assistant ids come from the sampler and are never re-encoded. Non-assistant segments are
tokenized exactly once, at the moment they are created. `Conversation.assert_ids_intact()`
is Gate 2.

Probed on the rig 2026-08-15 03:08: the processor adds no stray special tokens, so plain
concatenation is exact.
"""
from __future__ import annotations

from PIL import Image

from .zoom_env import FORCE_ANSWER, SYSTEM_PROMPT, USER_TEMPLATE

#: What the processor expands into the per-image vision tokens.
IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"

#: Wrapper for a returned crop. The leading <|im_end|> closes the assistant turn that the
#: sampler left open when it stopped on </tool_call>.
TOOL_RESULT_TEMPLATE = (
    "<|im_end|>\n<|im_start|>user\n<tool_response>"
    + IMAGE_TOKEN
    + "</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"
)

#: Same, for a tool call we could not execute. The model still spent the look.
TOOL_ERROR_TEMPLATE = (
    "<|im_end|>\n<|im_start|>user\n<tool_response>"
    "That box lies outside the image. Use coordinates inside the thumbnail."
    "</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"
)

#: Sent once the zoom budget is spent.
FORCE_ANSWER_TEMPLATE = (
    "<|im_end|>\n<|im_start|>user\n" + FORCE_ANSWER
    + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
)


class Conversation:
    """One episode's growing token stream plus the images it references, in order."""

    def __init__(self, proc, tok, question: str, thumbnail: Image.Image):
        self.proc = proc
        self.tok = tok
        self.ids: list[int] = []
        self.images: list[Image.Image] = []
        #: True where the token is one the policy sampled -> the only tokens we train on.
        self.trainable: list[bool] = []
        self.vision_tokens = 0
        self._sampled_spans: list[tuple[int, int]] = []

        text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n"
            + USER_TEMPLATE.format(image_token=IMAGE_TOKEN, question=question)
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        )
        self._append_segment(text, [thumbnail])

    # --- growth ---------------------------------------------------------------

    def _append_segment(self, text: str, images: list[Image.Image]) -> None:
        enc = self.proc(text=[text], images=images or None, return_tensors="pt")
        seg_ids = enc["input_ids"][0].tolist()
        self.ids.extend(seg_ids)
        self.trainable.extend([False] * len(seg_ids))
        self.images.extend(images)
        if images:
            grid = enc["image_grid_thw"]
            merge = self.proc.image_processor.merge_size
            for row in grid.tolist():
                self.vision_tokens += (row[0] * row[1] * row[2]) // (merge * merge)

    def append_sampled(self, token_ids: list[int]) -> None:
        """Append the sampler's own ids. Never a re-encoding of its text."""
        start = len(self.ids)
        self.ids.extend(token_ids)
        self.trainable.extend([True] * len(token_ids))
        self._sampled_spans.append((start, len(self.ids)))

    def append_crop(self, crop: Image.Image) -> int:
        """Append a tool result. Returns the vision tokens the crop cost."""
        before = self.vision_tokens
        self._append_segment(TOOL_RESULT_TEMPLATE, [crop])
        return self.vision_tokens - before

    def append_tool_error(self) -> None:
        self._append_segment(TOOL_ERROR_TEMPLATE, [])

    def append_force_answer(self) -> None:
        self._append_segment(FORCE_ANSWER_TEMPLATE, [])

    # --- Gate 2 ---------------------------------------------------------------

    def assert_ids_intact(self, sampled_history: list[list[int]]) -> None:
        """Gate 2 (GOAL §9). The ids sitting in the stream at every assistant span must be
        byte-for-byte the ids the sampler returned. Raises rather than degrading quietly."""
        if len(sampled_history) != len(self._sampled_spans):
            raise AssertionError(
                f"Gate 2: {len(sampled_history)} sampled turns but "
                f"{len(self._sampled_spans)} spans recorded"
            )
        for i, ((a, b), sampled) in enumerate(zip(self._sampled_spans, sampled_history)):
            in_stream = self.ids[a:b]
            if in_stream != sampled:
                raise AssertionError(
                    f"Gate 2 FAILED at turn {i}: stream ids != sampled ids. "
                    f"stream[{a}:{b}] len={len(in_stream)} vs sampled len={len(sampled)}. "
                    f"first divergence at "
                    f"{next((j for j, (x, y) in enumerate(zip(in_stream, sampled)) if x != y), 'len')}"
                )

    # --- readout --------------------------------------------------------------

    @property
    def n_sampled(self) -> int:
        return sum(b - a for a, b in self._sampled_spans)

    def as_request(self) -> dict:
        return {"prompt_token_ids": list(self.ids), "images": list(self.images), "prompt": None}
