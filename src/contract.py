"""Shared data contract. Every track codes against this file; nothing here depends on
torch, vllm, or transformers, so it imports on the Mac and on the rig.

Frozen 2026-08-15 02:50 PDT. Do not change a field name without updating every caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --- geometry -----------------------------------------------------------------

#: Thumbnail downsample factor. The model sees H/DOWNSAMPLE x W/DOWNSAMPLE and emits
#: bbox_2d in *thumbnail* pixels; the env multiplies by this to crop the original.
#: Gate 1 may change it (GOAL S9) -- it is read from the run config, never hardcoded
#: outside of this default.
DOWNSAMPLE = 4

#: Longest side of the thumbnail, in pixels, after downsampling.
THUMB_MAX_SIDE = 384

#: Fixed token budget for a returned crop: the crop is resized so its longest side is
#: this many pixels. Keeps vision_tokens per zoom roughly constant, which is what the
#: cost model assumes.
CROP_MAX_SIDE = 512

#: Hard ceiling on zooms per episode (GOAL S1). After this the model must answer.
MAX_ZOOMS = 3


@dataclass
class Sample:
    """One task instance. `image` is the ORIGINAL full-resolution RGB image."""

    sid: str  # stable id, unique within a split
    image: Any  # PIL.Image.Image, RGB, full resolution
    question: str  # raw question text, no formatting scaffold
    gold: str  # gold answer string
    source: str  # 'deepeyes' | 'vstar'
    options: list[str] | None = None  # multiple-choice options if the task has them
    meta: dict = field(default_factory=dict)


@dataclass
class Turn:
    """One assistant turn inside an episode. `token_ids` are the ids the SAMPLER
    emitted -- never a re-tokenization of `text` (GOAL S5, Gate 2)."""

    text: str
    token_ids: list[int]
    logprobs: list[float] | None = None
    bbox_2d: list[int] | None = None  # in thumbnail pixels, as emitted
    crop_vision_tokens: int = 0  # vision tokens the resulting crop cost, 0 if none


@dataclass
class Episode:
    """A full rollout: prompt -> turns -> answer -> reward."""

    sid: str
    question: str
    gold: str
    turns: list[Turn] = field(default_factory=list)
    answer: str | None = None
    correct: bool = False
    n_tool_calls: int = 0
    vision_tokens: int = 0  # thumbnail + every crop
    decode_tokens: int = 0  # every assistant token sampled
    cost_ms: float = 0.0
    reward: float = 0.0
    invalid_format: bool = False  # never produced a parseable <answer>
    #: which parse rule produced the answer: tagged | unclosed | prose | none
    answer_source: str = "none"

    def to_json(self) -> dict:
        return {
            "sid": self.sid,
            "question": self.question,
            "gold": self.gold,
            "answer": self.answer,
            "correct": self.correct,
            "n_tool_calls": self.n_tool_calls,
            "vision_tokens": self.vision_tokens,
            "decode_tokens": self.decode_tokens,
            "cost_ms": self.cost_ms,
            "reward": self.reward,
            "invalid_format": self.invalid_format,
            "answer_source": self.answer_source,
            "turns": [
                {
                    "text": t.text,
                    "n_ids": len(t.token_ids),
                    "bbox_2d": t.bbox_2d,
                    "crop_vision_tokens": t.crop_vision_tokens,
                }
                for t in self.turns
            ],
        }
