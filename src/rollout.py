"""Multi-turn rollout collector.

Runs every episode in the step concurrently and advances them one turn at a time, so the
generator always sees a full batch. An episode ends when it emits <answer>, when it emits
nothing parseable, or when the zoom budget is spent and the forced final turn is taken.

Gate 2 runs on every episode before it is returned.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from . import parse
from .contract import Episode, Sample, Turn
from .conversation import Conversation
from .zoom_env import crop_from_bbox, make_thumbnail


@dataclass
class _State:
    sample: Sample
    conv: Conversation
    thumb: Image.Image
    scale: float
    ep: Episode
    sampled_history: list[list[int]] = field(default_factory=list)
    boxes: list[list[int]] = field(default_factory=list)
    crops: list[Image.Image] = field(default_factory=list)
    box_info: list[dict] = field(default_factory=list)
    zooms: int = 0
    n_boxes_in_frame: int = 0
    n_boxes_emitted: int = 0
    done: bool = False
    forced: bool = False


def collect(backend, proc, tok, samples: list[Sample], group_size: int, cfg: dict,
            temperature: float | None = None) -> list[Episode]:
    """Return group_size episodes per sample, in sample-major order."""
    max_zooms = cfg.get("max_zooms", 3)
    max_new = cfg.get("max_new_tokens", 160)
    temp = cfg.get("temperature", 1.0) if temperature is None else temperature
    downsample = cfg.get("downsample", 4)
    thumb_max = cfg.get("thumb_max_side", 384)
    crop_max = cfg.get("crop_max_side", 512)

    states: list[_State] = []
    for s in samples:
        thumb, scale = make_thumbnail(s.image, downsample, thumb_max)
        for g in range(group_size):
            conv = Conversation(proc, tok, s.question, thumb)
            ep = Episode(sid=s.sid, question=s.question, gold=s.gold)
            ep.vision_tokens = conv.vision_tokens
            states.append(_State(sample=s, conv=conv, thumb=thumb, scale=scale, ep=ep))

    # One extra pass so a forced final answer still gets generated.
    for _turn in range(max_zooms + 2):
        active = [st for st in states if not st.done]
        if not active:
            break
        outs = backend.generate([st.conv.as_request() for st in active],
                                max_new_tokens=max_new, temperature=temp, stop=None)
        for st, out in zip(active, outs):
            ids = out["token_ids"]
            st.conv.append_sampled(ids)
            st.sampled_history.append(list(ids))
            text = out["text"]
            st.ep.decode_tokens += len(ids)
            kind = parse.classify(text)

            if kind == "answer":
                st.ep.answer, st.ep.answer_source = parse.extract_answer_source(text)
                st.ep.turns.append(Turn(text=text, token_ids=ids))
                st.done = True
                continue

            if kind == "tool_call" and not st.forced:
                # Check the budget BEFORE spending it. Executing the crop first and then
                # noticing the budget was zero gave every "thumbnail-only" baseline one free
                # look, which quietly inflated it.
                if st.zooms >= max_zooms:
                    st.ep.turns.append(Turn(text=text, token_ids=ids))
                    st.conv.append_force_answer()
                    st.forced = True
                    continue
                box = parse.extract_bbox(text)
                st.ep.n_tool_calls += 1
                st.zooms += 1
                if box is None:
                    st.n_boxes_emitted += 1
                    st.ep.turns.append(Turn(text=text, token_ids=ids))
                    st.conv.append_tool_error()
                    st.box_info.append({"reason": "unparseable bbox"})
                else:
                    st.n_boxes_emitted += 1
                    crop, info = crop_from_bbox(st.sample.image, box, crop_max)
                    st.boxes.append(box)
                    st.box_info.append(info)
                    st.n_boxes_in_frame += int(bool(info.get("in_frame")))
                    if crop is None:
                        st.ep.turns.append(Turn(text=text, token_ids=ids, bbox_2d=box))
                        st.conv.append_tool_error()
                    else:
                        vt = st.conv.append_crop(crop)
                        st.ep.vision_tokens += vt
                        st.crops.append(crop)
                        st.ep.turns.append(
                            Turn(text=text, token_ids=ids, bbox_2d=box, crop_vision_tokens=vt))
                if st.zooms >= max_zooms:
                    st.conv.append_force_answer()
                    st.forced = True
                continue

            # Either malformed, or it called the tool after the budget was spent.
            st.ep.turns.append(Turn(text=text, token_ids=ids))
            if st.forced:
                # It had its forced turn and still did not answer. Take the last text as-is.
                st.ep.answer, st.ep.answer_source = parse.extract_answer_source(text)
                st.ep.invalid_format = st.ep.answer is None
                st.done = True
            else:
                st.conv.append_force_answer()
                st.forced = True

    episodes = []
    for st in states:
        if not st.done:
            st.ep.invalid_format = st.ep.answer is None
        st.conv.assert_ids_intact(st.sampled_history)  # Gate 2
        st.ep.meta_conv = st.conv          # type: ignore[attr-defined]
        st.ep.meta_boxes = st.boxes        # type: ignore[attr-defined]
        st.ep.meta_box_info = st.box_info  # type: ignore[attr-defined]
        st.ep.meta_crops = st.crops        # type: ignore[attr-defined]
        st.ep.meta_thumb = st.thumb        # type: ignore[attr-defined]
        st.ep.n_boxes_emitted = st.n_boxes_emitted    # type: ignore[attr-defined]
        st.ep.n_boxes_in_frame = st.n_boxes_in_frame  # type: ignore[attr-defined]
        episodes.append(st.ep)
    return episodes
