"""vLLM generation backend for the GRPO rollout collector.

PROBE RESULT -- 2026-08-15 03:0x PDT, rig `home-lab`, 1x RTX 3090 (GPU 2).
Full write-up in `.notes/vllm_probe.md`.

* **vllm version:** `0.23.1rc1.dev350+g0a3e2dbc0` (torch 2.11.0+cu130,
  transformers 5.13.0.dev0, peft 0.20.0).
* **LoRA works.** `Qwen3_5ForConditionalGeneration` declares `SupportsLoRA`, and an
  `LLM(enable_lora=True, max_lora_rank=16, max_loras=2)` engine loads in ~39 s and generates
  with a rank-16 PEFT adapter. Multimodal is not a blocker for LoRA on this build.
* **Hot swap works.** Two different adapters were served by the SAME live engine, back to back,
  with different `lora_int_id`s. No restart, and the sampled ids differed -- the swap really
  takes effect. `load_adapter()` therefore just re-points a `LoRARequest`; it never rebuilds
  the engine.
* **`prompt_token_ids` CAN be combined with images.** This is the risky one and it came back
  **green**: `{"prompt_token_ids": [...], "multi_modal_data": {"image": [...]}}` generates
  normally. This build does NOT require a text prompt alongside `multi_modal_data`. The harness
  can feed sampler ids straight back in on turn 2+ and never re-tokenize. That is the project's
  #1 landmine (GOAL S16, Gate 2) and it is closed. Do not build the text-prompt fallback.
* **Token ids + logprobs out.** `output.outputs[0].token_ids` are the sampler's own ids, and
  `SamplingParams(logprobs=0)` returns the sampled token's logprob per step.

Measured throughput -- 32 concurrent requests, one 384x288 thumbnail each, 194-token text
prompt (301 tokens per request after image expansion), 128 new tokens, temperature 1.0:

    GPU 2      tp=1   rank-16 LoRA   15.2 s   266.5 gen tok/s
    GPU 2      tp=1   base           8.1 s    504.0 gen tok/s
    GPU 2+3    tp=2   rank-16 LoRA   16.6 s   245.1 gen tok/s
    GPU 2+3    tp=2   base           9.0 s    453.5 gen tok/s

**Do not use tensor parallelism.** `tp=2` is SLOWER than `tp=1` -- these 3090s have no NVLink,
so the per-layer all-reduce costs more than the split compute saves on a 4B model. Run two
independent backends instead, `VLLMBackend(gpus=[2])` and `VLLMBackend(gpus=[3])` in separate
processes, and split the rollout batch across them for ~533 gen tok/s aggregate.

LoRA costs ~1.9x on generation (266 vs 504 tok/s) and is not optional. Budget with 266 tok/s
per GPU. Batching is what makes this viable at all: GOAL S8 measured ~6 tok/s single-stream, so
always issue one large `generate()` call, never a loop over single requests.

Traps this module already handles, learned the hard way:

1. **`spawn` re-runs the importing script.** vLLM forces
   `VLLM_WORKER_MULTIPROC_METHOD=spawn` once CUDA is initialized. Any entry point that
   constructs this class at module scope will fork-bomb itself and die with a
   `freeze_support()` RuntimeError that looks like a vLLM bug and is not.
   **Guard every entry point with `if __name__ == "__main__":`.**
2. **AppleDouble junk in the model dir.** `/srv/ai/models/current/qwen35-4b/` contains 4 KB
   `._*.safetensors` files that match vLLM's glob, and the shards have non-standard names. The
   dir belongs to another user -- do not clean it. Point `model_path` at the symlink farm
   `/srv/ai/runs/_probe/model` (see `.notes/vllm_probe.md`).
3. **`VLLM_ATTENTION_BACKEND` is ignored by this build** -- it logs
   "Unknown vLLM environment variable". Harmless. `VLLM_USE_FLASHINFER_SAMPLER=0` still matters.
4. **`lora_int_id` is the cache key.** vLLM caches an adapter by its int id, so serving new
   weights under an old id silently replays the stale adapter. `load_adapter(path, version)`
   sets `lora_int_id = version`, so the caller must pass a strictly increasing version.
"""

from __future__ import annotations

import gc
import os
from typing import Any

#: Filled in from the probe. Aggregate generation tokens/sec, batch of 32, 128 new tokens each.
THROUGHPUT: dict[str, Any] = {
    "note": "see .notes/vllm_probe.md for the measured table",
}

_REQUIRED_ENV = {
    # This build hits a failing nvcc compile with the flashinfer sampler on.
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    # vLLM switches to spawn anyway once CUDA is up; setting it early avoids a late override.
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
}


class VLLMBackend:
    """Batched multi-image generation with a hot-swappable LoRA adapter.

    The engine is built once. `load_adapter` only swaps which adapter subsequent `generate`
    calls use, which is what makes a per-step GRPO adapter refresh affordable.
    """

    def __init__(self, model_path: str, gpus: list[int], max_lora_rank: int = 16,
                 max_images: int = 5) -> None:
        self.model_path = model_path
        self.gpus = list(gpus)
        self.max_lora_rank = max_lora_rank
        self.max_images = max_images
        self._lora = None          # current LoRARequest, or None for the base model
        self._adapter_path = None
        self._adapter_version = None

        for k, v in _REQUIRED_ENV.items():
            os.environ.setdefault(k, v)
        # Must be set before torch touches CUDA. Constructing this class at module scope in a
        # process that already initialized CUDA elsewhere will silently use the wrong devices.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.gpus)

        from vllm import LLM  # imported late so CUDA_VISIBLE_DEVICES lands first

        self.llm = LLM(
            model=model_path,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            max_loras=2,
            enforce_eager=True,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": max_images},
            max_model_len=8192,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=len(self.gpus),
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

    # -- adapter ---------------------------------------------------------------

    def load_adapter(self, adapter_path: str, version: int) -> None:
        """Make this adapter the one used by subsequent `generate()` calls.

        `version` becomes the `lora_int_id`. It MUST increase every time the weights on disk
        change, or vLLM serves the cached previous adapter. Pass `adapter_path=None` to drop
        back to the base model.
        """
        if adapter_path is None:
            self._lora = None
            self._adapter_path = None
            self._adapter_version = None
            return

        from vllm.lora.request import LoRARequest

        if not os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
            raise FileNotFoundError(f"no adapter_config.json under {adapter_path}")
        if self._adapter_version is not None and version <= self._adapter_version:
            raise ValueError(
                f"lora version must increase: got {version}, already served "
                f"{self._adapter_version}. vLLM caches by lora_int_id, so a repeated id "
                f"would replay stale weights."
            )
        self._lora = LoRARequest(f"policy_v{version}", version, adapter_path)
        self._adapter_path = adapter_path
        self._adapter_version = version

    # -- generation ------------------------------------------------------------

    def generate(self, requests: list[dict], max_new_tokens: int, temperature: float,
                 stop: list[str]) -> list[dict]:
        """Batched generation.

        requests: [{"prompt_token_ids": list[int] | None, "prompt": str | None,
                    "images": list[PIL.Image]}]
        returns:  [{"token_ids": list[int], "text": str, "logprobs": list[float] | None,
                    "finish_reason": str}]

        Returns the sampler's own token ids -- never a re-tokenization of the text.
        `prompt_token_ids` wins when both are given; images may accompany either form
        (verified on this build).

        `text` is `tokenizer.decode(token_ids)`, NOT vLLM's own `output.text`. Measured
        difference, not a stylistic one: with `stop=["\\n"]` the sampler emitted a single
        `"\\n\\n"` token, and vLLM truncated its `text` at the stop string's character position
        while keeping the whole token in `token_ids` -- so `decode(ids) != output.text` by a
        partial token. The ids are what GRPO trains on, so the text must be derived from them or
        the parser and the loss disagree about what the model said (Gate 2, GOAL S5).
        Consequence the caller must know: when a stop string ends mid-token, `text` runs a few
        characters PAST the stop string. Parse for the closing tag, do not assume it is the last
        thing in the string.

        Stop strings are included rather than stripped (`include_stop_str_in_output=True`) --
        the closing tag is part of the sequence GRPO trains on.
        """
        from vllm import SamplingParams

        if not requests:
            return []

        prompts = []
        for i, r in enumerate(requests):
            ids = r.get("prompt_token_ids")
            text = r.get("prompt")
            if ids:
                p: dict[str, Any] = {"prompt_token_ids": list(ids)}
            elif text is not None:
                p = {"prompt": text}
            else:
                raise ValueError(f"request {i}: needs prompt_token_ids or prompt")
            images = r.get("images") or []
            if len(images) > self.max_images:
                raise ValueError(
                    f"request {i}: {len(images)} images exceeds max_images={self.max_images}"
                )
            if images:
                p["multi_modal_data"] = {"image": list(images)}
            prompts.append(p)

        sp = SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_new_tokens,
            stop=list(stop) if stop else None,
            include_stop_str_in_output=True,
            logprobs=0,
        )

        outs = self.llm.generate(prompts, sp, lora_request=self._lora)

        results = []
        for o in outs:
            c = o.outputs[0]
            token_ids = list(c.token_ids)
            logprobs = None
            if c.logprobs:
                # c.logprobs[i] maps token_id -> Logprob for step i. Pull the sampled token's.
                logprobs = []
                for step, tid in zip(c.logprobs, token_ids):
                    lp = step.get(tid)
                    logprobs.append(float(lp.logprob) if lp is not None else float("nan"))
            results.append({
                "token_ids": token_ids,
                # Derived from the ids on purpose -- see the docstring. Never c.text.
                "text": self.tokenizer.decode(token_ids),
                "logprobs": logprobs,
                "finish_reason": c.finish_reason,
            })
        return results

    # -- teardown --------------------------------------------------------------

    def shutdown(self) -> None:
        """Release the GPUs. Safe to call twice."""
        llm = getattr(self, "llm", None)
        if llm is None:
            return
        try:
            engine = getattr(llm, "llm_engine", None)
            core = getattr(engine, "engine_core", None)
            if core is not None and hasattr(core, "shutdown"):
                core.shutdown()
        except Exception:
            pass
        self.llm = None
        self._lora = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
