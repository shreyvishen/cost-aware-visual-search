#!/usr/bin/env python
"""Gate 3 -- vLLM / trainer sync. Run after every adapter swap.

Generates ONE fixed prompt at temperature 0 from
  (i)  vLLM with adapter X, and
  (ii) HF `generate` on the same base + adapter X,
then asserts the sampled TOKEN IDS match. Ids, not text -- text can agree while the ids differ,
and it is the ids GRPO trains on (GOAL S5, S9).

Exits 0 on match, nonzero on mismatch.

Why a prefix and not the whole sequence: vLLM and HF run different kernels, so greedy decoding
drifts apart once two logits sit within bf16 noise of each other. The gate checks the first
`--prefix` ids, which is what catches the failure this gate exists for -- a wrong adapter, a
stale `lora_int_id`, a retokenized prompt, a different chat template. Those break token 1, not
token 40. `--prefix 0` compares everything and is expected to be flaky; do not wire it into the
training loop.

MEASURED 2026-08-15 with the rank-16 dummy adapter: the two sides agree for 24 tokens and
diverge at index 24 (vllm=1156, hf=12515). `--prefix 0` therefore exits 1, which is the proof
this gate can actually fail. The default is **16**, deliberately inside that margin -- 24 would
pass by exactly one token and flake the first time anything shifts.

Usage:
    python gates/g3_sync.py --adapter /srv/ai/runs/<run>/adapter_step0010 --version 10
    python gates/g3_sync.py --adapter ... --version 11 --prefix 16

VERDICT 2026-08-15: vLLM is IN (`.notes/vllm_probe.md`), so this gate does real work.
Verified both ways: exit 0 on match, exit 1 on mismatch.
"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_MODEL = "/srv/ai/runs/_probe/model"
#: The fixed prompt. Text-only on purpose: the image pipeline is exercised by Gate 2 and the
#: smoke roll, and holding the vision tower out keeps this gate fast and deterministic.
FIXED_PROMPT = "Answer with one word. What color is a clear midday sky?"


def build_prompt(tokenizer) -> tuple[str, list[int]]:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": FIXED_PROMPT}],
        tokenize=False, add_generation_prompt=True,
    )
    return text, tokenizer.encode(text)


def run_vllm(model_path: str, adapter: str, version: int, ids: list[int], n: int) -> list[int]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=model_path, enable_lora=True, max_lora_rank=16, max_loras=2,
              enforce_eager=True, dtype="bfloat16", limit_mm_per_prompt={"image": 5},
              max_model_len=8192, gpu_memory_utilization=0.85, trust_remote_code=True)
    lr = LoRARequest(f"g3_v{version}", version, adapter) if adapter else None
    out = llm.generate([{"prompt_token_ids": list(ids)}],
                       SamplingParams(temperature=0.0, max_tokens=n), lora_request=lr)[0]
    return list(out.outputs[0].token_ids)


def run_hf(model_path: str, adapter: str, ids: list[int], n: int) -> list[int]:
    import torch
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda:0")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    inp = torch.tensor([ids], device=model.device)
    with torch.no_grad():
        out = model.generate(input_ids=inp, max_new_tokens=n, do_sample=False,
                             temperature=None, top_p=None, top_k=None)
    return out[0, len(ids):].tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("G3_MODEL", DEFAULT_MODEL))
    ap.add_argument("--adapter", default=None, help="PEFT adapter dir; omit for base model")
    ap.add_argument("--version", type=int, default=1, help="lora_int_id; must increase per swap")
    ap.add_argument("--tokens", type=int, default=32, help="tokens to generate")
    ap.add_argument("--prefix", type=int, default=16,
                    help="ids that must match; 0 means all of them (expected to be flaky)")
    ap.add_argument("--side", choices=["both", "vllm", "hf"], default="both",
                    help="internal: run one side only, used by the two-process runner")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    _, ids = build_prompt(tok)

    if args.side == "vllm":
        print("IDS " + ",".join(str(i) for i in
                                run_vllm(args.model, args.adapter, args.version, ids, args.tokens)))
        return 0
    if args.side == "hf":
        print("IDS " + ",".join(str(i) for i in run_hf(args.model, args.adapter, ids, args.tokens)))
        return 0

    # Both sides in one process would put two full copies of the model on one GPU and race for
    # vLLM's 85% pool, so each side runs as its own subprocess.
    import subprocess

    def side(name: str) -> list[int]:
        cmd = [sys.executable, os.path.abspath(__file__), "--side", name,
               "--model", args.model, "--version", str(args.version),
               "--tokens", str(args.tokens)]
        if args.adapter:
            cmd += ["--adapter", args.adapter]
        env = dict(os.environ)
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        for line in p.stdout.splitlines():
            if line.startswith("IDS "):
                return [int(x) for x in line[4:].split(",") if x]
        print(f"G3 FAIL: {name} side produced no ids (exit {p.returncode})")
        print(p.stdout[-2000:])
        print(p.stderr[-3000:])
        sys.exit(2)

    v_ids = side("vllm")
    h_ids = side("hf")

    n = args.prefix if args.prefix > 0 else max(len(v_ids), len(h_ids))
    a, b = v_ids[:n], h_ids[:n]

    print(f"G3 adapter={args.adapter or '<base>'} version={args.version} prefix={n}")
    print(f"  vllm ids: {v_ids[:n]}")
    print(f"  hf   ids: {h_ids[:n]}")
    print(f"  vllm text: {tok.decode(v_ids)!r}")
    print(f"  hf   text: {tok.decode(h_ids)!r}")

    if a == b:
        print(f"G3 PASS: first {len(a)} sampled token ids match.")
        return 0

    first = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
    print(f"G3 FAIL: ids diverge at index {first} "
          f"(vllm={a[first] if first < len(a) else None}, "
          f"hf={b[first] if first < len(b) else None}).")
    print("  The trainer and the generator are not the same policy. Do not train on this.")
    print("  Usual causes: stale lora_int_id (reused version), adapter not actually loaded,")
    print("  a retokenized prompt, or a different chat template on one side.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
