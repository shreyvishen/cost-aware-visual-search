"""HF `generate` rollout backend — the guaranteed path.

It generates with the *same* module object the trainer updates. So the generating weights are
the training weights by construction: Gate 3 (vLLM/trainer sync) cannot fail because there is
nothing to sync. That is the whole reason this is the default. It is slower than vLLM; it is
never wrong.

Optionally replicates the model onto extra GPUs for generation only (`replicas=[2, 3]`).
Replicas are inference copies; before each round of generation the LoRA weights are copied
across, which is tens of MB.
"""
from __future__ import annotations

import threading

import torch

#: Sequences that end an assistant turn. Ids resolved once at construction.
STOP_STRINGS = ["</tool_call>", "</answer>", "<|im_end|>"]


def _find_subseq(hay: list[int], needle: list[int], start: int = 0) -> int:
    n = len(needle)
    if n == 0:
        return -1
    for i in range(start, len(hay) - n + 1):
        if hay[i:i + n] == needle:
            return i
    return -1


class HFBackend:
    def __init__(self, model, proc, tok, device="cuda:0", replicas=None, max_images: int = 6,
                 micro_batch: int = 16):
        self.model = model
        self.proc = proc  # gates reach through for the processor/tokenizer
        self.tok = tok
        self.device = device
        self.max_images = max_images
        self.micro_batch = micro_batch
        self.replicas: list = replicas or []
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.stop_id_seqs = [tok.encode(s, add_special_tokens=False) for s in STOP_STRINGS]
        if tok.eos_token_id is not None:
            self.stop_id_seqs.append([tok.eos_token_id])

    # --- interface parity with VLLMBackend ------------------------------------

    def load_adapter(self, adapter_path: str, version: int) -> None:
        """No-op: this backend generates with the live training module. Kept so the training
        loop can call the same method against either backend."""
        self._sync_replicas()

    def shutdown(self) -> None:
        self.replicas = []

    # --- generation -----------------------------------------------------------

    @torch.no_grad()
    def generate(self, requests: list[dict], max_new_tokens: int, temperature: float,
                 stop: list[str] | None = None) -> list[dict]:
        if not requests:
            return []
        if len(self.replicas) > 0:
            return self._generate_sharded(requests, max_new_tokens, temperature)
        return self._generate_on(self.model, self.device, requests, max_new_tokens, temperature)

    def _generate_sharded(self, requests, max_new_tokens, temperature):
        """Split the batch across the live model plus its replicas. HF generate spends its
        time in CUDA kernels, which release the GIL, so plain threads overlap properly."""
        workers = [(self.model, self.device)] + list(self.replicas)
        k = len(workers)
        shards: list[list[tuple[int, dict]]] = [[] for _ in range(k)]
        for i, r in enumerate(requests):
            shards[i % k].append((i, r))

        out: list[dict | None] = [None] * len(requests)
        errors: list[BaseException] = []

        def run(w_idx):
            model, dev = workers[w_idx]
            items = shards[w_idx]
            if not items:
                return
            try:
                res = self._generate_on(model, dev, [r for _, r in items],
                                        max_new_tokens, temperature)
                for (orig_i, _), got in zip(items, res):
                    out[orig_i] = got
            except BaseException as e:  # noqa: BLE001 - re-raised on the main thread
                errors.append(e)

        threads = [threading.Thread(target=run, args=(j,)) for j in range(k)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return [o for o in out if o is not None]

    @torch.no_grad()
    def _generate_on(self, model, device, requests, max_new_tokens, temperature):
        """Chunked so one worker never holds more than `micro_batch` sequences of KV cache."""
        if len(requests) > self.micro_batch:
            out = []
            for s in range(0, len(requests), self.micro_batch):
                out.extend(self._generate_on(model, device,
                                             requests[s:s + self.micro_batch],
                                             max_new_tokens, temperature))
            return out
        # The training loop always supplies ids (that is the point). The gates supply text,
        # which we tokenize here — a first turn has nothing to re-tokenize, so this is safe.
        seqs = []
        for r in requests:
            ids = r.get("prompt_token_ids")
            if ids is None:
                enc = self.proc(text=[r["prompt"]], images=r.get("images") or None,
                                return_tensors="pt")
                ids = enc["input_ids"][0].tolist()
            seqs.append(ids)
        maxlen = max(len(s) for s in seqs)
        bsz = len(seqs)

        input_ids = torch.full((bsz, maxlen), self.pad_id, dtype=torch.long)
        attn = torch.zeros((bsz, maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):  # left pad, so every row ends at the same column
            input_ids[i, maxlen - len(s):] = torch.tensor(s, dtype=torch.long)
            attn[i, maxlen - len(s):] = 1

        flat = [im for r in requests for im in r["images"]]
        kw = {}
        if flat:
            bat = self.proc.image_processor(images=flat, return_tensors="pt")
            kw["pixel_values"] = bat["pixel_values"].to(device, torch.bfloat16)
            kw["image_grid_thw"] = bat["image_grid_thw"].to(device)

        gen = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attn.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.95 if temperature > 0 else None,
            pad_token_id=self.pad_id,
            return_dict_in_generate=True,
            use_cache=True,
            **kw,
        )
        raw = gen.sequences[:, maxlen:].tolist()

        results = []
        for ids in raw:
            ids, reason = self._truncate(ids)
            results.append({
                "token_ids": ids,
                "text": self.tok.decode(ids, skip_special_tokens=False),
                "logprobs": None,
                "finish_reason": reason,
            })
        return results

    def _truncate(self, ids: list[int]) -> tuple[list[int], str]:
        """Cut at the earliest stop sequence, keeping the stop tokens themselves so the next
        segment concatenates onto a well-formed turn."""
        best, reason = None, "length"
        for seq in self.stop_id_seqs:
            j = _find_subseq(ids, seq)
            if j >= 0 and (best is None or j + len(seq) < best):
                best, reason = j + len(seq), "stop"
        if best is not None:
            return ids[:best], reason
        # strip right padding a finished row may carry
        while ids and ids[-1] == self.pad_id:
            ids.pop()
        return ids, reason

    # --- replicas -------------------------------------------------------------

    def _sync_replicas(self) -> None:
        """Copy the trainable (LoRA) tensors from the live model onto each replica."""
        if not self.replicas:
            return
        src = {k: v for k, v in self.model.state_dict().items() if "lora_" in k}
        if not src:
            return
        for model, dev in self.replicas:
            tgt = model.state_dict()
            with torch.no_grad():
                for k, v in src.items():
                    if k in tgt:
                        tgt[k].copy_(v.to(dev, non_blocking=True))
