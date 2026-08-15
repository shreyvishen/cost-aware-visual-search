"""Generation backends. Both expose the same three methods so the training loop does not
care which one is live: load_adapter(path, version), generate(requests, ...), shutdown()."""

DEFAULT_MODEL = "/srv/ai/models/current/qwen35-4b"


def get_backend(name: str, model=None, **kw):
    """`model` may be an already-loaded module (the training loop passes the live one) or a
    path string (the gates pass nothing and just want the base model loaded for them)."""
    if name == "hf":
        from .hf_backend import HFBackend
        if model is None or isinstance(model, str):
            model, proc, tok = _load_hf(model or DEFAULT_MODEL, kw.pop("device", "cuda:0"))
            kw.setdefault("proc", proc)
            kw.setdefault("tok", tok)
        return HFBackend(model=model, **kw)
    if name == "vllm":
        from .vllm_backend import VLLMBackend
        return VLLMBackend(model_path=model or DEFAULT_MODEL, **kw)
    raise ValueError(f"unknown backend {name!r}")


def _load_hf(path: str, device: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    proc = AutoProcessor.from_pretrained(path)
    m = AutoModelForImageTextToText.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device})
    m.eval()
    return m, proc, tok
