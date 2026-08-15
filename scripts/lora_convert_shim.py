#!/usr/bin/env python3
"""Run llama.cpp's convert_lora_to_gguf.py with one correctness patch.

WHY THIS EXISTS
---------------
Qwen3.5 is a hybrid: some blocks are ordinary attention, others are DeltaNet linear
attention. llama.cpp's converter reorders the V heads of the linear-attention blocks from
HF's grouped layout to ggml's tiled layout (`_LinearAttentionVReorderBase._reorder_v_heads`).

For `linear_attn.out_proj` that reorder runs along **dim 1 — the INPUT columns**. A LoRA is
stored factored, W = B @ A, and `LoraTorchTensor` only knows how to reshape the OUTPUT
dims; reshaping the input dim raises `NotImplementedError: can't reshape the row size
trivially`. So any adapter that touches `out_proj` cannot be converted at all.

THE FIX, AND WHY IT IS CORRECT
------------------------------
Permuting the columns of W is exactly permuting the columns of A:

    (B @ A)[:, p] == B @ (A[:, p])

and permuting the rows of W is exactly permuting the rows of B. So the reorder is applied
to whichever factor owns the axis being permuted, and the other factor is untouched. No
approximation.

Everything else runs unmodified. The upstream file is not edited.

Usage: lora_convert_shim.py <lora_dir> <base_config_dir> <outfile>
"""
import os
import runpy
import sys

LLAMA = os.path.expanduser("~/code/forks/llama.cpp")


def main():
    lora_dir, base_dir, outfile = sys.argv[1:4]
    sys.path.insert(0, LLAMA)

    import conversion.qwen as Q

    cls = Q._LinearAttentionVReorderBase
    original = cls._reorder_v_heads.__func__ if hasattr(
        cls._reorder_v_heads, "__func__") else cls._reorder_v_heads

    def patched(tensor, dim, num_k_heads, num_v_per_k, head_dim):
        # Duck-typed: LoraTorchTensor is defined inside convert_lora_to_gguf.py, which is
        # not importable without running it, so detect the factored form by its fields.
        if hasattr(tensor, "_lora_A") and hasattr(tensor, "_lora_B"):
            ndim = len(tensor.shape)
            d = dim + ndim if dim < 0 else dim
            klass = type(tensor)
            if d == ndim - 1:
                # Input-column reorder -> belongs to A, whose last axis is in_features.
                new_a = original(tensor._lora_A, -1, num_k_heads, num_v_per_k, head_dim)
                return klass(new_a, tensor._lora_B)
            # Output-row reorder -> belongs to B, whose leading axes are out_features.
            new_b = original(tensor._lora_B, d, num_k_heads, num_v_per_k, head_dim)
            return klass(tensor._lora_A, new_b)
        return original(tensor, dim, num_k_heads, num_v_per_k, head_dim)

    cls._reorder_v_heads = staticmethod(patched)
    print("[shim] patched _LinearAttentionVReorderBase._reorder_v_heads for LoRA tensors",
          flush=True)

    sys.argv = ["convert_lora_to_gguf.py", lora_dir, "--base", base_dir,
                "--outtype", "f16", "--outfile", outfile]
    runpy.run_path(os.path.join(LLAMA, "convert_lora_to_gguf.py"), run_name="__main__")


if __name__ == "__main__":
    main()
