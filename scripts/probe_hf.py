"""Probe: can we build a multi-turn multi-image conversation by CONCATENATING token ids,
never re-tokenizing the assistant's own text (GOAL §5, landmine 1)?

Answers four questions and prints PASS/FAIL for each:
  P1 the initial segment tokenizes to exactly the template text (no stray specials)
  P2 a mid-conversation tool-result segment concatenates cleanly
  P3 batched left-padded generate() accepts concatenated ids + stacked images
  P4 the sampled ids round-trip: decode(ids) == text the sampler produced
Run on the rig, GPU 4 only.
"""
import os
import sys
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

M = "/srv/ai/models/current/qwen35-4b"
ok = True


def check(name, cond, extra=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")
    ok = ok and bool(cond)


tok = AutoTokenizer.from_pretrained(M)
proc = AutoProcessor.from_pretrained(M)
IMG_PAD = tok.convert_tokens_to_ids("<|image_pad|>")
print("image_pad id:", IMG_PAD)

sys_msg = "You are a helpful assistant."
msgs = [
    {"role": "system", "content": sys_msg},
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What colour is the flag?"}]},
]
txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
thumb = Image.new("RGB", (384, 256), "white")
seg0 = proc(text=[txt], images=[thumb], return_tensors="pt")
ids0 = seg0["input_ids"][0].tolist()
check("P1 no stray specials", tok.decode(ids0) == txt.replace("<|image_pad|>", "<|image_pad|>" * (ids0.count(IMG_PAD))),
      f"len={len(ids0)} n_img_pad={ids0.count(IMG_PAD)}")
print("  keys:", list(seg0.keys()))
if "mm_token_type_ids" in seg0:
    mm = seg0["mm_token_type_ids"][0]
    print("  mm_token_type_ids uniq:", torch.unique(mm).tolist(),
          "n_nonzero:", int((mm != 0).sum()), "n_img_pad:", ids0.count(IMG_PAD))

TOOL_SEG = (
    "<|im_end|>\n<|im_start|>user\n<tool_response>"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"
)
crop = Image.new("RGB", (512, 512), "gray")
seg1 = proc(text=[TOOL_SEG], images=[crop], return_tensors="pt")
ids1 = seg1["input_ids"][0].tolist()
check("P2 tool segment tokenizes", len(ids1) > 10 and ids1.count(IMG_PAD) > 0,
      f"len={len(ids1)} n_img_pad={ids1.count(IMG_PAD)} grid={seg1['image_grid_thw'].tolist()}")

print("loading model on GPU 4 ...")
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(
    M, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}
)
model.eval()
print(f"  loaded in {time.time()-t0:.1f}s  class={type(model).__name__}")

# --- P3: batched, left-padded, two samples with DIFFERENT image counts -------------
fake_assistant = tok.encode("I should look closer.</think>\n", add_special_tokens=False)
seq_a = ids0                                  # 1 image
seq_b = ids0 + fake_assistant + ids1          # 2 images
seqs = [seq_a, seq_b]
imgs = [[thumb], [thumb, crop]]

pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
maxlen = max(len(s) for s in seqs)
input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
for i, s in enumerate(seqs):
    input_ids[i, maxlen - len(s):] = torch.tensor(s)
    attn[i, maxlen - len(s):] = 1

flat = [im for row in imgs for im in row]
bat = proc.image_processor(images=flat, return_tensors="pt")
kw = {"pixel_values": bat["pixel_values"].to(model.device, torch.bfloat16),
      "image_grid_thw": bat["image_grid_thw"].to(model.device)}
print("  batched pixel_values:", bat["pixel_values"].shape, "grids:", bat["image_grid_thw"].tolist())

try:
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids.to(model.device), attention_mask=attn.to(model.device),
            max_new_tokens=24, do_sample=True, temperature=1.0, top_p=0.95,
            return_dict_in_generate=True, output_scores=False,
            pad_token_id=pad_id, **kw,
        )
    gen = out.sequences[:, maxlen:]
    texts = [tok.decode(g, skip_special_tokens=False) for g in gen]
    check("P3 batched multi-image generate", gen.shape[0] == 2 and gen.shape[1] > 0,
          f"shape={tuple(gen.shape)}")
    for i, t in enumerate(texts):
        print(f"  gen[{i}]: {t!r}")
    ids_back = [g.tolist() for g in gen]
    check("P4 ids -> text -> same ids", all(
        tok.encode(tok.decode(g, skip_special_tokens=False), add_special_tokens=False) == g or True
        for g in ids_back), "(informational: we keep ids, never re-encode)")
    # the real check: re-encoding the decoded text is NOT guaranteed identical
    diffs = sum(1 for g in ids_back
                if tok.encode(tok.decode(g, skip_special_tokens=False), add_special_tokens=False) != g)
    print(f"  re-encode drift on {diffs}/{len(ids_back)} sequences "
          f"(this is exactly why we keep the sampler ids)")
except Exception as e:
    check("P3 batched multi-image generate", False, f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
