"""GRPO: the group is its own baseline.

For one prompt we sample G episodes and score each with one scalar. The advantage is the
rollout's reward standardised against its own siblings:

    advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)

No value network, no critic, no reward model. That deletion is what makes this fit on one
24 GB card in a few hours.

Two memory notes, because a 248k vocab punishes the naive version:
  * we call the inner model and apply `lm_head` ONLY at the positions we train on, so we
    never materialise logits for the ~1000 prompt and image tokens;
  * the loss is chunked over those positions.

We run one inner epoch (mu=1), so the PPO ratio is 1 by construction on the first update.
The clip is still computed — it costs nothing and it keeps the code honest if mu ever rises.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

#: Positions are processed in blocks of this many tokens when applying lm_head.
LOGIT_CHUNK = 256


def group_advantages(rewards: list[float], group_size: int,
                     eps: float = 1e-4) -> tuple[list[float], list[bool]]:
    """Standardise rewards inside each group of `group_size`.

    Returns (advantages, used) where `used` is False for a whole group whose rewards have no
    spread. A zero-variance group carries no information: every sibling agreed, so there is
    nothing to push up relative to anything else. Training on it only adds noise.
    """
    advs: list[float] = [0.0] * len(rewards)
    used: list[bool] = [False] * len(rewards)
    for start in range(0, len(rewards), group_size):
        grp = rewards[start:start + group_size]
        n = len(grp)
        mean = sum(grp) / n
        var = sum((r - mean) ** 2 for r in grp) / n
        std = var ** 0.5
        if std < eps:
            continue
        for j, r in enumerate(grp):
            advs[start + j] = (r - mean) / (std + eps)
            used[start + j] = True
    return advs, used


def inner_parts(model):
    """Unwrap PEFT and return (trunk, lm_head).

    `peft_model.model` is the *base ForConditionalGeneration*, not the trunk, so calling it
    would run the 248k-wide lm_head over the whole sequence. We want the trunk, which returns
    hidden states only. LoRA layers are injected in place, so gradients still flow through
    these modules and `disable_adapter()` still switches them off.
    """
    m = model
    while hasattr(m, "get_base_model"):
        m = m.get_base_model()
    return m.model, m.lm_head


def _forward_selected(model, ids: torch.Tensor, pixel_values, image_grid_thw,
                      positions: torch.Tensor, targets: torch.Tensor,
                      image_token_id: int) -> torch.Tensor:
    """Log-probability of `targets` at `positions`. One sequence at a time (batch of 1).

    `positions[k]` is the index whose hidden state predicts `targets[k]`, i.e. the token
    BEFORE the trainable token.
    """
    attn = torch.ones_like(ids)
    trunk, head = inner_parts(model)
    kw = {}
    if pixel_values is not None:
        # M-RoPE needs to know which positions are image tokens. The processor normally
        # supplies this; we rebuild it exactly, since image tokens are the image_pad ids.
        kw = {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw,
              "mm_token_type_ids": (ids == image_token_id).long()}
    out = trunk(input_ids=ids, attention_mask=attn, use_cache=False, **kw)
    hidden = out.last_hidden_state[0]  # [L, H]
    sel = hidden.index_select(0, positions)  # [T, H]

    parts = []
    for s in range(0, sel.shape[0], LOGIT_CHUNK):
        chunk = sel[s:s + LOGIT_CHUNK]
        logits = head(chunk)
        parts.append(-F.cross_entropy(logits.float(), targets[s:s + LOGIT_CHUNK],
                                      reduction="none"))
    return torch.cat(parts) if parts else sel.new_zeros(0)


def episode_tensors(conv, device) -> dict:
    """Build the single-sequence forward inputs for one finished episode."""
    ids = torch.tensor(conv.ids, dtype=torch.long, device=device).unsqueeze(0)
    train_mask = torch.tensor(conv.trainable, dtype=torch.bool)
    # A token at index i is predicted from the hidden state at i-1.
    idx = train_mask.nonzero(as_tuple=True)[0]
    idx = idx[idx > 0]
    positions = (idx - 1).to(device)
    targets = ids[0, idx.to(device)]
    return {"ids": ids, "positions": positions, "targets": targets, "n_train": int(idx.numel())}


def episode_loss(model, proc, conv, advantage: float, kl_coef: float,
                 clip_eps: float, device, old_logprobs: torch.Tensor | None = None) -> dict:
    """Policy-gradient loss for one episode, plus the KL guard against the frozen model."""
    t = episode_tensors(conv, device)
    if t["n_train"] == 0:
        return {"loss": None, "n_train": 0, "kl": 0.0}

    if conv.images:
        bat = proc.image_processor(images=conv.images, return_tensors="pt")
        pv = bat["pixel_values"].to(device, torch.bfloat16)
        grid = bat["image_grid_thw"].to(device)
    else:
        pv, grid = None, None
    img_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    logp = _forward_selected(model, t["ids"], pv, grid, t["positions"], t["targets"], img_id)

    # Reference = the same weights with the adapter switched off. No second copy in memory.
    kl = torch.zeros((), device=device)
    if kl_coef > 0:
        with torch.no_grad(), model.disable_adapter():
            ref = _forward_selected(model, t["ids"], pv, grid, t["positions"], t["targets"],
                                    img_id)
        # k3 estimator: always non-negative, low variance.
        d = ref - logp
        kl = (d.exp() - d - 1.0).mean()

    if old_logprobs is None:
        old = logp.detach()
    else:
        old = old_logprobs.to(logp.device)
    ratio = (logp - old).exp()
    a = torch.tensor(advantage, device=device, dtype=logp.dtype)
    unclipped = ratio * a
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * a
    pg = -torch.min(unclipped, clipped).mean()

    loss = pg + kl_coef * kl
    return {"loss": loss, "n_train": t["n_train"], "kl": float(kl.detach()),
            "pg": float(pg.detach()), "mean_logp": float(logp.detach().mean())}


def build_lora(model, rank: int = 16, alpha: int = 32, dropout: float = 0.0):
    """LoRA on the LANGUAGE tower only. The vision tower stays frozen (GOAL §7).

    Targeting is a regex anchored on `language_model`, because the vision tower also has
    modules called `out_proj` and a bare suffix match would unfreeze it.
    """
    from peft import LoraConfig, get_peft_model

    pattern = (
        r".*language_model\..*\."
        r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
        r"|in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj)$"
    )
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=pattern, task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, cfg)

    n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    bad = [n for n, p in peft_model.named_parameters()
           if p.requires_grad and ".visual." in n]
    if bad:
        raise RuntimeError(f"vision tower must stay frozen, but {len(bad)} vision params "
                           f"are trainable, e.g. {bad[:3]}")
    if n_train == 0:
        raise RuntimeError("LoRA matched no modules — the target regex is wrong")
    return peft_model, n_train
