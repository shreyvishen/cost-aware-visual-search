"""One training run. A, B, C and D are this file with a different config.

Duration is a WALL-CLOCK TIMER, never a step count (GOAL §14). A and B each get exactly the
same number of minutes so their curves are comparable. When the timer fires we finish the
current step, checkpoint, run the full eval, write DONE, and stop.

Fail-safe rules this file implements (GOAL §17):
  * checkpoint every N steps, written to a temp dir then fsynced and renamed
  * every log line flushed and fsynced, so a kill -9 loses nothing already written
  * SIGTERM/SIGINT checkpoint before exit; any exception checkpoints before re-raising
  * --resume picks up step, lambda, RNG and data position
  * a DONE marker makes a completed run idempotent
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src import data as data_mod  # noqa: E402
from src import grpo, rollout  # noqa: E402
from src.contract import Episode  # noqa: E402
from src.reward import CostModel, answer_correct, compute_reward  # noqa: E402

MIN_FREE_GB = 10.0


# --- small helpers ------------------------------------------------------------------


class Log:
    """Append-only, flushed and fsynced after every line."""

    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "a", buffering=1)

    def write(self, obj) -> None:
        self.f.write(json.dumps(obj) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def atomic_save(fn, dest: Path) -> None:
    """Write through a temp sibling, fsync, then rename. Never a half-written checkpoint."""
    dest = Path(dest)
    tmp = dest.parent / (dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    fn(tmp)
    for root, _dirs, files in os.walk(tmp):
        for name in files:
            fd = os.open(os.path.join(root, name), os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    os.rename(tmp, dest)


def free_gb(path: str) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- episode scoring ----------------------------------------------------------------


def score_episodes(eps: list[Episode], cost: CostModel, lam: float,
                   options_by_sid: dict | None = None) -> None:
    for ep in eps:
        opts = (options_by_sid or {}).get(ep.sid)
        ep.correct = answer_correct(ep.answer or "", ep.gold, opts)
        ep.cost_ms = cost.cost_ms(ep.vision_tokens, ep.decode_tokens, ep.n_tool_calls)
        ep.reward = compute_reward(ep.correct, ep.cost_ms, lam)


def summarize(eps: list[Episode]) -> dict:
    n = max(1, len(eps))
    zooms = [e.n_tool_calls for e in eps]
    return {
        "n": len(eps),
        "accuracy": sum(e.correct for e in eps) / n,
        "tool_rate": sum(1 for e in eps if e.n_tool_calls > 0) / n,
        "mean_zooms": sum(zooms) / n,
        "zoom_hist": {str(k): zooms.count(k) for k in sorted(set(zooms))},
        "mean_cost_ms": sum(e.cost_ms for e in eps) / n,
        "mean_vision_tokens": sum(e.vision_tokens for e in eps) / n,
        "mean_decode_tokens": sum(e.decode_tokens for e in eps) / n,
        "invalid_format_rate": sum(e.invalid_format for e in eps) / n,
        # How much we lean on the prose fallback. Reported, never hidden.
        "answer_source": {k: sum(1 for e in eps if e.answer_source == k)
                          for k in ("tagged", "unclosed", "prose", "none")},
        "mean_reward": sum(e.reward for e in eps) / n,
        # Box validity: the >=90% gate from TASKS Phase 1. If this falls, the policy is
        # zooming at nothing and the environment is silently broken.
        "boxes_emitted": sum(getattr(e, "n_boxes_emitted", 0) for e in eps),
        "box_in_frame_rate": (
            sum(getattr(e, "n_boxes_in_frame", 0) for e in eps)
            / max(1, sum(getattr(e, "n_boxes_emitted", 0) for e in eps))),
    }


# --- the run ------------------------------------------------------------------------


class Run:
    def __init__(self, cfg: dict, out_dir: Path, resume: bool):
        self.cfg = cfg
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        for sub in ("adapters", "rollouts", "crops", "eval"):
            (self.dir / sub).mkdir(exist_ok=True)
        self.metrics = Log(self.dir / "metrics.jsonl")
        self.step = 0
        self.lam = 0.0
        self.data_pos = 0
        self.best_score = -1e9
        self.resume = resume
        self.stop_requested = False
        self.t_start = time.perf_counter()

    # -- setup --

    def build(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

        cfg = self.cfg
        set_seed(cfg["seed"])
        mp = cfg["model_path"]
        say(f"loading {mp}")
        self.tok = AutoTokenizer.from_pretrained(mp)
        self.proc = AutoProcessor.from_pretrained(mp)
        self.device = cfg.get("train_device", "cuda:0")
        base = AutoModelForImageTextToText.from_pretrained(
            mp, dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map={"": self.device})
        self.model, n_train = grpo.build_lora(base, rank=cfg["lora_rank"],
                                              alpha=cfg["lora_rank"] * 2)
        say(f"LoRA rank {cfg['lora_rank']}: {n_train/1e6:.2f}M trainable params")
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        self.model.config.use_cache = False

        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=0.0)

        # Extra GPUs generate. Generation is ~80% of a step, so replicas convert idle cards
        # straight into gradient steps. They are inference copies of the same PEFT model; the
        # LoRA tensors are copied across before every round, so they never serve stale weights.
        replicas = []
        for dev in cfg.get("gen_replica_devices", []):
            say(f"building generation replica on {dev}")
            rbase = AutoModelForImageTextToText.from_pretrained(
                mp, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": dev})
            rmodel, _ = grpo.build_lora(rbase, rank=cfg["lora_rank"],
                                        alpha=cfg["lora_rank"] * 2)
            rmodel.eval()
            replicas.append((rmodel, dev))

        from src.backends.hf_backend import HFBackend
        self.backend = HFBackend(self.model, self.proc, self.tok, device=self.device,
                                 replicas=replicas,
                                 micro_batch=cfg.get("gen_micro_batch", 16))
        if replicas:
            self.backend.load_adapter("", 0)  # first sync

        self.cost = CostModel.from_config(cfg, str(REPO))
        say(f"cost model: {self.cost.to_json()}")

        # data
        pool = data_mod.load_train(limit=cfg["train_pool"], seed=cfg["seed"])
        if cfg.get("pool_filter") == "non_binary":
            before = len(pool)
            pool = [s for s in pool if data_mod.answer_shape(s.gold) != "yes_no"]
            say(f"non-binary filter: {before} -> {len(pool)} samples")
        if cfg.get("balance_polarity", True) and cfg.get("pool_filter") != "non_binary":
            before = len(pool)
            pool = data_mod.balance_polarity(pool, seed=cfg["seed"])
            say(f"polarity balanced: {before} -> {len(pool)} samples")
        self.pool = pool
        self.eval_set = data_mod.load_vstar(limit=cfg["eval_n"])
        self.eval_full = None
        self.eval_options = {s.sid: s.options for s in self.eval_set}
        say(f"train pool {len(self.pool)} · eval subset {len(self.eval_set)}")

        # lambda comes from the frozen coefficients, once (GOAL §4)
        self.lam_target = self.cost.lam_for(cfg["mean_crop_vision_tokens"])
        say(f"lambda target {self.lam_target:.6g} "
            f"(one extra zoom ~= {self.lam_target * (self.cost.a * cfg['mean_crop_vision_tokens'] + self.cost.c):.3f} reward)")

        if self.resume:
            self.load_checkpoint()
        self.write_config()

    def write_config(self) -> None:
        payload = dict(self.cfg)
        payload["cost_model"] = self.cost.to_json()
        payload["lam_target"] = self.lam_target
        payload["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        (self.dir / "config.json").write_text(json.dumps(payload, indent=2))

    # -- checkpointing --

    def save_checkpoint(self, tag: str) -> None:
        if free_gb(str(self.dir)) < MIN_FREE_GB:
            say(f"REFUSING to checkpoint: only {free_gb(str(self.dir)):.1f} GB free")
            return
        dest = self.dir / "adapters" / tag

        def _write(tmp: Path):
            self.model.save_pretrained(str(tmp))
            torch.save({
                "step": self.step,
                "lam": self.lam,
                "data_pos": self.data_pos,
                "best_score": self.best_score,
                "opt": self.opt.state_dict(),
                "rng_torch": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all(),
                "rng_numpy": np.random.get_state(),
                "rng_python": random.getstate(),
            }, tmp / "trainer_state.pt")

        atomic_save(_write, dest)
        say(f"checkpoint -> {dest}")

    def load_checkpoint(self) -> None:
        last = self.dir / "adapters" / "last"
        state = last / "trainer_state.pt"
        if not state.exists():
            say("no checkpoint to resume from, starting fresh")
            return
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        w = last / "adapter_model.safetensors"
        if w.exists():
            set_peft_model_state_dict(self.model, load_file(str(w)))
        st = torch.load(state, map_location="cpu", weights_only=False)
        self.step = st["step"]
        self.lam = st["lam"]
        self.data_pos = st["data_pos"]
        self.best_score = st.get("best_score", -1e9)
        self.opt.load_state_dict(st["opt"])
        torch.set_rng_state(st["rng_torch"])
        random.setstate(st["rng_python"])
        np.random.set_state(st["rng_numpy"])
        say(f"resumed at step {self.step}, lam {self.lam:.6g}, data_pos {self.data_pos}")

    # -- one step --

    def next_batch(self):
        k = self.cfg["prompts_per_step"]
        out = []
        for _ in range(k):
            out.append(self.pool[self.data_pos % len(self.pool)])
            self.data_pos += 1
        return out

    def train_step(self) -> dict:
        cfg = self.cfg
        t0 = time.perf_counter()
        samples = self.next_batch()

        self.model.eval()
        self.backend.load_adapter("", self.step)  # push LoRA to the replicas
        eps = rollout.collect(self.backend, self.proc, self.tok, samples,
                              cfg["group_size"], cfg)
        t_gen = time.perf_counter() - t0

        score_episodes(eps, self.cost, self.lam)
        rewards = [e.reward for e in eps]
        advs, used = grpo.group_advantages(rewards, cfg["group_size"])

        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        n_used = sum(used)
        stats = {"kl": 0.0, "pg": 0.0, "n_tok": 0}
        if n_used:
            for ep, adv, ok in zip(eps, advs, used):
                if not ok:
                    continue
                res = grpo.episode_loss(self.model, self.proc, ep.meta_conv, adv,
                                        cfg["kl_coef"], cfg["clip_eps"], self.device)
                if res["loss"] is None:
                    continue
                (res["loss"] / n_used).backward()
                stats["kl"] += res["kl"] / n_used
                stats["pg"] += res["pg"] / n_used
                stats["n_tok"] += res["n_train"]
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], cfg["max_grad_norm"])
            stats["grad_norm"] = float(gn)
            self.opt.step()
        self.opt.zero_grad(set_to_none=True)

        row = summarize(eps)
        row.update({
            "step": self.step, "phase": "train", "lam": self.lam,
            "groups_used": n_used // cfg["group_size"],
            "groups_total": len(eps) // cfg["group_size"],
            "reward_std": float(np.std(rewards)), "t_gen_s": round(t_gen, 1),
            "t_step_s": round(time.perf_counter() - t0, 1),
            "elapsed_min": round((time.perf_counter() - self.t_start) / 60, 2),
            **{k: v for k, v in stats.items() if k != "n_tok"},
            "train_tokens": stats["n_tok"],
        })
        self.metrics.write(row)
        self.dump_rollouts(eps)
        return row

    def dump_rollouts(self, eps: list[Episode]) -> None:
        keep = self.cfg.get("rollout_dump", 8)
        path = self.dir / "rollouts" / f"step_{self.step:04d}.jsonl"
        with open(path, "w") as f:
            for ep in eps[:keep]:
                d = ep.to_json()
                d["boxes"] = getattr(ep, "meta_boxes", [])
                d["box_info"] = getattr(ep, "meta_box_info", [])
                f.write(json.dumps(d) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # -- eval --

    @torch.no_grad()
    def evaluate(self, tag: str, samples=None, max_zooms=None) -> dict:
        cfg = dict(self.cfg)
        if max_zooms is not None:
            cfg["max_zooms"] = max_zooms
        samples = samples or self.eval_set
        self.model.eval()
        t0 = time.perf_counter()
        eps = rollout.collect(self.backend, self.proc, self.tok, samples, 1, cfg,
                              temperature=0.0)
        score_episodes(eps, self.cost, self.lam, self.eval_options)
        row = summarize(eps)
        row.update({"step": self.step, "phase": f"eval:{tag}", "lam": self.lam,
                    "t_eval_s": round(time.perf_counter() - t0, 1),
                    "elapsed_min": round((time.perf_counter() - self.t_start) / 60, 2)})
        self.metrics.write(row)
        say(f"eval[{tag}] acc={row['accuracy']:.3f} tool_rate={row['tool_rate']:.3f} "
            f"mean_zooms={row['mean_zooms']:.2f} cost_ms={row['mean_cost_ms']:.1f}")
        with open(self.dir / "eval" / f"vstar_predictions_{tag}.jsonl", "w") as f:
            for ep in eps:
                d = ep.to_json()
                d["boxes"] = getattr(ep, "meta_boxes", [])
                f.write(json.dumps(d) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.save_crops(eps, tag)
        return row

    def save_crops(self, eps: list[Episode], tag: str) -> None:
        """The 'where it looked' picture: thumbnail with its boxes, plus the crops."""
        from src.zoom_env import draw_boxes
        n = self.cfg.get("crop_dump", 6)
        out = self.dir / "crops" / tag
        out.mkdir(parents=True, exist_ok=True)
        for ep in eps[:n]:
            thumb = getattr(ep, "meta_thumb", None)
            boxes = getattr(ep, "meta_boxes", [])
            if thumb is None:
                continue
            draw_boxes(thumb, boxes).save(out / f"{ep.sid}_thumb.jpg", quality=88)
            for i, c in enumerate(getattr(ep, "meta_crops", [])[:4]):
                c.save(out / f"{ep.sid}_crop{i}.jpg", quality=88)

    # -- loop --

    def loop(self) -> None:
        cfg = self.cfg
        budget_s = cfg["minutes"] * 60
        warm = max(1, int(cfg["lambda_ramp_frac"] * cfg["expected_steps"]))
        say(f"training for {cfg['minutes']} min · lambda ramps over ~{warm} steps")

        self.evaluate("step0")
        while True:
            elapsed = time.perf_counter() - self.t_start
            if elapsed >= budget_s:
                say(f"timer fired at {elapsed/60:.1f} min after {self.step} steps")
                break
            if self.stop_requested:
                say("stop requested, finishing gracefully")
                break

            self.lam = self.lam_target * min(1.0, (self.step + 1) / warm)
            row = self.train_step()
            self.step += 1
            say(f"step {self.step} acc={row['accuracy']:.3f} tool={row['tool_rate']:.3f} "
                f"zooms={row['mean_zooms']:.2f} R={row['mean_reward']:.3f} "
                f"used={row['groups_used']}/{row['groups_total']} "
                f"{row['t_step_s']:.0f}s ({row['elapsed_min']:.1f}/{cfg['minutes']} min)")

            if not np.isfinite(row["mean_reward"]):
                say("NON-FINITE REWARD — aborting this run and keeping the checkpoint")
                self.save_checkpoint("last")
                break

            if self.step % cfg["checkpoint_every"] == 0:
                self.save_checkpoint("last")
            if self.step % cfg["eval_every"] == 0:
                ev = self.evaluate(f"step{self.step}")
                sc = ev["accuracy"] - 0.001 * ev["mean_zooms"]
                if sc > self.best_score:
                    self.best_score = sc
                    self.save_checkpoint("best")

        self.finish()

    def finish(self) -> None:
        say("final checkpoint + full eval")
        self.save_checkpoint("last")
        if self.best_score <= -1e9:
            self.save_checkpoint("best")
        full = data_mod.load_vstar()
        self.eval_options = {s.sid: s.options for s in full}
        self.evaluate("final", samples=full)
        # Baselines that make the A-vs-B claim readable (GOAL §12).
        # The thumbnail-only baseline. A subset is enough — it is a reference line, not the
        # headline number, and the full set would cost minutes we owe to training.
        self.evaluate("never_zoom", samples=full[: self.cfg.get("never_zoom_n", 96)],
                      max_zooms=0)
        (self.dir / "DONE").write_text(
            json.dumps({"step": self.step,
                        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "minutes": self.cfg["minutes"]}, indent=2))
        self.metrics.close()
        say("DONE")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None, help="smoke test: stop after N steps")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    if args.minutes is not None:
        cfg["minutes"] = args.minutes
    out = Path(args.out)
    if (out / "DONE").exists():
        say(f"{out}/DONE exists — run already complete, nothing to do")
        return 0
    if free_gb(str(out.parent)) < MIN_FREE_GB:
        say(f"PREFLIGHT FAIL: {free_gb(str(out.parent)):.1f} GB free, need {MIN_FREE_GB}")
        return 2

    run = Run(cfg, out, resume=args.resume)

    def on_signal(signum, _frame):
        say(f"signal {signum} — checkpointing before exit")
        run.stop_requested = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    try:
        run.build()
        if args.steps:
            cfg["minutes"] = 1e9
            run.evaluate("step0")
            for _ in range(args.steps):
                run.lam = run.lam_target
                run.train_step()
                run.step += 1
            run.save_checkpoint("last")
            say(f"smoke: {args.steps} step(s) complete")
            return 0
        run.loop()
    except BaseException as e:  # noqa: BLE001 — checkpoint on ANY exit path
        say(f"EXCEPTION {type(e).__name__}: {e}")
        try:
            run.save_checkpoint("crash")
        except Exception as e2:
            say(f"  checkpoint on crash also failed: {e2}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
