#!/usr/bin/env python3
"""Pre-warm a llama-server for the demo page, then hold it.

    python3 app/warm.py --model base --quant q4

Loading a model costs 10-40 s (bf16 is 8.4 GB and the slow end of that). The page must
not pay it on the first click, so run this first and leave it running: the process holds
one warm server until you Ctrl-C it.

Two servers on one Metal GPU corrupt each other's timings, so this waits for any other
llama-server — the benchmark queue, most likely — to finish before it starts. `--now`
skips the wait.

NOTE: this holds a server in ITS OWN process. `app/server.py` holds its own in-process
server via `infer.warmup()`. Running both at once is exactly the contention this warns
about — use this to pre-load the weights into the OS page cache before the web server
starts, or use `--in-process` from the web server instead.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import infer  # noqa: E402


def wait_for_exclusive(timeout_s: float) -> bool:
    """Block until no other llama-server is running. True if we got exclusivity."""
    deadline = time.time() + timeout_s
    warned = False
    while time.time() < deadline:
        other = infer.bench_queue_busy()
        if not other:
            return True
        if not warned:
            print(f"[warm] waiting for another llama-server to finish -> {other}",
                  flush=True)
            warned = True
        time.sleep(10)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="base", choices=sorted(infer.MODELS))
    ap.add_argument("--quant", default="q4", choices=list(infer.QUANTS))
    ap.add_argument("--now", action="store_true",
                    help="start even if another llama-server is running")
    ap.add_argument("--wait-timeout", type=float, default=3600.0,
                    help="seconds to wait for exclusivity before giving up")
    ap.add_argument("--hold", action="store_true", default=True,
                    help="keep the server alive until Ctrl-C (default)")
    ap.add_argument("--no-hold", dest="hold", action="store_false",
                    help="load, warm, then stop — just fills the OS page cache")
    args = ap.parse_args()

    if not args.now and not wait_for_exclusive(args.wait_timeout):
        print("[warm] gave up waiting for exclusivity; rerun with --now to force",
              file=sys.stderr)
        return 2

    t0 = time.time()
    try:
        info = infer.warmup(args.model, args.quant)
    except Exception as exc:
        print(f"[warm] FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if info["error"]:
        # The weights are loaded even if the throwaway episode was odd. Say so and stay.
        print(f"[warm] loaded, but the warm-up episode reported: {info['error']}")
    print(f"[warm] {args.model}/{args.quant} ready on :{info['port']} "
          f"in {time.time() - t0:.1f}s")

    if not args.hold:
        infer.shutdown()
        return 0

    print("[warm] holding the server. Ctrl-C to stop it.", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[warm] stopping")
    finally:
        infer.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
