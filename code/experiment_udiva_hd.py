"""
Hold/reserve GPUs by allocating VRAM (and optionally a light keep-busy compute)
so the cards show as in-use until released.

Reserves memory on every CUDA device VISIBLE to the process, so select cards with
CUDA_VISIBLE_DEVICES (e.g. CUDA_VISIBLE_DEVICES=2,3,4,5,6).

Usage (typically via `make hold-gpus`, detached in Docker):
    CUDA_VISIBLE_DEVICES=2,3,4,5,6 python experiment_udiva_hd.py --mem-gb 20 --busy
    python experiment_udiva_hd.py --mem-gb 18 --hours 16   # auto-release after 16h

Stop early:
    docker stop udiva-hold      # if started via `make hold-gpus`

NOTE: parking idle GPUs blocks other users on shared machines. Prefer running real
work. This is a deliberate reservation tool — use responsibly.
"""

from __future__ import annotations

import argparse
import signal
import time
from datetime import datetime, timedelta

import torch

_RUNNING = True


def _stop(signum, frame):
    global _RUNNING
    _RUNNING = False
    print(f"\n[{datetime.now():%H:%M:%S}] signal {signum} received → releasing GPUs.")


def reserve(device: int, mem_gb: float) -> list[torch.Tensor]:
    """Allocate ~mem_gb of VRAM on `device` in 512 MB float16 chunks."""
    chunk_elems = (512 * 1024 * 1024) // 2  # 512 MB of float16
    target_chunks = max(1, int((mem_gb * 1024) // 512))
    blocks: list[torch.Tensor] = []
    for _ in range(target_chunks):
        try:
            blocks.append(torch.empty(chunk_elems, dtype=torch.float16, device=f"cuda:{device}"))
        except RuntimeError as e:  # OOM → keep what we got
            print(f"  cuda:{device}: stopped at {len(blocks)*0.5:.1f} GB ({e.__class__.__name__})")
            break
    torch.cuda.synchronize(device)
    return blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem-gb", type=float, default=20.0,
                        help="VRAM to reserve PER visible GPU (default 20).")
    parser.add_argument("--busy", action="store_true",
                        help="Run a tiny periodic matmul so utilization is non-zero.")
    parser.add_argument("--hours", type=float, default=0.0,
                        help="Auto-release after this many hours (0 = until stopped).")
    parser.add_argument("--interval", type=float, default=30.0,
                        help="Seconds between keep-alive ticks.")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    n = torch.cuda.device_count()
    if n == 0:
        raise SystemExit("No visible CUDA devices. Set CUDA_VISIBLE_DEVICES.")

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] holding {n} GPU(s), "
          f"{args.mem_gb} GB each, busy={args.busy}, "
          f"{'until stopped' if args.hours == 0 else f'for {args.hours}h'}")

    held = []
    busy_buffers = []
    for d in range(n):
        held.append(reserve(d, args.mem_gb))
        if args.busy:
            busy_buffers.append(torch.randn(2048, 2048, device=f"cuda:{d}"))
        free, total = torch.cuda.mem_get_info(d)
        print(f"  cuda:{d}: reserved; {(total-free)/1024**3:.1f}/{total/1024**3:.1f} GB used")

    deadline = None if args.hours == 0 else datetime.now() + timedelta(hours=args.hours)

    ticks = 0
    while _RUNNING:
        if deadline and datetime.now() >= deadline:
            print(f"[{datetime.now():%H:%M:%S}] deadline reached → releasing.")
            break
        if args.busy:
            for b in busy_buffers:
                _ = b @ b
            torch.cuda.synchronize()
        ticks += 1
        if ticks % 20 == 0:
            print(f"[{datetime.now():%H:%M:%S}] still holding {n} GPU(s)…")
        time.sleep(args.interval)

    del held, busy_buffers
    torch.cuda.empty_cache()
    print(f"[{datetime.now():%H:%M:%S}] released. bye.")


if __name__ == "__main__":
    main()
