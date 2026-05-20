#!/usr/bin/env python3
"""Clean stale AI2-THOR simulator processes on Linux/WSL.

By default this only targets Unity simulator backends whose command line
contains ``thor-Linux64``. Pass ``--include-training`` when the Python trainer
itself is stuck and should be stopped too.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple


Proc = Tuple[int, str, str]


def iter_processes() -> Iterable[Proc]:
    proc_root = Path("/proc")
    if not proc_root.exists():
        raise RuntimeError("This cleanup script expects a Linux/WSL /proc filesystem.")

    self_pid = os.getpid()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == self_pid:
            continue
        try:
            cmdline = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
                .strip()
            )
            comm = (entry / "comm").read_text(errors="replace").strip()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        yield pid, comm, cmdline


def matching_processes(include_training: bool) -> List[Proc]:
    matches: List[Proc] = []
    for pid, comm, cmdline in iter_processes():
        haystack = f"{comm} {cmdline}"
        if "thor-Linux64" in haystack:
            matches.append((pid, "ai2thor", cmdline or comm))
        elif include_training and "scripts/train_method1.py" in haystack:
            matches.append((pid, "trainer", cmdline or comm))
    return matches


def signal_processes(procs: List[Proc], sig: int) -> None:
    for pid, _kind, _cmd in procs:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            print(f"[cleanup] permission denied for pid {pid}: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean stale AI2-THOR processes")
    parser.add_argument(
        "--include-training",
        action="store_true",
        help="Also stop scripts/train_method1.py processes.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only list matching processes.")
    parser.add_argument("--grace", type=float, default=3.0, help="Seconds before SIGKILL.")
    args = parser.parse_args()

    procs = matching_processes(args.include_training)
    if not procs:
        print("[cleanup] no matching AI2-THOR processes found")
        return 0

    print("[cleanup] matching processes:")
    for pid, kind, cmd in procs:
        print(f"  {pid:>7}  {kind:<8}  {cmd}")

    if args.dry_run:
        return 0

    signal_processes(procs, signal.SIGTERM)
    time.sleep(max(0.0, args.grace))

    remaining_pids = {pid for pid, _kind, _cmd in matching_processes(args.include_training)}
    remaining = [proc for proc in procs if proc[0] in remaining_pids]
    if remaining:
        signal_processes(remaining, signal.SIGKILL)
        time.sleep(0.5)

    final = matching_processes(args.include_training)
    if final:
        print("[cleanup] warning: some matching processes are still present:")
        for pid, kind, cmd in final:
            print(f"  {pid:>7}  {kind:<8}  {cmd}")
        return 1

    print("[cleanup] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
