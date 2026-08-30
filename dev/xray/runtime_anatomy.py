#!/usr/bin/env python3
"""Non-invasive runtime telemetry for llm.c experiments.

This tool intentionally does not modify model math or training behavior. It launches
an llm.c executable, mirrors its output, samples NVIDIA GPU telemetry with
nvidia-smi, and summarizes the runtime anatomy after the child exits.

Example:
    python3 dev/xray/runtime_anatomy.py -- ./train_gpt2fp32cu -b 4 -t 512 -v 1000 -s 1000
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field


ALLOC_RE = re.compile(r"allocated\s+(\d+)\s+MiB\s+for\s+(.+)$")
STEP_RE = re.compile(
    r"step\s+(\d+)/(\d+):.*?\(([0-9.]+)\s+ms,\s+([0-9.]+)\s+tok/s\)"
)
AVG_RE = re.compile(r"total average iteration time:\s*([0-9.]+)\s*ms")


@dataclass
class GpuSample:
    ts: float
    used_mib: int
    total_mib: int
    util_pct: int
    temp_c: int


@dataclass
class RunData:
    allocations: dict[str, int] = field(default_factory=dict)
    step_ms: list[float] = field(default_factory=list)
    step_tok_s: list[float] = field(default_factory=list)
    reported_avg_ms: float | None = None
    gpu_samples: list[GpuSample] = field(default_factory=list)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def sample_gpu(stop: threading.Event, interval_s: float, out: list[GpuSample]) -> None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    while not stop.is_set():
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=2, check=False)
            if r.returncode == 0 and r.stdout.strip():
                # Single-GPU experiment: consume the first line only.
                fields = [x.strip() for x in r.stdout.splitlines()[0].split(",")]
                if len(fields) >= 4:
                    out.append(
                        GpuSample(
                            ts=time.monotonic(),
                            used_mib=int(fields[0]),
                            total_mib=int(fields[1]),
                            util_pct=int(fields[2]),
                            temp_c=int(fields[3]),
                        )
                    )
        except (OSError, subprocess.SubprocessError, ValueError):
            # Telemetry is observational only; never fail the training run because
            # nvidia-smi is unavailable or one sample is malformed.
            pass
        stop.wait(interval_s)


def parse_line(line: str, data: RunData) -> None:
    m = ALLOC_RE.search(line)
    if m:
        data.allocations[m.group(2).strip()] = int(m.group(1))
        return
    m = STEP_RE.search(line)
    if m:
        data.step_ms.append(float(m.group(3)))
        data.step_tok_s.append(float(m.group(4)))
        return
    m = AVG_RE.search(line)
    if m:
        data.reported_avg_ms = float(m.group(1))


def fmt(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.2f}{suffix}"


def print_report(data: RunData, elapsed_s: float, returncode: int) -> None:
    print("\n[xray] runtime anatomy")
    print("[xray] ------------------------------")
    print(f"[xray] child exit code       : {returncode}")
    print(f"[xray] wall time             : {elapsed_s:.2f} s")

    if data.allocations:
        total = sum(data.allocations.values())
        print(f"[xray] reported allocations  : {total} MiB")
        for name, mib in data.allocations.items():
            pct = 100.0 * mib / total if total else 0.0
            print(f"[xray]   {name:<27} {mib:>5} MiB  ({pct:5.1f}%)")

    if data.step_ms:
        print(f"[xray] training steps        : {len(data.step_ms)}")
        print(f"[xray] step latency mean     : {statistics.fmean(data.step_ms):.2f} ms")
        print(f"[xray] step latency p50      : {fmt(percentile(data.step_ms, 0.50), ' ms')}")
        print(f"[xray] step latency p95      : {fmt(percentile(data.step_ms, 0.95), ' ms')}")
        print(f"[xray] throughput mean       : {statistics.fmean(data.step_tok_s):.0f} tok/s")
    if data.reported_avg_ms is not None:
        print(f"[xray] program avg iteration : {data.reported_avg_ms:.2f} ms")

    if data.gpu_samples:
        peak = max(data.gpu_samples, key=lambda s: s.used_mib)
        util = [s.util_pct for s in data.gpu_samples]
        temps = [s.temp_c for s in data.gpu_samples]
        print(f"[xray] GPU samples           : {len(data.gpu_samples)}")
        print(
            f"[xray] peak GPU memory       : {peak.used_mib}/{peak.total_mib} MiB "
            f"({100.0 * peak.used_mib / peak.total_mib:.1f}%)"
        )
        print(f"[xray] GPU util mean / peak  : {statistics.fmean(util):.1f}% / {max(util)}%")
        print(f"[xray] GPU temp peak         : {max(temps)} C")
    else:
        print("[xray] GPU telemetry         : unavailable")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Observe an llm.c run without changing model/training behavior."
    )
    parser.add_argument(
        "--sample-ms",
        type=int,
        default=250,
        help="nvidia-smi sampling interval in milliseconds (default: 250)",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command to run; separate it with --",
    )
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command; example: -- ./train_gpt2fp32cu -b 4 -t 512")
    if args.sample_ms < 50:
        parser.error("--sample-ms must be >= 50")

    data = RunData()
    stop = threading.Event()
    sampler = threading.Thread(
        target=sample_gpu,
        args=(stop, args.sample_ms / 1000.0, data.gpu_samples),
        daemon=True,
    )

    print(f"[xray] launching: {' '.join(command)}")
    sampler.start()
    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            parse_line(line, data)
        returncode = proc.wait()
    except KeyboardInterrupt:
        if 'proc' in locals() and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        returncode = 130
    finally:
        stop.set()
        sampler.join(timeout=3)

    elapsed_s = time.monotonic() - start
    print_report(data, elapsed_s, returncode)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
