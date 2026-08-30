# llm.c runtime x-ray

This directory contains experimental observability tooling for looking at the runtime behavior of `llm.c` without changing model math, kernels, optimizer behavior, or training results.

The first tool, `runtime_anatomy.py`, launches an existing llm.c executable, mirrors its stdout, samples NVIDIA GPU telemetry through `nvidia-smi`, parses the allocation and iteration data already emitted by the program, and prints a compact summary after the run.

## Why this exists

High-level model descriptions hide much of the machine behavior that determines whether a model is practical: activation footprint, optimizer state, peak device memory, latency distribution, GPU utilization, and transient slowdowns. The x-ray tooling is intended to make those properties visible before we modify the implementation itself.

The initial rule is deliberately strict:

> Observe first. Do not change the model or training behavior just to make it easier to measure.

## Quick start

Build the existing single-GPU FP32 target as usual:

```bash
make train_gpt2fp32cu
```

Then run it through the x-ray harness:

```bash
python3 dev/xray/runtime_anatomy.py -- \
  ./train_gpt2fp32cu -b 4 -t 512 -v 1000 -s 1000
```

The training program's normal output is preserved. At the end, an additional `[xray]` report is printed with information such as:

```text
[xray] reported allocations  : 4251 MiB
[xray] step latency mean     : ... ms
[xray] step latency p50      : ... ms
[xray] step latency p95      : ... ms
[xray] throughput mean       : ... tok/s
[xray] peak GPU memory       : ... MiB
[xray] GPU util mean / peak  : ...
[xray] GPU temp peak         : ... C
```

GPU telemetry is observational. A failed `nvidia-smi` sample does not fail or alter the child training process.

## Sampling interval

The default GPU sampling interval is 250 ms. It can be changed with:

```bash
python3 dev/xray/runtime_anatomy.py --sample-ms 100 -- \
  ./train_gpt2fp32cu -b 4 -t 512 -v 1000 -s 1000
```

The minimum accepted interval is 50 ms. Sampling itself has overhead, so this tool should not be treated as a replacement for Nsight when precise kernel-level profiling is required.

## Scope

This first stage observes the process from the outside and parses data the executable already exposes. It intentionally does **not** yet instrument individual forward, backward, optimizer, or CUDA-kernel phases.

Those internal measurements should be added only after this external baseline is stable, so that the cost and perturbation of instrumentation can be measured rather than assumed away.
