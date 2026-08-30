# llm.c runtime x-ray

This branch is not for re-measuring facts already visible in normal `llm.c` output. It is for instrumenting execution paths that are otherwise hidden by the high-level model view.

The probe deliberately asks questions whose answers are not obvious from parameter counts, tensor shapes, or the existing training log:

- Where does steady-state GPU time actually go: forward, zero-grad, backward, or AdamW?
- How different is the cold path from steady state?
- Does CUDA/cuBLAS lazily allocate device memory behind the model's own allocations?
- How large is the wall-time vs GPU-time gap for each semantic phase?
- Which actual CUDA kernels and cuBLAS calls dominate the timeline?
- Are memory operations or API overhead unexpectedly significant?

`runtime_probe.cu` includes the existing FP32 implementation under `TESTING`, so the original training source remains untouched. It drives the same `GPT2` functions directly, adds CUDA-event timing, `cudaMemGetInfo()` snapshots, and NVTX semantic ranges, then exposes the run to Nsight Systems.

## Run it

From the repository root:

```bash
chmod +x dev/xray/run_xray.sh
./dev/xray/run_xray.sh 4 512 5
```

Arguments are:

```text
B T STEPS [OUTPUT_BASENAME]
```

The script builds `xray_runtime_probe`, runs the internal phase probe once, then — when `nsys` is available — captures a CUDA/cuBLAS/NVTX timeline and prints kernel, CUDA API, and GPU-memory-operation summaries.

Typical output begins with memory snapshots around the cold path:

```text
[xray][mem] process start      ...
[xray][mem] after model load   ...
[xray][cold] forward    gpu=... wall=... mem_delta=...
[xray][mem] after first forward ...
[xray][cold] backward   gpu=... wall=... mem_delta=...
[xray][cold] adamw      gpu=... wall=... mem_delta=...
```

Then it reports steady-state phase means and their share of measured GPU time.

## Important interpretation rule

This is a **discovery probe**, not a throughput benchmark. Phase boundaries use synchronization so their timings are individually attributable; that intentionally perturbs launch overlap and CPU scheduling. Use the generated Nsight Systems report for the least-distorted kernel timeline.

If a result merely confirms something already explicit in the source or normal log, it is not the interesting result. The useful output is whatever exposes behavior we could not confidently know before measuring it.
