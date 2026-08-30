#!/usr/bin/env bash
set -euo pipefail

B="${1:-4}"
T="${2:-512}"
STEPS="${3:-5}"
OUT="${4:-xray_runtime}"

CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')"
NVCC="${NVCC:-nvcc}"

printf '[xray] building probe for sm_%s\n' "$CC"
"$NVCC" --threads=0 --use_fast_math -std=c++17 -O3 \
  --generate-code "arch=compute_${CC},code=sm_${CC}" \
  -I. dev/xray/runtime_probe.cu \
  -lcublas -lcublasLt \
  -o xray_runtime_probe

printf '[xray] direct probe: B=%s T=%s steps=%s\n' "$B" "$T" "$STEPS"
./xray_runtime_probe "$B" "$T" "$STEPS"

if command -v nsys >/dev/null 2>&1; then
  printf '\n[xray] capturing CUDA/cuBLAS/NVTX timeline with Nsight Systems\n'
  nsys profile \
    --force-overwrite=true \
    --sample=none \
    --trace=cuda,cublas,nvtx,osrt \
    -o "$OUT" \
    ./xray_runtime_probe "$B" "$T" "$STEPS"

  printf '\n[xray] CUDA kernel summary\n'
  nsys stats --report cuda_gpu_kern_sum "${OUT}.nsys-rep" || true

  printf '\n[xray] CUDA API summary\n'
  nsys stats --report cuda_api_sum "${OUT}.nsys-rep" || true

  printf '\n[xray] CUDA memory operation summary\n'
  nsys stats --report cuda_gpu_mem_time_sum "${OUT}.nsys-rep" || true
else
  printf '[xray] nsys not found; phase probe completed, timeline capture skipped\n'
fi
