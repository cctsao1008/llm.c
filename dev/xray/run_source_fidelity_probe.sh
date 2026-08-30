#!/usr/bin/env bash
set -euo pipefail
B="${1:-4}"
T="${2:-512}"
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')"
NVCC="${NVCC:-nvcc}"
printf '[xray] building natural source fidelity probe for sm_%s\n' "$CC"
"$NVCC" --threads=0 --use_fast_math -std=c++17 -O3 \
  --generate-code "arch=compute_${CC},code=sm_${CC}" \
  -I. dev/xray/natural_source_fidelity_probe.cu \
  -lcublas -lcublasLt \
  -o xray_natural_source_fidelity_probe
printf '[xray] natural source fidelity discovery: B=%s T=%s\n' "$B" "$T"
./xray_natural_source_fidelity_probe "$B" "$T"
