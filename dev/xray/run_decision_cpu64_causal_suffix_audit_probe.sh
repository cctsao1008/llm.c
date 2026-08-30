#!/usr/bin/env bash
set -euo pipefail

B=${1:-4}
T=${2:-512}
TOKEN=${3:-1186}
ARCH=${XRAY_ARCH:-sm_89}
OUT=${XRAY_OUT:-/tmp/decision_cpu64_causal_suffix_audit_probe}

echo "[xray] building CPU64 causal suffix audit for ${ARCH}"
nvcc -O3 -std=c++17 -arch=${ARCH} \
  -I. -I./dev/cuda \
  -Xcompiler -fopenmp \
  dev/xray/decision_cpu64_causal_suffix_audit_probe.cu \
  -lcublas -lcublasLt -lgomp \
  -o "${OUT}"

echo "[xray] CPU64 causal suffix audit: B=${B} T=${T} token=${TOKEN}"
"${OUT}" "${B}" "${T}" "${TOKEN}"
