#!/usr/bin/env bash
set -euo pipefail

B=${1:-4}
T=${2:-512}
TOKEN=${3:-1186}
ARCH=${XRAY_ARCH:-sm_89}
OUT=${XRAY_OUT:-/tmp/decision_cpu64_readout_audit_probe}

echo "[xray] building CPU64 decision readout audit for ${ARCH}"
nvcc -O3 -std=c++17 -arch=${ARCH} \
  -I. -I./dev/cuda \
  dev/xray/decision_cpu64_readout_audit_probe.cu \
  -lcublas -lcublasLt \
  -o "${OUT}"

echo "[xray] CPU64 decision readout audit: B=${B} T=${T} token=${TOKEN}"
"${OUT}" "${B}" "${T}" "${TOKEN}"
