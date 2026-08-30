#!/usr/bin/env bash
set -euo pipefail

B=${1:-4}
T=${2:-512}
TOKEN=${3:-1186}
ARCH=${XRAY_ARCH:-sm_89}
OUT=${XRAY_OUT:-/tmp/attention_interaction_conditioning_probe}

echo "[xray] building attention interaction conditioning probe for ${ARCH}"
nvcc -O3 -std=c++17 -arch=${ARCH} \
  -I. -I./dev/cuda \
  dev/xray/attention_interaction_conditioning_probe.cu \
  -lcublas -lcublasLt \
  -o "${OUT}"

echo "[xray] attention interaction conditioning: B=${B} T=${T} token=${TOKEN}"
"${OUT}" "${B}" "${T}" "${TOKEN}"
