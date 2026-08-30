#!/usr/bin/env bash
set -euo pipefail

B=${1:-4}
T=${2:-512}
TOKEN=${3:-1186}
ARCH=${XRAY_ARCH:-sm_89}
OUT=${XRAY_OUT:-/tmp/execution_pair_interaction_generator_probe}

echo "[xray] building execution pair interaction generator probe for ${ARCH}"
nvcc -O3 -std=c++17 -arch=${ARCH} \
  -I. -I./dev/cuda \
  dev/xray/execution_pair_interaction_generator_probe.cu \
  -lcublas -lcublasLt \
  -o "${OUT}"

echo "[xray] execution pair interaction generator: B=${B} T=${T} token=${TOKEN}"
"${OUT}" "${B}" "${T}" "${TOKEN}"
