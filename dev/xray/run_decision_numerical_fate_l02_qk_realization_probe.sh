#!/usr/bin/env bash
set -euo pipefail

B=${1:-4}
T=${2:-512}
TOKEN=${3:-1186}
ARCH=${XRAY_ARCH:-sm_89}
OUT=${XRAY_OUT:-/tmp/decision_numerical_fate_l02_qk_realization_probe}

echo "[xray] building L02 QK numerical realization probe for ${ARCH}"
nvcc -O3 -std=c++17 -arch=${ARCH} -Xcompiler -fopenmp \
  -I. -I./dev/cuda \
  dev/xray/decision_numerical_fate_l02_qk_realization_probe.cu \
  -lcublas -lcublasLt -lgomp \
  -o "${OUT}"

echo "[xray] L02 QK numerical realization: B=${B} T=${T} token=${TOKEN}"
"${OUT}" "${B}" "${T}" "${TOKEN}"
