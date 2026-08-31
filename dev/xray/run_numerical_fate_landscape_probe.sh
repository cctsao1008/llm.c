#!/usr/bin/env bash
set -euo pipefail

B=${1:-4}
T=${2:-512}
BATCHES=${3:-8}
CSV=${4:-/tmp/numerical_fate_landscape.csv}
NEAR_MARGIN=${5:-0.01}
REPEATS=${6:-2}
ARCH=${XRAY_ARCH:-sm_89}
OUT=${XRAY_OUT:-/tmp/numerical_fate_landscape_probe}

echo "[xray] building numerical fate landscape probe for ${ARCH}"
nvcc -O3 -std=c++17 -arch=${ARCH} -Xcompiler -fopenmp \
  -I. -I./dev/cuda \
  dev/xray/numerical_fate_landscape_probe.cu \
  -lcublas -lcublasLt -lgomp \
  -o "${OUT}"

echo "[xray] numerical fate landscape: B=${B} T=${T} batches=${BATCHES} near_margin=${NEAR_MARGIN} repeats=${REPEATS}"
echo "[xray] csv=${CSV}"
"${OUT}" "${B}" "${T}" "${BATCHES}" "${CSV}" "${NEAR_MARGIN}" "${REPEATS}"
