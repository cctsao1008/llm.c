#!/usr/bin/env bash
set -euo pipefail

B=${1:-4}
T=${2:-512}
BATCH_INDEX=${3:-5}
TOKEN=${4:-134}
FAMILY=${5:-all}
ARCH=${XRAY_ARCH:-sm_89}
OUT=${XRAY_OUT:-/tmp/decision_numerical_fate_case_depth_probe}

echo "[xray] building numerical fate landscape-case depth probe for ${ARCH}"
nvcc -O3 -std=c++17 -arch=${ARCH} \
  -Xcompiler -fopenmp \
  -I. -I./dev/cuda \
  dev/xray/decision_numerical_fate_case_depth_probe.cu \
  -lcublas -lcublasLt -Xcompiler -lgomp \
  -o "${OUT}"

echo "[xray] numerical fate case depth: B=${B} T=${T} batch=${BATCH_INDEX} token=${TOKEN} family=${FAMILY}"
"${OUT}" "${B}" "${T}" "${BATCH_INDEX}" "${TOKEN}" "${FAMILY}"
