#!/usr/bin/env bash
set -euo pipefail

B=${1:-4}
T=${2:-512}
BATCH=${3:-0}
TARGET=${4:-42}
FAMILY=${5:-attproj}
ARCH=${XRAY_ARCH:-sm_89}
OUT=${XRAY_OUT:-/tmp/numerical_fate_terminal_readout_probe}

echo "[xray] building numerical fate terminal-readout probe for ${ARCH}"
nvcc -O3 -std=c++17 -arch=${ARCH} \
  -Xcompiler -fopenmp \
  -I. -I./dev/cuda \
  dev/xray/numerical_fate_terminal_readout_probe.cu \
  -lcublas -lcublasLt -Xcompiler -lgomp \
  -o "${OUT}"

echo "[xray] terminal-readout: B=${B} T=${T} batch=${BATCH} target=${TARGET} family=${FAMILY}"
"${OUT}" "${B}" "${T}" "${BATCH}" "${TARGET}" "${FAMILY}"
