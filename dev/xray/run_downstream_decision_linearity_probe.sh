#!/usr/bin/env bash
set -euo pipefail

B="${1:-4}"
T="${2:-512}"
TOKEN="${3:-1186}"
ARCH="${ARCH:-sm_89}"
OUT="${OUT:-/tmp/xray_downstream_decision_linearity_probe}"

echo "[xray] building downstream decision linearity probe for ${ARCH}"
nvcc -O3 -std=c++17 -arch="${ARCH}" \
  -I. -I./llmc \
  dev/xray/downstream_decision_linearity_probe.cu \
  -lcublas -lcublasLt \
  -o "${OUT}"

echo "[xray] downstream decision linearity: B=${B} T=${T} token=${TOKEN}"
"${OUT}" "${B}" "${T}" "${TOKEN}"
