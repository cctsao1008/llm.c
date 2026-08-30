#!/usr/bin/env bash
set -euo pipefail

B="${1:-4}"
T="${2:-512}"
ARCH="${XRAY_ARCH:-sm_89}"
OUT="/tmp/tf32_rasterization_probe"

echo "[xray] building TF32 rasterization probe for ${ARCH}"
nvcc -O3 -arch="${ARCH}" -I. dev/xray/tf32_rasterization_probe.cu -lcublas -lcublasLt -o "${OUT}"

echo "[xray] TF32 rasterization discovery: B=${B} T=${T}"
"${OUT}" "${B}" "${T}"
