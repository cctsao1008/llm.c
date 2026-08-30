#!/usr/bin/env bash
set -euo pipefail

B="${1:-4}"
T="${2:-512}"
ARCH="${XRAY_ARCH:-sm_89}"
OUT="/tmp/qk_precision_probe"

echo "[xray] building QK precision response probe for ${ARCH}"
nvcc -O3 -arch="${ARCH}" -I. dev/xray/qk_precision_response_probe.cu -lcublas -lcublasLt -o "${OUT}"

echo "[xray] QK precision response discovery: B=${B} T=${T}"
"${OUT}" "${B}" "${T}"
