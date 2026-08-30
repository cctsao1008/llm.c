#!/usr/bin/env bash
set -euo pipefail

B="${1:-4}"
T="${2:-512}"
ARCH="${XRAY_ARCH:-sm_89}"
OUT="/tmp/hardware_directional_risk_probe"

echo "[xray] building hardware directional risk probe for ${ARCH}"
nvcc -O3 -arch="${ARCH}" -I. dev/xray/hardware_directional_risk_probe.cu -lcublas -lcublasLt -o "${OUT}"

echo "[xray] hardware directional decision geometry: B=${B} T=${T}"
"${OUT}" "${B}" "${T}"
