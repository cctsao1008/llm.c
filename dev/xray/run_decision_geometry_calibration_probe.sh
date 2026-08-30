#!/usr/bin/env bash
set -euo pipefail

B="${1:-4}"
T="${2:-512}"
ARCH="${XRAY_ARCH:-sm_89}"
OUT="/tmp/decision_geometry_calibration_probe"

echo "[xray] building decision geometry calibration probe for ${ARCH}"
nvcc -O3 -arch="${ARCH}" -I. dev/xray/decision_geometry_calibration_probe.cu -lcublas -lcublasLt -o "${OUT}"

echo "[xray] decision geometry calibration: B=${B} T=${T}"
"${OUT}" "${B}" "${T}"
