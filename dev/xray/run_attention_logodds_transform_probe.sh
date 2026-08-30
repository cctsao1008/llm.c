#!/usr/bin/env bash
set -euo pipefail

B="${1:-4}"
T="${2:-512}"
ARCH="${XRAY_ARCH:-sm_89}"
OUT="/tmp/attention_logodds_transform_probe"

echo "[xray] building attention log-odds transform probe for ${ARCH}"
nvcc -O3 -arch="${ARCH}" -I. dev/xray/attention_logodds_transform_probe.cu -lcublas -lcublasLt -o "${OUT}"

echo "[xray] attention log-odds transform discovery: B=${B} T=${T}"
"${OUT}" "${B}" "${T}"
