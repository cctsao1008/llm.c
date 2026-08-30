#!/usr/bin/env bash
set -euo pipefail

B="${1:-4}"
T="${2:-512}"
ARCH="${XRAY_ARCH:-sm_89}"
SRC="dev/xray/response_stage_localization_probe.cu"
BIN="/tmp/xray_stage_response_probe"

echo "[xray] building stage response localization probe for ${ARCH}"
nvcc -O3 -std=c++17 -arch="${ARCH}" -I. "${SRC}" -lcublas -o "${BIN}"

echo "[xray] stage response localization discovery: B=${B} T=${T}"
"${BIN}" "${B}" "${T}"
