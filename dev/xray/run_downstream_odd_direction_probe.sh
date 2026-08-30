#!/usr/bin/env bash
set -euo pipefail

B="${1:-4}"
T="${2:-512}"
TOKEN="${3:-1186}"
ARCH="${XRAY_ARCH:-sm_89}"
OUT="${XRAY_OUT:-build/xray_downstream_odd_direction_probe}"

mkdir -p "$(dirname "$OUT")"
echo "[xray] building downstream odd-direction probe for ${ARCH}"
nvcc -O3 -std=c++17 -arch="${ARCH}" -I. dev/xray/downstream_odd_direction_probe.cu -lcublas -o "$OUT"
echo "[xray] downstream odd-direction: B=${B} T=${T} token=${TOKEN}"
"$OUT" "$B" "$T" "$TOKEN"
