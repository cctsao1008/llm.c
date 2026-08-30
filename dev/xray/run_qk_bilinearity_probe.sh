#!/usr/bin/env bash
set -euo pipefail

B="${1:-4}"
T="${2:-512}"
ARCH="${XRAY_ARCH:-89}"
OUT="/tmp/qk_bilinearity_probe"

printf '[xray] building QK bilinearity probe for sm_%s\n' "$ARCH"
nvcc -O3 -std=c++17 -arch="sm_${ARCH}" -I. \
  dev/xray/qk_bilinearity_probe.cu \
  -lcublas -o "$OUT"

printf '[xray] QK bilinearity discovery: B=%s T=%s\n' "$B" "$T"
"$OUT" "$B" "$T"
