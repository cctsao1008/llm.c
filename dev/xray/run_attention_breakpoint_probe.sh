#!/usr/bin/env bash
set -euo pipefail

B="${1:-4}"
T="${2:-512}"
ARCH="${XRAY_ARCH:-89}"
BIN="/tmp/xray_attention_breakpoint_probe"

printf '[xray] building attention breakpoint probe for sm_%s\n' "$ARCH"
nvcc -O3 -std=c++17 -arch="sm_${ARCH}" -I. \
  dev/xray/attention_breakpoint_probe.cu \
  -lcublas \
  -o "$BIN"

printf '[xray] attention breakpoint discovery: B=%s T=%s\n' "$B" "$T"
"$BIN" "$B" "$T"
