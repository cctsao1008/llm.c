#!/usr/bin/env bash
set -euo pipefail

B="${1:-4}"
T="${2:-512}"
THRESHOLD="${3:-1e-3}"
ARCH="${ARCH:-sm_89}"
OUT="${OUT:-/tmp/adaptive_precision_actual_rescue_probe}"

printf '[xray] building actual adaptive precision rescue probe for %s\n' "$ARCH"
nvcc -O3 -arch="$ARCH" -I. \
  dev/xray/adaptive_precision_actual_rescue_probe.cu \
  -o "$OUT" -lcublas -lcublasLt

printf '[xray] actual adaptive precision rescue: B=%s T=%s threshold=%s\n' "$B" "$T" "$THRESHOLD"
"$OUT" "$B" "$T" "$THRESHOLD"
