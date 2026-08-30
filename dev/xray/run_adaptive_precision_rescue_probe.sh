#!/usr/bin/env bash
set -euo pipefail

B="${1:-4}"
T="${2:-512}"
ARCH="${ARCH:-sm_89}"
OUT="${OUT:-/tmp/adaptive_precision_rescue_probe}"

printf '[xray] building adaptive precision rescue probe for %s\n' "$ARCH"
nvcc -O3 -arch="$ARCH" -I. \
  dev/xray/adaptive_precision_rescue_probe.cu \
  -o "$OUT" -lcublas -lcublasLt

printf '[xray] adaptive precision rescue: B=%s T=%s\n' "$B" "$T"
"$OUT" "$B" "$T"
