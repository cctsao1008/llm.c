#!/usr/bin/env bash
set -euo pipefail

B="${1:-4}"
T="${2:-512}"
TOKEN="${3:-1186}"
ARCH="${ARCH:-sm_89}"
OUT="${OUT:-/tmp/adaptive_precision_causal_cone_probe}"

printf '[xray] building adaptive precision causal-cone probe for %s\n' "$ARCH"
nvcc -O3 -arch="$ARCH" -I. \
  dev/xray/adaptive_precision_causal_cone_probe.cu \
  -o "$OUT" -lcublas -lcublasLt

printf '[xray] adaptive precision causal-cone: B=%s T=%s token=%s\n' "$B" "$T" "$TOKEN"
"$OUT" "$B" "$T" "$TOKEN"
