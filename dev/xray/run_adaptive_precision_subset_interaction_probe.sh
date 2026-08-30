#!/usr/bin/env bash
set -euo pipefail

B="${1:-4}"
T="${2:-512}"
TOKEN="${3:-1186}"
ARCH="${ARCH:-sm_89}"
OUT="${OUT:-/tmp/adaptive_precision_subset_interaction_probe}"

printf '[xray] building adaptive precision subset-interaction probe for %s\n' "$ARCH"
nvcc -O3 -arch="$ARCH" -I. \
  dev/xray/adaptive_precision_subset_interaction_probe.cu \
  -o "$OUT" -lcublas -lcublasLt

printf '[xray] adaptive precision subset-interaction: B=%s T=%s token=%s\n' "$B" "$T" "$TOKEN"
"$OUT" "$B" "$T" "$TOKEN"
