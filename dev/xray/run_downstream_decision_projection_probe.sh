#!/usr/bin/env bash
set -euo pipefail

B="${1:-4}"
T="${2:-512}"
TOKEN="${3:-1186}"
ARCH="${ARCH:-sm_89}"
NVCC="${NVCC:-nvcc}"
OUT="/tmp/xray_downstream_decision_projection_probe"

printf '[xray] building downstream decision projection probe for %s\n' "$ARCH"
"$NVCC" -O3 -std=c++17 -arch="$ARCH" \
  -I. -Idev/xray \
  dev/xray/downstream_decision_projection_probe.cu \
  -lcublas -lcudart -o "$OUT"

printf '[xray] downstream decision projection: B=%s T=%s token=%s\n' "$B" "$T" "$TOKEN"
"$OUT" "$B" "$T" "$TOKEN"
