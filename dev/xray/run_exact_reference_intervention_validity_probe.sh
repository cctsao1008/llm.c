#!/usr/bin/env bash
set -euo pipefail

B="${1:-4}"
T="${2:-512}"
TOKEN="${3:-1186}"
ARCH="${ARCH:-sm_89}"
OUT="${OUT:-/tmp/exact_reference_intervention_validity_probe}"

printf '[xray] building exact reference intervention validity probe for %s\n' "$ARCH"
nvcc -O3 -arch="$ARCH" -I. \
  dev/xray/exact_reference_intervention_validity_probe.cu \
  -o "$OUT" -lcublas -lcublasLt

printf '[xray] exact reference intervention validity: B=%s T=%s token=%s\n' "$B" "$T" "$TOKEN"
"$OUT" "$B" "$T" "$TOKEN"
