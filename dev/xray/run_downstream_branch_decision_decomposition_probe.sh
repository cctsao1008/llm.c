#!/usr/bin/env bash
set -euo pipefail

B="${1:-4}"
T="${2:-512}"
TOKEN="${3:-1186}"
ARCH="${ARCH:-sm_89}"
OUT="/tmp/downstream_branch_decision_decomposition_probe"

echo "[xray] building downstream branch decision decomposition probe for ${ARCH}"
nvcc -O3 -arch="${ARCH}" -I. \
  dev/xray/downstream_branch_decision_decomposition_probe.cu \
  -o "${OUT}" -lcublas -lcublasLt

echo "[xray] downstream branch decision decomposition: B=${B} T=${T} token=${TOKEN}"
"${OUT}" "${B}" "${T}" "${TOKEN}"
