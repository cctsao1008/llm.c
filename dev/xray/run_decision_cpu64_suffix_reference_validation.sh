#!/usr/bin/env bash
set -euo pipefail

B=${1:-4}
T=${2:-512}
TOKEN=${3:-1186}
PAIR_A=${4:-11906}
PAIR_B=${5:-262}
OUT=${XRAY_OUT:-/tmp/decision_cpu64_suffix_reference_validation}
CXX=${CXX:-g++}

echo "[xray] building CPU64 suffix reference validator"
"${CXX}" -O2 -std=gnu++17 -fopenmp -I. \
  dev/xray/decision_cpu64_suffix_reference_validation.cpp \
  -lm -o "${OUT}"

echo "[xray] CPU64 suffix reference validation: B=${B} T=${T} token=${TOKEN} pair=${PAIR_A}/${PAIR_B}"
"${OUT}" "${B}" "${T}" "${TOKEN}" "${PAIR_A}" "${PAIR_B}"
