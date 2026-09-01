#!/usr/bin/env bash
set -euo pipefail

ARCH=${XRAY_ARCH:-sm_89}
OUT=${XRAY_OUT:-/tmp/numerical_fate_crossed_state_execution_probe}

echo "[xray] building crossed exact-state x suffix-execution probe for ${ARCH}"
nvcc -O3 -std=c++17 -arch=${ARCH} \
  -Xcompiler -fopenmp \
  -I. -I./dev/cuda \
  dev/xray/numerical_fate_crossed_state_execution_probe.cu \
  -lcublas -lcublasLt -Xcompiler -lgomp \
  -o "${OUT}"

run_case() {
  local B=$1
  local T=$2
  local BATCH=$3
  local TARGET=$4
  local FAMILY=$5
  echo
  echo "[xray] crossed-state-exec pilot: B=${B} T=${T} batch=${BATCH} target=${TARGET} family=${FAMILY}"
  "${OUT}" "${B}" "${T}" "${BATCH}" "${TARGET}" "${FAMILY}"
}

# Pre-registered pilot specimens:
# 42/attproj  : terminal-sensitive case
# 9766/fc     : CPU64 persistent vs GPU-terminal reversible in prior dual-readout
# 1186/fcproj : monotone case with prior topology agreement
run_case 4 512 0 42   attproj
run_case 4 512 4 1574 fc
run_case 4 512 0 1186 fcproj
