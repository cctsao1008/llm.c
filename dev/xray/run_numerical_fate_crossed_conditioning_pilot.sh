#!/usr/bin/env bash
set -euo pipefail

B=${XRAY_B:-4}
T=${XRAY_T:-512}
ARCH=${XRAY_ARCH:-sm_89}
OUT=${XRAY_OUT:-/tmp/numerical_fate_crossed_conditioning_probe}

echo "[xray] building crossed conditioning probe for ${ARCH}"
nvcc -O3 -std=c++17 -arch=${ARCH} \
  -Xcompiler -fopenmp \
  -I. -I./dev/cuda \
  dev/xray/numerical_fate_crossed_conditioning_probe.cu \
  -lcublas -lcublasLt -Xcompiler -lgomp \
  -o "${OUT}"

run_case() {
  local batch=$1
  local target=$2
  local family=$3
  local layer=$4
  local label=$5
  echo
  echo "[xray] crossed-conditioning pilot: ${label} B=${B} T=${T} batch=${batch} target=${target} family=${family} layer=${layer}"
  "${OUT}" "${B}" "${T}" "${batch}" "${target}" "${family}" "${layer}"
}

# Pre-registered direct specimens.  These are intentionally few: the purpose is
# to separate terminal classifier conditioning from full-suffix state transport
# before spending GPU time on a population study.
run_case 0 42   attproj 11 "42/attproj L11 terminal-sensitive positive control"
run_case 0 1186 fcproj   0 "1186/fcproj L0 created disagreement"
run_case 0 1186 fcproj   1 "1186/fcproj L1 created disagreement"
run_case 0 1186 fcproj   2 "1186/fcproj L2 first execution-agreement competitor state"
run_case 4 1574 fc       0 "9766/fc L0 removed disagreement; within-context near-|margin| contrast"
