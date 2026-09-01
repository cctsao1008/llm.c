#!/usr/bin/env bash
set -euo pipefail

B=${XRAY_B:-4}
T=${XRAY_T:-512}
ARCH=${XRAY_ARCH:-sm_89}
OUT=${XRAY_OUT:-/tmp/numerical_fate_execution_boundary_path_probe}

cd "$(dirname "$0")/../.."

echo "[xray] building execution-boundary path probe for ${ARCH}"
nvcc -O3 -std=c++17 -arch=${ARCH} \
  -Xcompiler -fopenmp \
  -I. -I./dev/cuda \
  dev/xray/numerical_fate_execution_boundary_path_probe.cu \
  -lcublas -lcublasLt -Xcompiler -lgomp \
  -o "${OUT}"

run_case() {
  local label=$1
  local batch=$2
  local target=$3
  local family=$4
  local layer=$5
  echo
  echo "[xray] execution-boundary path pilot: ${label} B=${B} T=${T} batch=${batch} target=${target} family=${family} layer=${layer}"
  "${OUT}" "${B}" "${T}" "${batch}" "${target}" "${family}" "${layer}"
}

# Pre-registered representative cases:
# 1) terminal-classifier dominated endpoint;
# 2) full-suffix state-change-required created disagreement;
# 3) first tested-execution agreement point after that created region;
# 4) within-context removal of execution disagreement.
run_case "42/attproj L11 terminal-dominated C" 0 42 attproj 11
run_case "1186/fcproj L0 suffix-required C" 0 1186 fcproj 0
run_case "1186/fcproj L2 first agreement C->." 0 1186 fcproj 2
run_case "9766/fc L0 suffix-required R" 4 1574 fc 0
