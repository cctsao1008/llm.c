#!/usr/bin/env bash
set -euo pipefail

MANIFEST=${1:-/tmp/numerical_fate_atlas_seeds.csv}
LOG=${2:-/tmp/numerical_fate_atlas_trajectory.log}
SUMMARY=${3:-/tmp/numerical_fate_atlas_trajectory_summary.csv}
MAX_CASES=${4:-0}
ROLES=${5:-}
ARCH=${XRAY_ARCH:-sm_89}
OUT=${XRAY_OUT:-/tmp/numerical_fate_trajectory_probe}

echo "[xray] building numerical fate trajectory probe for ${ARCH}"
nvcc -O3 -std=c++17 -arch=${ARCH} \
  -Xcompiler -fopenmp \
  -I. -I./dev/cuda \
  dev/xray/numerical_fate_trajectory_probe.cu \
  -lcublas -lcublasLt -Xcompiler -lgomp \
  -o "${OUT}"

ARGS=(
  "${MANIFEST}"
  --exe "${OUT}"
  --log "${LOG}"
  --summary-csv "${SUMMARY}"
  --max-cases "${MAX_CASES}"
)
if [[ -n "${ROLES}" ]]; then
  ARGS+=(--roles "${ROLES}")
fi

python3 dev/xray/run_numerical_fate_atlas_manifest.py "${ARGS[@]}"
