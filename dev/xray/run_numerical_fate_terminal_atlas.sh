#!/usr/bin/env bash
set -euo pipefail

MANIFEST=${1:-/tmp/numerical_fate_atlas_all_flips.csv}
LOG=${2:-/tmp/numerical_fate_terminal_atlas.log}
SUMMARY=${3:-/tmp/numerical_fate_terminal_atlas.csv}
MAX_CASES=${4:-0}
ROLES=${5:-flip}
ARCH=${XRAY_ARCH:-sm_89}
OUT=${XRAY_OUT:-/tmp/numerical_fate_terminal_readout_probe}

echo "[xray] building numerical fate terminal-readout probe for ${ARCH}"
nvcc -O3 -std=c++17 -arch=${ARCH} \
  -Xcompiler -fopenmp \
  -I. -I./dev/cuda \
  dev/xray/numerical_fate_terminal_readout_probe.cu \
  -lcublas -lcublasLt -Xcompiler -lgomp \
  -o "${OUT}"

python3 dev/xray/run_numerical_fate_terminal_atlas.py \
  "${MANIFEST}" \
  --exe "${OUT}" \
  --log "${LOG}" \
  --summary-csv "${SUMMARY}" \
  --max-cases "${MAX_CASES}" \
  --roles "${ROLES}"
