#!/usr/bin/env bash
set -euo pipefail

B=${1:-4}
T=${2:-512}
BATCH_INDEX=${3:-5}
TOKEN=${4:-134}
shift $(( $# >= 4 ? 4 : $# )) || true

ARCH=${XRAY_ARCH:-sm_89}
OUT=${XRAY_OUT:-/tmp/decision_numerical_fate_stable_lock_stage_probe}

echo "[xray] building stable-lock stage probe for ${ARCH}"
nvcc -O3 -std=c++17 -arch=${ARCH} \
  -Xcompiler -fopenmp \
  -I. -I./dev/cuda \
  dev/xray/decision_numerical_fate_stable_lock_stage_probe.cu \
  -lcublas -lcublasLt -Xcompiler -lgomp \
  -o "${OUT}"

if (( $# == 0 )); then
  echo "[xray] stable-lock stage: B=${B} T=${T} batch=${BATCH_INDEX} token=${TOKEN} specs=default(qkv:10 attproj:7 fc:2 fcproj:3)"
  "${OUT}" "${B}" "${T}" "${BATCH_INDEX}" "${TOKEN}"
else
  echo "[xray] stable-lock stage: B=${B} T=${T} batch=${BATCH_INDEX} token=${TOKEN} specs=$*"
  "${OUT}" "${B}" "${T}" "${BATCH_INDEX}" "${TOKEN}" "$@"
fi
