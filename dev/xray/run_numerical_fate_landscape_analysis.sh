#!/usr/bin/env bash
set -euo pipefail

CSV=${1:-/tmp/numerical_fate_landscape.csv}
REPORT=${2:-/tmp/numerical_fate_landscape_report.md}
UNIQUE=${3:-/tmp/numerical_fate_landscape_unique_flips.csv}

echo "[xray] analyzing numerical fate landscape"
echo "[xray] input=${CSV}"
echo "[xray] report=${REPORT}"
echo "[xray] unique_flips=${UNIQUE}"

python3 dev/xray/analyze_numerical_fate_landscape.py \
  "${CSV}" \
  --report "${REPORT}" \
  --unique-csv "${UNIQUE}"
