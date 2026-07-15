#!/usr/bin/env bash
# Run the Gemma-2-9B pipeline ENTIRELY ON CPU (no GPU), offline.
# Proves the competition hardware requirement (CPU, 16 GB RAM). /usr/bin/time -v
# reports peak resident memory and wall-clock.
#
#   ./run_cpu.sh                 # full 50-row Test submission -> submission_task2_gemma9b.csv
#   ./run_cpu.sh --limit 3       # quick 3-row smoke test
#   ./run_cpu.sh --validate 20   # local WordNet-proxy score on 20 Train rows
set -euo pipefail
cd "$(dirname "$0")"
export TASK2_GPU_LAYERS=0
export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if command -v /usr/bin/time >/dev/null 2>&1; then
  /usr/bin/time -v python gemma9b_pipeline.py "$@"
else
  python gemma9b_pipeline.py "$@"
fi
