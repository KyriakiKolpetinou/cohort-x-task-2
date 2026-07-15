#!/usr/bin/env bash
# Optional: run on an NVIDIA GPU for speed (identical output at temperature 0).
# Requires a CUDA-enabled llama-cpp-python build. Gemma-2-9B has 42 layers, so 43
# offloads everything (fits in ~6-7 GB VRAM).
#
#   ./run_gpu.sh                 # full 50-row Test submission
#   ./run_gpu.sh --limit 3       # quick smoke test
set -euo pipefail
cd "$(dirname "$0")"
export TASK2_GPU_LAYERS="${TASK2_GPU_LAYERS:-43}"
export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec python gemma9b_pipeline.py "$@"
