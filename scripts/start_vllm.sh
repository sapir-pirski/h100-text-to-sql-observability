#!/usr/bin/env bash
#
# Start vLLM with your chosen configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-6144}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-48}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-24576}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.94}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
if [[ ! -x "$UV_BIN" ]]; then
    UV_BIN="uv"
fi

if command -v "$UV_BIN" >/dev/null 2>&1; then
    PYTHON_CMD=("$UV_BIN" run python)
elif [[ -x ".venv/bin/python" ]]; then
    PYTHON_CMD=(".venv/bin/python")
else
    printf 'Could not find uv or .venv/bin/python for vLLM startup.\n' >&2
    exit 1
fi

exec "${PYTHON_CMD[@]}" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
    --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --disable-log-requests \
    --uvicorn-log-level warning
