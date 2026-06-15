#!/usr/bin/env bash
#
# Start vLLM with your chosen configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
#
# Memory budget (H100 80 GB):
#   FP8 weights (~30 GB) + BF16 KV cache for 64 seqs × 4096 tokens (~28 GB)
#   + activations/overhead (~5 GB) ≈ 63 GB < 73.6 GB (0.92 × 80 GB)
#
# Without FP8: BF16 weights alone are ~60 GB, leaving <12 GB for KV cache
# — not enough for the 30+ concurrent sequences needed at 10 RPS.

set -euo pipefail

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --quantization fp8 \
    --max-model-len 4096 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --trust-remote-code
