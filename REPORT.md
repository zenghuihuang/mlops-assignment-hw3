# MLOps Assignment Report

---

## Phase 1 — vLLM Serving Configuration

### Model & hardware

- **Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507` (MoE, 30 B total / 3 B active params per forward pass)
- **Hardware:** 1× H100 80 GB SXM

### Memory budget

The dominant constraint is VRAM. In BF16 the weights alone occupy ≈60 GB, leaving only ≈12 GB for KV cache — far too little for the 30+ concurrent sequences needed at 10 RPS. FP8 weight quantization halves that to ≈30 GB and opens up ≈30 GB for KV cache.

Estimated breakdown with the chosen config:

| Component | Size |
|---|---|
| FP8 weights | ≈30 GB |
| BF16 KV cache (64 seqs × 4096 tok, 28 layers, 8 KV heads, head_dim 128) | ≈28 GB |
| Activations + CUDA overhead | ≈5 GB |
| **Total** | **≈63 GB < 73.6 GB** (0.92 × 80 GB) ✓ |

### Configuration flags

| Flag | Value | Justification |
|---|---|---|
| `--dtype bfloat16` | `bfloat16` | Native compute dtype for H100 Tensor Cores; higher numerical range than FP16, no subnormal issues. |
| `--quantization fp8` | `fp8` | Halves weight storage (60 GB → 30 GB); H100 has hardware FP8 Tensor Cores so no throughput penalty; typical quality loss <1% on instruction-tuned models. This is the single flag that makes the rest of the config viable. |
| `--max-model-len` | `4096` | Workload prompts are 1.5–3 K tokens; SQL outputs are 50–200 tokens. Capping at 4 K vs the model's 32 K+ default reduces per-sequence KV allocation by 8×, which is what allows 64 concurrent seqs to fit in VRAM. |
| `--max-num-seqs` | `64` | At 10 RPS with 2–3 sequential LLM calls per agent run, steady-state concurrency is ≈30 LLM requests. 64 provides headroom for burst and allows the continuous-batching scheduler to build larger decode batches, improving throughput. |
| `--gpu-memory-utilization` | `0.92` | Small bump above the 0.90 default reclaims ≈1.6 GB of additional KV cache. We can afford this because the memory budget is well-understood with FP8 weights. |
| `--enable-prefix-caching` | — | Each question for a given DB always has the same schema prefix (500–1 K tokens) in the prompt. vLLM caches those KV states and reuses them across requests, eliminating redundant prefill work for every subsequent question on the same DB. |
| `--enable-chunked-prefill` | — | 3 K-token prompts take several milliseconds to prefill on their own; without chunking they block decode batches and create P95 spikes. Chunked prefill interleaves prefill chunks with decode steps, keeping latency smooth under load. |
| `--max-num-batched-tokens` | `8192` | Allows 2–3 full prompts to be prefilled per scheduler step, keeping GPU compute saturated and reducing time-to-first-token for queued requests. |
| `--trust-remote-code` | — | Qwen3-2507 is a July 2025 release; vLLM's bundled tokenizer configs may not include it. Trusting remote code ensures the exact tokenizer and architecture from HuggingFace is used. |

### Why these levers, given this workload

The SLO is P95 end-to-end agent latency < 5 s at ≥10 RPS. Each agent run makes 2–3 **sequential** LLM calls, so the wall-clock budget per call is roughly 1–2 s. The model's MoE architecture (3 B active params) means the per-token compute is closer to a 3 B dense model — prefill and decode are fast. The risk is not raw throughput but **queuing**: if too few sequences can be batched simultaneously the scheduler idles GPU cycles or accumulates latency.

The chain of decisions therefore is:

1. **FP8 quantization** to open VRAM for KV cache → enables high concurrency.
2. **`--max-model-len 4096`** to make each sequence's KV footprint small → keeps 64 seqs in cache simultaneously.
3. **`--enable-chunked-prefill`** to prevent long prefills from stalling decode → protects P95 tail latency.
4. **`--enable-prefix-caching`** to eliminate redundant schema prefill → reduces TTFT for repeat-DB queries and saves bandwidth.

### Verification

After launch, confirmed with:

```bash
# Health check
curl http://localhost:8000/health

# Model list
curl http://localhost:8000/v1/models

# Manual query (example)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "messages": [
      {"role": "system", "content": "You are a SQL expert. Return only a SQL query, no explanation."},
      {"role": "user", "content": "Schema:\nCREATE TABLE \"singers\" (\"Singer_ID\" INTEGER PRIMARY KEY, \"Name\" TEXT, \"Country\" TEXT, \"Song_Name\" TEXT, \"Song_release_year\" TEXT, \"Age\" INTEGER, \"Is_male\" TEXT);\n\nQuestion: How many singers are from the US?"}
    ],
    "temperature": 0,
    "max_tokens": 256
  }'
```

3–5 queries from `evals/eval_set.jsonl` returned valid SQL. Screenshot: `screenshots/vllm_manual_query.png`.

---

## Phase 2 — Observability Dashboard

*To be filled in after Phase 2.*

---

## Phase 3 — Agent Design

*To be filled in after Phase 3.*

---

## Phase 4 — Agent Tracing

*To be filled in after Phase 4.*

---

## Phase 5 — Evals

*To be filled in after Phase 5.*

---

## Phase 6 — SLO Diagnosis & Iteration

*To be filled in after Phase 6.*

---

## Phase 7 — Wrap-up

*To be filled in after Phase 7.*
