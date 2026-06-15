"""Send 3-5 questions from eval_set.jsonl directly to vLLM and print the SQL responses.

Run:
    uv run python scripts/manual_query.py

Calls vLLM's OpenAI-compatible /v1/chat/completions endpoint without the agent
stack — useful to confirm the model is up and generating SQL before running evals.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
N_QUESTIONS = 5

SYSTEM = (
    "You are an expert SQL assistant. Given a natural language question, "
    "write a single SQLite SQL query that answers it. "
    "Return ONLY the SQL query — no explanation, no markdown fences. "
    "End the query with a semicolon."
)


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def looks_like_sql(text: str) -> bool:
    upper = text.upper().lstrip()
    return any(upper.startswith(kw) for kw in ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE"))


def query_vllm(question: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nSQL:"},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }
    resp = httpx.post(VLLM_URL, json=payload, timeout=60.0)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return strip_thinking(raw).strip()


def main() -> None:
    questions = [
        json.loads(line)
        for line in EVAL_FILE.read_text().splitlines()
        if line.strip()
    ][:N_QUESTIONS]

    print(f"vLLM manual query — {VLLM_URL}")
    print(f"Model : {MODEL}")
    print(f"Sample: first {N_QUESTIONS} questions from eval_set.jsonl")
    print("=" * 72)

    passed = 0
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{N_QUESTIONS}] db={q['db_id']}")
        print(f"Q: {q['question']}")
        try:
            sql = query_vllm(q["question"])
            valid = looks_like_sql(sql)
            status = "OK " if valid else "WARN"
            if valid:
                passed += 1
            print(f"SQL [{status}]: {sql}")
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)

    print("\n" + "=" * 72)
    print(f"Result: {passed}/{N_QUESTIONS} responses look like valid SQL")


if __name__ == "__main__":
    main()
