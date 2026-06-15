"""Fire 10 questions from eval_set.jsonl through the agent server and print results.

Run AFTER starting the agent server:
    uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001

Then:
    uv run python scripts/fire_traces.py

Each request is tagged with db_id + eval_run=phase4 so traces are filterable
in Langfuse. The script prints per-question status and a summary at the end.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
AGENT_URL = "http://localhost:8001/answer"
N_QUESTIONS = 10
EVAL_RUN_TAG = "phase4"


def main() -> None:
    questions = [
        json.loads(line)
        for line in EVAL_FILE.read_text().splitlines()
        if line.strip()
    ][:N_QUESTIONS]

    print(f"Firing {len(questions)} questions → {AGENT_URL}")
    print(f"Langfuse tag: eval_run={EVAL_RUN_TAG}")
    print("=" * 72)

    ok_count = 0
    revised_count = 0

    for i, q in enumerate(questions, 1):
        payload = {
            "question": q["question"],
            "db": q["db_id"],
            "tags": {
                "eval_run": EVAL_RUN_TAG,
                "db_id": q["db_id"],
                "question_idx": str(i),
            },
        }
        try:
            resp = httpx.post(AGENT_URL, json=payload, timeout=120.0)
            resp.raise_for_status()
            data = resp.json()

            status = "OK  " if data["ok"] else "FAIL"
            iters = data["iterations"]
            revised = iters > 1
            if data["ok"]:
                ok_count += 1
            if revised:
                revised_count += 1

            revised_marker = " [REVISED]" if revised else ""
            print(f"[{i:02d}/{N_QUESTIONS}] {status} iters={iters}{revised_marker}  db={q['db_id']}")
            print(f"       Q: {q['question'][:80]}")
            if data["ok"]:
                print(f"       SQL: {data['sql'][:100]}")
            else:
                print(f"       ERR: {data.get('error', '?')[:100]}")
        except Exception as exc:  # noqa: BLE001
            print(f"[{i:02d}/{N_QUESTIONS}] ERROR: {exc}", file=sys.stderr)

        print()

    print("=" * 72)
    print(f"Result: {ok_count}/{N_QUESTIONS} answered OK, {revised_count} triggered a revise")
    print(f"Open Langfuse at http://localhost:3001 → filter by tag 'eval_run:{EVAL_RUN_TAG}'")


if __name__ == "__main__":
    main()
