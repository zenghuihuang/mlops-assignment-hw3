"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    db_id = question["db_id"]

    # Gold answer — what we're comparing against.
    _, gold_rows, _ = run_sql(db_id, question["gold_sql"])

    # Call the agent.
    try:
        resp = httpx.post(
            agent_url,
            json={
                "question": question["question"],
                "db": db_id,
                "tags": {"eval_run": "baseline"},
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        return {
            "question": question["question"],
            "db_id": db_id,
            "error": str(exc),
            "iterations": 0,
            "per_iter_correct": {},
            "final_correct": False,
        }

    # Walk the history to collect the SQL produced at each generate/revise step.
    # history entry nodes: "generate_sql", "verify", "revise"
    history = data.get("history", [])
    sql_per_iter = [
        entry["sql"]
        for entry in history
        if entry.get("node") in ("generate_sql", "revise")
    ]

    # Score each iteration against the gold rows.
    per_iter_correct: dict[int, bool] = {}
    for i, sql in enumerate(sql_per_iter):
        _, pred_rows, _ = run_sql(db_id, sql)
        per_iter_correct[i] = matches(gold_rows, pred_rows)

    final_correct = per_iter_correct.get(len(sql_per_iter) - 1, False) if sql_per_iter else False

    return {
        "question": question["question"],
        "db_id": db_id,
        "gold_sql": question["gold_sql"],
        "agent_sql": data.get("sql", ""),
        "agent_ok": data.get("ok", False),
        "iterations": data.get("iterations", 0),
        "per_iter_correct": per_iter_correct,
        "final_correct": final_correct,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    if not results:
        return {"n": 0, "overall_accuracy": 0.0, "per_iteration_accuracy": {}}

    # JSON round-trips turn int keys into strings; normalise to int here.
    def _int_keys(d: dict) -> dict[int, bool]:
        return {int(k): v for k, v in d.items()}

    # How many iteration slots do we need?
    max_iter = max(
        (max(_int_keys(r["per_iter_correct"]).keys(), default=-1) + 1
         for r in results),
        default=0,
    )

    per_iter_accuracy: dict[str, float] = {}
    for k in range(max_iter):
        correct = 0
        for r in results:
            pc = _int_keys(r["per_iter_correct"])
            if not pc:
                # Agent failed entirely — not correct at any iteration.
                continue
            # Carry-forward: use the last result at or before k.
            available = [i for i in pc if i <= k]
            if available and pc[max(available)]:
                correct += 1
        per_iter_accuracy[f"iter_{k}"] = round(correct / len(results), 4)

    overall = sum(1 for r in results if r.get("final_correct", False)) / len(results)

    return {
        "n": len(results),
        "overall_accuracy": round(overall, 4),
        "per_iteration_accuracy": per_iter_accuracy,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
