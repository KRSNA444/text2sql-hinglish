"""Execution-accuracy evaluation for Text-to-SQL, using any model served by Ollama.

Usage:
    python eval_exec.py --model qwen2.5-coder:3b --limit 50      # quick smoke test
    python eval_exec.py --model qwen2.5-coder:3b                 # full dev set (baseline)
    python eval_exec.py --model my-sql-model --out results/finetuned.jsonl

For each question the model's SQL and the gold SQL are both run (read-only) on the
SQLite database. A prediction is correct if both return the same rows.

Note: this checker is a little stricter than the official Spider evaluation because it
requires the same column order, so scores can be slightly lower than official numbers.
State this in your README when you report results.
"""
import argparse
import json
import sqlite3
import time
import urllib.request
from collections import Counter
from pathlib import Path

from common import SYSTEM_PROMPT, build_user_prompt, extract_sql


def run_sql(db_path, sql, timeout=10.0):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)  # read-only
    conn.text_factory = lambda b: b.decode("utf-8", errors="ignore")
    deadline = time.time() + timeout
    conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10000)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def same_result(gold_rows, pred_rows, ordered):
    if ordered:
        return gold_rows == pred_rows
    return Counter(gold_rows) == Counter(pred_rows)


def ask_model(host, model, schema, question, timeout=120):
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(schema, question)},
        ],
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 300},
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-coder:3b")
    ap.add_argument("--data", default="data/dev.jsonl")
    ap.add_argument("--db-dir", default="spider/database")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=0, help="0 = all examples")
    ap.add_argument("--out", default="results/baseline.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_valid = n_correct = 0
    t0 = time.time()

    with open(args.out, "w", encoding="utf-8") as fout:
        for i, ex in enumerate(rows, 1):
            db_path = Path(args.db_dir) / ex["db_id"] / f"{ex['db_id']}.sqlite"
            raw = ask_model(args.host, args.model, ex["schema"], ex["question"])
            pred = extract_sql(raw)

            valid = correct = False
            error = ""
            try:
                pred_rows = run_sql(db_path, pred)
                valid = True
                gold_rows = run_sql(db_path, ex["gold_sql"])
                ordered = "order by" in ex["gold_sql"].lower()
                correct = same_result(gold_rows, pred_rows, ordered)
            except Exception as e:  # invalid SQL, timeout, missing db ...
                error = f"{type(e).__name__}: {e}"

            n_valid += valid
            n_correct += correct
            fout.write(json.dumps({
                "idx": i, "db_id": ex["db_id"], "question": ex["question"],
                "gold_sql": ex["gold_sql"], "pred_sql": pred,
                "valid": valid, "correct": correct, "error": error,
            }, ensure_ascii=False) + "\n")

            if i % 20 == 0 or i == len(rows):
                print(f"[{i}/{len(rows)}] exec acc so far: {n_correct / i:.1%}", flush=True)

    n = len(rows)
    summary = {
        "model": args.model, "data": args.data, "examples": n,
        "valid_sql_rate": round(n_valid / n, 4),
        "execution_accuracy": round(n_correct / n, 4),
        "seconds": round(time.time() - t0, 1),
    }
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
