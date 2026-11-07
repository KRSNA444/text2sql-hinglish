"""Convert the official Spider dataset into training / validation / dev JSONL files.

Usage:
    python prepare_data.py --spider-dir spider

Expects this layout inside --spider-dir (from the official Spider download):
    tables.json, train_spider.json, dev.json, database/<db_id>/<db_id>.sqlite

Writes to --out-dir (default: data):
    train.jsonl  chat-format examples (system / user / assistant) for fine-tuning
    val.jsonl    200 held-out train examples, same format, for monitoring training
    dev.jsonl    Spider dev questions with schema + gold SQL, used by eval_exec.py
"""
import argparse
import json
import random
from pathlib import Path

from common import SYSTEM_PROMPT, build_user_prompt, format_schema


def est_tokens(text):
    # rough estimate: ~3.5 characters per token for code-like text
    return int(len(text) / 3.5)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider-dir", required=True)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--max-tokens", type=int, default=1500,
                    help="drop training examples whose estimated length exceeds this")
    ap.add_argument("--val-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.spider_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    schemas = {db["db_id"]: format_schema(db) for db in load(root / "tables.json")}

    # ---- training data ----
    train_rows, dropped = [], 0
    for ex in load(root / "train_spider.json"):
        schema = schemas[ex["db_id"]]
        user = build_user_prompt(schema, ex["question"])
        if est_tokens(SYSTEM_PROMPT + user + ex["query"]) > args.max_tokens:
            dropped += 1
            continue
        train_rows.append({
            "db_id": ex["db_id"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": ex["query"]},
            ],
        })

    random.Random(args.seed).shuffle(train_rows)
    val_rows = train_rows[: args.val_size]
    train_rows = train_rows[args.val_size:]

    # ---- dev data (for evaluation) ----
    dev_rows = []
    for ex in load(root / "dev.json"):
        dev_rows.append({
            "db_id": ex["db_id"],
            "question": ex["question"],
            "gold_sql": ex["query"],
            "schema": schemas[ex["db_id"]],
        })

    for name, rows in (("train", train_rows), ("val", val_rows), ("dev", dev_rows)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"train: {len(train_rows)}  val: {len(val_rows)}  dev: {len(dev_rows)}")
    print(f"dropped {dropped} over-long training examples (> {args.max_tokens} est. tokens)")
    print(f"files written to: {out.resolve()}")


if __name__ == "__main__":
    main()
