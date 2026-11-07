"""Pick 600 random training questions (with their gold SQL) so they can be translated to Hinglish.

Usage (from C:\\text2sql):
    python sample_train_questions.py
Output: hinglish_train_input.json   (small file; send this one to Claude)

Only rows from data/train.jsonl are used (Spider train split), never the dev set.
"""
import argparse
import json
import random
import re

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="data/train.jsonl")
ap.add_argument("--out", default="hinglish_train_input.json")
ap.add_argument("--n", type=int, default=600)
ap.add_argument("--seed", type=int, default=11)
args = ap.parse_args()

rows = [json.loads(l) for l in open(args.train, encoding="utf-8") if l.strip()]
pat = re.compile(r"\n\nQuestion: (.*)\n\nSQL:$", re.S)

items = []
for i, r in enumerate(rows):
    user = r["messages"][1]["content"]
    m = pat.search(user)
    if not m:
        continue
    items.append({"i": i, "db_id": r["db_id"], "question_en": m.group(1),
                  "sql": r["messages"][2]["content"]})

random.Random(args.seed).shuffle(items)
items = sorted(items[: args.n], key=lambda x: x["i"])
json.dump(items, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"wrote {len(items)} questions to {args.out}")
