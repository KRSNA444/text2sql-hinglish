"""Build data/train_mixed.jsonl = original English training data + Hinglish versions.

Each Hinglish example copies a training row, replaces only the question text, and keeps the
same schema and the same gold SQL. A safety check makes sure the English question in the
JSON matches the row it points to.

Usage (from C:\\text2sql):
    python merge_hinglish_train.py
Then train:
    python train_qlora.py --train data/train_mixed.jsonl --epochs 1 --out outputs/lora_sql_hinglish
"""
import argparse
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="data/train.jsonl")
ap.add_argument("--hinglish", default="hinglish_train_questions.json")
ap.add_argument("--out", default="data/train_mixed.jsonl")
args = ap.parse_args()

rows = [json.loads(l) for l in open(args.train, encoding="utf-8") if l.strip()]
items = json.load(open(args.hinglish, encoding="utf-8"))
pat = re.compile(r"\n\nQuestion: (.*)\n\nSQL:$", re.S)

extra = []
for it in items:
    if not it["question"].strip():
        continue
    row = rows[it["i"]]
    user = row["messages"][1]["content"]
    m = pat.search(user)
    if not m or m.group(1) != it["question_en"] or row["db_id"] != it["db_id"]:
        raise SystemExit(f"Mismatch at row {it['i']}: train.jsonl does not match the questions file")
    new_user = user[: m.start(1)] + it["question"] + user[m.end(1):]
    new_row = json.loads(json.dumps(row))
    new_row["messages"][1]["content"] = new_user
    extra.append(new_row)

with open(args.out, "w", encoding="utf-8") as f:
    for r in rows + extra:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{len(rows)} English + {len(extra)} Hinglish = {len(rows) + len(extra)} examples -> {args.out}")
