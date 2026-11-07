"""Build a Hinglish evaluation file from a questions JSON + data/dev.jsonl.

Each Hinglish question reuses the schema and gold SQL of the Spider dev question it was
translated from. A safety check makes sure the English question in the JSON matches
the dev row, so nothing is misaligned. Rows with an empty "question" are skipped.

Usage (from C:\\text2sql):
    python make_hinglish.py                                   # the 200-question set
    python make_hinglish.py --questions my_hinglish_test.json --out data/hinglish_mine.jsonl
Then evaluate:
    python eval_hf.py --data data/hinglish.jsonl --out results/hinglish_base.jsonl
    python eval_hf.py --data data/hinglish.jsonl --adapter outputs/lora_sql --out results/hinglish_finetuned.jsonl
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--questions", default="hinglish_questions.json")
ap.add_argument("--dev", default="data/dev.jsonl")
ap.add_argument("--out", default="data/hinglish.jsonl")
args = ap.parse_args()

dev = [json.loads(l) for l in open(args.dev, encoding="utf-8") if l.strip()]
items = json.load(open(args.questions, encoding="utf-8"))

n = skipped = 0
with open(args.out, "w", encoding="utf-8") as f:
    for it in items:
        if not it["question"].strip():
            skipped += 1
            continue
        row = dev[it["idx"] - 1]  # idx is the 1-based position in dev.jsonl
        if row["question"] != it["question_en"] or row["db_id"] != it["db_id"]:
            raise SystemExit(f"Mismatch at idx {it['idx']}: dev.jsonl does not match {args.questions}")
        f.write(json.dumps({
            "db_id": row["db_id"],
            "question": it["question"],
            "question_en": row["question"],
            "gold_sql": row["gold_sql"],
            "schema": row["schema"],
        }, ensure_ascii=False) + "\n")
        n += 1
print(f"wrote {n} Hinglish examples to {args.out}" + (f" (skipped {skipped} empty)" if skipped else ""))
