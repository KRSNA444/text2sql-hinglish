"""Compare results files on the SAME first 200 questions.

Run from C:\\text2sql:
    python first200.py                          # table for every results\\*.jsonl
    python first200.py results\\ollama_ft_200.jsonl results\\<hf file>.jsonl   # + examples where they differ
"""
import glob
import json
import sys


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


print(f"{'file':52} {'rows':>5} {'first-200 correct':>18} {'valid':>6}")
for p in sorted(glob.glob("results/*.jsonl")):
    rows = load(p)
    f = [r for r in rows if r["idx"] <= 200]
    if len(f) < 200:
        print(f"{p:52} {len(rows):5}  (fewer than 200 rows)")
        continue
    print(f"{p:52} {len(rows):5} {sum(r['correct'] for r in f):>12}/200 {sum(r['valid'] for r in f):>6}")

if len(sys.argv) == 3:
    a = {r["idx"]: r for r in load(sys.argv[1])}
    b = {r["idx"]: r for r in load(sys.argv[2])}
    diff = [i for i in sorted(a) if i in b and i <= 200 and b[i]["correct"] and not a[i]["correct"]]
    print(f"\n{len(diff)} questions where the 2nd file is correct but the 1st is wrong. First 8:")
    for i in diff[:8]:
        print(f"\n#{i} {a[i]['db_id']} | {a[i]['question']}")
        print("  1st:", a[i]["pred_sql"][:220])
        print("  2nd:", b[i]["pred_sql"][:220])
