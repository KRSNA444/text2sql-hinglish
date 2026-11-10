"""Show what goes wrong in an eval_exec.py results file.

Run from C:\\text2sql:   python show_errors.py results\\ollama_ft_200.jsonl
"""
import json
import sys
from collections import Counter

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
bad = [r for r in rows if not r["valid"]]
print(f"{len(rows)} questions, {len(bad)} invalid SQL, {sum(r['correct'] for r in rows)} correct\n")
print("Error kinds:")
for k, v in Counter(r["error"].split(":")[0] + ": " + r["error"].split(":", 1)[-1].strip()[:45] for r in bad).most_common(8):
    print(f"  {v:3}  {k}")
print("\nFirst 8 invalid predictions (repr, so hidden characters show up):")
for r in bad[:8]:
    print(f"\n#{r['idx']} {r['db_id']} | Q: {r['question']}")
    print("  pred:", repr(r["pred_sql"])[:300])
    print("  gold:", r["gold_sql"][:200])
    print("  err :", r["error"][:150])
