"""Re-ask only the questions that were wrong in an eval_exec.py results file, with different Ollama settings.

Run from C:\\text2sql, e.g.:
    python retest_errors.py results\\ollama_ft_200.jsonl --repeat-penalty 1.0
"""
import argparse
import json
import urllib.request
from pathlib import Path

from common import SYSTEM_PROMPT, build_user_prompt, extract_sql
from eval_exec import run_sql, same_result

ap = argparse.ArgumentParser()
ap.add_argument("results")
ap.add_argument("--model", default="text2sql-hinglish")
ap.add_argument("--data", default="data/dev.jsonl")
ap.add_argument("--db-dir", default="spider/database")
ap.add_argument("--host", default="http://localhost:11434")
ap.add_argument("--repeat-penalty", type=float, default=1.0)
args = ap.parse_args()

dev = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
res = [json.loads(l) for l in open(args.results, encoding="utf-8") if l.strip()]
wrong = [r for r in res if not r["correct"]]
print(f"{len(wrong)} wrong of {len(res)}; re-asking them with repeat_penalty={args.repeat_penalty} ...", flush=True)

fixed = now_valid = 0
for n, r in enumerate(wrong, 1):
    ex = dev[r["idx"] - 1]
    payload = {"model": args.model, "stream": False,
               "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": build_user_prompt(ex["schema"], ex["question"])}],
               "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 300,
                           "repeat_penalty": args.repeat_penalty}}
    req = urllib.request.Request(f"{args.host}/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    pred = extract_sql(json.loads(urllib.request.urlopen(req, timeout=120).read())["message"]["content"])
    db = Path(args.db_dir) / ex["db_id"] / f"{ex['db_id']}.sqlite"
    try:
        p = run_sql(db, pred)
        now_valid += 1
        ok = same_result(run_sql(db, ex["gold_sql"]), p, "order by" in ex["gold_sql"].lower())
    except Exception:
        ok = False
    fixed += ok
    if n % 10 == 0 or n == len(wrong):
        print(f"[{n}/{len(wrong)}] fixed so far: {fixed}, now valid SQL: {now_valid}", flush=True)

total = len(res)
c_old = sum(r["correct"] for r in res)
print(f"\nBefore: {c_old}/{total} correct. After this setting: {c_old + fixed}/{total} correct "
      f"= {(c_old + fixed) / total:.1%} (the previously-correct ones are assumed to stay correct).")
