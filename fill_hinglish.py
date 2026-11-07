"""Type your own Hinglish versions of 40 questions, one by one (no JSON editing needed).

Usage (from C:\\text2sql, with my_hinglish_test_template.json in the same folder):
    python fill_hinglish.py

For each English question, type it in your own Hinglish words and press Enter.
  - just Enter (empty)  = skip this question
  - q + Enter           = save and quit (run the script again later to continue where you left off)
Your answers are saved after every question into my_hinglish_test.json.
Please do NOT copy from Claude's translations: the point is an independent test set.
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--template", default="my_hinglish_test_template.json")
ap.add_argument("--out", default="my_hinglish_test.json")
args = ap.parse_args()

items = json.load(open(args.template, encoding="utf-8"))
if os.path.exists(args.out):  # resume
    for old in json.load(open(args.out, encoding="utf-8")):
        for it in items:
            if it["idx"] == old["idx"] and old.get("question", "").strip():
                it["question"] = old["question"]


def save():
    json.dump(items, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


todo = [it for it in items if not it["question"].strip()]
done = len(items) - len(todo)
print(f"{len(items)} questions, {done} already done, {len(todo)} to go.")
print("Likho jaise dost ko bolte ho (Roman Hindi + English words). Enter = skip, q = save & quit.\n")

try:
    for n, it in enumerate(todo, 1):
        print(f"[{done + n}/{len(items)}]  database: {it['db_id']}")
        print(f"  English: {it['question_en']}")
        ans = input("  Hinglish: ").strip()
        if ans.lower() == "q":
            break
        if ans:
            it["question"] = ans
            save()
        print()
except (EOFError, KeyboardInterrupt):
    print("\nstopped.")

save()
filled = sum(1 for it in items if it["question"].strip())
print(f"Saved {filled}/{len(items)} answers to {args.out}")
