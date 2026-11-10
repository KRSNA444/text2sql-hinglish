"""Check that the app's schema text matches the schema text the model was trained/evaluated on.

Run from C:\\text2sql (needs data/dev.jsonl and spider/database from earlier steps):
    python check_schema.py
"""
import json
from pathlib import Path

from sql_utils import open_readonly, schema_from_connection

rows = [json.loads(l) for l in open("data/dev.jsonl", encoding="utf-8") if l.strip()]
by_db = {}
for r in rows:
    by_db.setdefault(r["db_id"], r["schema"])

same, diff = 0, []
for db_id, trained in sorted(by_db.items()):
    path = Path("spider/database") / db_id / f"{db_id}.sqlite"
    got = schema_from_connection(open_readonly(path))
    if got == trained:
        same += 1
    else:
        diff.append((db_id, trained, got))

print(f"{same}/{len(by_db)} databases give EXACTLY the same schema text as in evaluation.")
for db_id, trained, got in diff[:3]:
    print(f"\n=== {db_id}: differs ===\n--- evaluation/training format:\n{trained}\n--- app format:\n{got}")
if diff:
    print("\nNames of databases that differ:", [d[0] for d in diff])
