"""Learn how the training schema text relates to the real SQLite files (types, primary keys).

Run from C:\\text2sql:   python analyze_schema.py
Needs data/train.jsonl, data/dev.jsonl and spider/database.
"""
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

schemas = {}
for fn in ("data/train.jsonl", "data/dev.jsonl"):
    for line in open(fn, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if "schema" in r:
            sch = r["schema"]
        else:
            user = next(m["content"] for m in r["messages"] if m["role"] == "user")
            sch = user.split("Database schema:\n", 1)[1].split("\n\nQuestion:", 1)[0]
        schemas.setdefault(r["db_id"], sch)


def unq(s):
    s = s.strip()
    return s[1:-1] if len(s) > 1 and s[0] == '"' and s[-1] == '"' else s


def parse(schema):
    out = {}
    for m in re.finditer(r'CREATE TABLE ("[^"]+"|\S+) \(\n(.*?)\n\);', schema, re.S):
        cols, pk = [], []
        for ln in m.group(2).split("\n"):
            ln = ln.strip().rstrip(",")
            if ln.startswith("PRIMARY KEY ("):
                pk = [unq(c) for c in ln[len("PRIMARY KEY ("):-1].split(", ")]
            elif ln.startswith("FOREIGN KEY"):
                pass
            else:
                name, typ = ln.rsplit(" ", 1)
                cols.append((unq(name), typ))
        out[unq(m.group(1)).lower()] = (cols, pk)
    return out


type_map = defaultdict(Counter)
pk_stats = Counter()
pk_examples = []
n_db = 0
for db_id, sch in schemas.items():
    path = Path("spider/database") / db_id / f"{db_id}.sqlite"
    if not path.exists():
        continue
    n_db += 1
    conn = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    parsed = parse(sch)
    for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        if t.lower() not in parsed:
            continue
        cols_t, pk_t = parsed[t.lower()]
        info = conn.execute(f'PRAGMA table_info("{t}")').fetchall()
        for c, (_, tcol) in zip(info, cols_t):
            type_map[(c[2] or "").lower()][tcol] += 1
        sq_pk = [c[1] for c in sorted((c for c in info if c[5]), key=lambda c: c[5])]
        if len(sq_pk) > 1:
            kind = ("composite: training has first column only" if len(pk_t) == 1 else
                    "composite: training has all columns" if len(pk_t) == len(sq_pk) else
                    f"composite: training has {len(pk_t)} of {len(sq_pk)}")
            pk_stats[kind] += 1
            if len(pk_examples) < 3 and kind.endswith("first column only"):
                pk_examples.append((db_id, t, sq_pk, pk_t))
        elif len(sq_pk) == 1:
            pk_stats["single PK: same in training" if pk_t == sq_pk else "single PK: differs"] += 1
        else:
            pk_stats["no PK in sqlite; training has PK" if pk_t else "no PK anywhere"] += 1

print(f"databases analysed: {n_db}\n\nSQLite declared type -> type shown in training (counts):")
for decl, c in sorted(type_map.items(), key=lambda kv: -sum(kv[1].values())):
    print(f"  {decl or '(empty)':18} -> {dict(c)}")
print("\nPrimary keys:")
for k, v in pk_stats.items():
    print(f"  {v:4}  {k}")
for e in pk_examples:
    print("  example:", e)
