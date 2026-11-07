"""Shared helpers: schema formatting, prompt building, SQL extraction.

The SAME prompt format is used for training data, baseline evaluation and the final app,
so the fine-tuned model sees exactly what it was trained on.
"""
import re

SYSTEM_PROMPT = (
    "You are a Text-to-SQL assistant. Given a SQLite database schema and a question "
    "(English or Hinglish), reply with a single SQLite SELECT query and nothing else."
)


def _q(name):
    """Quote identifiers that contain spaces or special characters."""
    return name if re.fullmatch(r"\w+", name) else f'"{name}"'


def format_schema(db):
    """Turn one entry of Spider's tables.json into CREATE TABLE statements."""
    tables = db["table_names_original"]
    cols = db["column_names_original"]
    types = db["column_types"]

    pks = set()
    for pk in db.get("primary_keys", []):
        if isinstance(pk, list):
            pks.update(pk)
        else:
            pks.add(pk)

    by_table = {i: [] for i in range(len(tables))}
    for ci, (ti, name) in enumerate(cols):
        if ti < 0:  # the "*" pseudo-column
            continue
        by_table[ti].append((ci, name, types[ci]))

    fks = {a: b for a, b in db.get("foreign_keys", [])}

    out = []
    for ti, tname in enumerate(tables):
        parts = [f"{_q(n)} {t}" for _, n, t in by_table[ti]]
        pk_cols = [_q(n) for ci, n, _ in by_table[ti] if ci in pks]
        if pk_cols:
            parts.append(f"PRIMARY KEY ({', '.join(pk_cols)})")
        for ci, n, _ in by_table[ti]:
            if ci in fks:
                rti, rname = cols[fks[ci]]
                parts.append(
                    f"FOREIGN KEY ({_q(n)}) REFERENCES {_q(tables[rti])}({_q(rname)})"
                )
        out.append(f"CREATE TABLE {_q(tname)} (\n  " + ",\n  ".join(parts) + "\n);")
    return "\n".join(out)


def build_user_prompt(schema, question):
    return f"Database schema:\n{schema}\n\nQuestion: {question}\n\nSQL:"


def extract_sql(text):
    """Pull a single SQL statement out of a model reply."""
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    if m:
        text = m.group(1)
    text = text.strip()
    if ";" in text:
        text = text.split(";")[0]
    return " ".join(text.split())
