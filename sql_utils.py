"""Helpers for the Text-to-SQL app: schema text, SELECT-only guard, safe execution, Ollama calls.

The schema text is written in the SAME format that the model was fine-tuned on
(CREATE TABLE ... with 'text'/'number' types, PRIMARY KEY and FOREIGN KEY lines).
"""
import io
import json
import re
import sqlite3
import time
import urllib.request
from pathlib import Path

from common import SYSTEM_PROMPT, build_user_prompt, extract_sql


# ----------------------------------------------------------------------------- schema
def _q(name):
    """Quote identifiers that contain spaces or special characters (same rule as training)."""
    return name if re.fullmatch(r"\w+", name) else f'"{name}"'


def _lit(name):
    """Identifier as a quoted literal for PRAGMA calls."""
    return '"' + name.replace('"', '""') + '"'


def _col_type(sqlite_type):
    t = (sqlite_type or "").lower()
    return "number" if any(k in t for k in ("int", "real", "floa", "doub", "num", "dec")) else "text"


def schema_from_connection(conn):
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    pk_of = {}
    info = {}
    for t in tables:
        cols = conn.execute(f"PRAGMA table_info({_lit(t)})").fetchall()  # cid,name,type,notnull,dflt,pk
        info[t] = cols
        pk_of[t] = [c[1] for c in sorted((c for c in cols if c[5]), key=lambda c: c[5])]

    out = []
    for t in tables:
        parts = [f"{_q(c[1])} {_col_type(c[2])}" for c in info[t]]
        if pk_of[t]:
            parts.append("PRIMARY KEY (" + ", ".join(_q(c) for c in pk_of[t]) + ")")
        for fk in conn.execute(f"PRAGMA foreign_key_list({_lit(t)})").fetchall():
            _id, seq, ref_table, col_from, col_to = fk[0], fk[1], fk[2], fk[3], fk[4]
            if not col_to:  # reference to the primary key of the other table
                refpk = pk_of.get(ref_table, [])
                col_to = refpk[seq] if seq < len(refpk) else ""
            if col_to:
                parts.append(f"FOREIGN KEY ({_q(col_from)}) REFERENCES {_q(ref_table)}({_q(col_to)})")
        out.append(f"CREATE TABLE {_q(t)} (\n  " + ",\n  ".join(parts) + "\n);")
    return "\n".join(out)


# ----------------------------------------------------------------------------- connections
def open_readonly(path):
    """Open an SQLite file read-only."""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.text_factory = lambda b: b.decode("utf-8", errors="ignore")
    conn.execute("PRAGMA query_only = ON")
    return conn


def memory_db_from_csvs(named_bytes):
    """named_bytes: list of (filename, bytes). Each CSV becomes a table in an in-memory database."""
    import pandas as pd

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    for fname, data in named_bytes:
        stem = re.sub(r"\W+", "_", Path(fname).stem).strip("_") or "table1"
        if stem[0].isdigit():
            stem = "t_" + stem
        pd.read_csv(io.BytesIO(data)).to_sql(stem, conn, index=False, if_exists="replace")
    conn.commit()
    conn.execute("PRAGMA query_only = ON")  # no writes after the data is loaded
    return conn


# ----------------------------------------------------------------------------- guard
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|reindex)\b", re.I)


def _strip_comments(sql):
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return re.sub(r"--[^\n]*", " ", sql)


def _strip_literals(sql):
    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)
    return re.sub(r'"(?:[^"]|"")*"', '""', sql)


def check_select_only(sql):
    """Return (True, cleaned_sql) if sql is a single read-only SELECT, else (False, reason)."""
    cleaned = _strip_comments(sql or "").strip()
    cleaned = cleaned.rstrip(";").strip()
    if not cleaned:
        return False, "empty query"
    probe = _strip_literals(cleaned)
    if ";" in probe:
        return False, "only one statement is allowed"
    if not re.match(r"(?is)^\s*(select|with)\b", probe):
        return False, "only SELECT queries are allowed"
    bad = _FORBIDDEN.search(probe)
    if bad:
        return False, f"keyword '{bad.group(0)}' is not allowed"
    return True, cleaned


def run_query(conn, sql, max_rows=200, timeout=10.0):
    deadline = time.time() + timeout
    conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10000)
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(max_rows + 1)
        return cols, rows[:max_rows], len(rows) > max_rows
    finally:
        conn.set_progress_handler(None, 0)


# ----------------------------------------------------------------------------- model
def chat(host, model, messages, timeout=180):
    payload = {
        "model": model, "stream": False, "messages": messages,
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 300},
    }
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["message"]["content"]


def text_to_sql(conn, schema, question, host, model, retry=True):
    """Generate SQL, run it safely, and (optionally) retry once with the error message.

    Returns a list of attempts: {sql, error, columns, rows, truncated}.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(schema, question)},
    ]
    attempts = []
    for _ in range(2 if retry else 1):
        sql = extract_sql(chat(host, model, messages))
        entry = {"sql": sql, "error": "", "columns": [], "rows": [], "truncated": False}
        ok, info = check_select_only(sql)
        if not ok:
            entry["error"] = f"Blocked: {info}"
        else:
            try:
                cols, rows, truncated = run_query(conn, info)
                entry.update(columns=cols, rows=rows, truncated=truncated)
            except Exception as e:  # invalid SQL, timeout, ...
                entry["error"] = f"{type(e).__name__}: {e}"
        attempts.append(entry)
        if not entry["error"]:
            break
        messages += [
            {"role": "assistant", "content": sql},
            {"role": "user", "content": f"That query failed with this error: {entry['error']}\n"
                                        "Return a corrected SQLite SELECT query only."},
        ]
    return attempts
