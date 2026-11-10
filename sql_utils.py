"""Helpers for the Text-to-SQL app: schema text, SELECT-only guard, safe execution, Ollama calls.

The schema text is written in the SAME format that the model was fine-tuned on
(CREATE TABLE ... with 'text'/'number' types, PRIMARY KEY and FOREIGN KEY lines).
"""
import io
import json
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from common import SYSTEM_PROMPT, build_user_prompt

_DB_LOCK = threading.Lock()  # one shared connection, several browser sessions


# ----------------------------------------------------------------------------- schema
def _q(name):
    """Quote identifiers that contain spaces or special characters (same rule as training)."""
    return name if re.fullmatch(r"\w+", name) else f'"{name}"'


def _lit(name):
    """Identifier as a quoted literal for PRAGMA calls."""
    return '"' + name.replace('"', '""') + '"'


def _col_type(sqlite_type):
    """Map a declared SQLite type to the type words used in training (Spider's text/number/time/boolean/others).

    Rules measured on all 158 Spider databases (see analyze_schema.py).
    """
    t = (sqlite_type or "").lower().strip()
    if t == "boolean":
        return "boolean"
    if t == "bit" or t.startswith(("bool", "blob")):
        return "others"
    if any(k in t for k in ("date", "time", "year")):
        return "time"
    if any(k in t for k in ("char", "text", "clob")) or not t:
        return "text"
    if any(k in t for k in ("int", "real", "floa", "doub", "num", "dec")):
        return "number"
    return "text"


def schema_from_connection(conn):
    """CREATE TABLE text in the same format the model was trained on.

    Matches the training data: tables/columns in creation order, only the FIRST column of a
    composite primary key, foreign keys listed in column order.
    """
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]  # creation order
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
            parts.append("PRIMARY KEY (" + _q(pk_of[t][0]) + ")")
        fk_of = {}  # column -> (referenced table, referenced column); the last one wins, like in training
        for fk in conn.execute(f"PRAGMA foreign_key_list({_lit(t)})").fetchall():
            _id, seq, ref_table, col_from, col_to = fk[0], fk[1], fk[2], fk[3], fk[4]
            if not col_to:  # reference to the primary key of the other table
                refpk = pk_of.get(ref_table, [])
                col_to = refpk[seq] if seq < len(refpk) else ""
            if col_to:
                fk_of[col_from] = (ref_table, col_to)
        for c in info[t]:
            if c[1] in fk_of:
                ref_table, col_to = fk_of[c[1]]
                parts.append(f"FOREIGN KEY ({_q(c[1])}) REFERENCES {_q(ref_table)}({_q(col_to)})")
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


def _read_csv_bytes(fname, data):
    """Read CSV bytes: tries utf-8 / Excel (cp1252) encodings and comma / semicolon / tab / pipe separators."""
    import pandas as pd

    if not data.strip():
        raise ValueError(f"'{fname}' khaali hai.")
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):  # Excel "Unicode text"
        text = data.decode("utf-16")
    elif b"\x00" in data[:2000]:
        raise ValueError(f"'{fname}' CSV file jaisi nahi lagti (binary data hai).")
    else:
        text = None
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
    header = text.lstrip("\ufeff").split("\n", 1)[0]
    sep = max((",", ";", "\t", "|"), key=header.count)  # most frequent in the header line
    if header.count(sep) == 0:
        sep = ","
    df = pd.read_csv(io.StringIO(text), sep=sep)
    if df.shape[1] == 0:
        raise ValueError(f"'{fname}' me koi column nahi mila.")
    return df


def memory_db_from_csvs(named_bytes):
    """named_bytes: list of (filename, bytes). Each CSV becomes a table in an in-memory database."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    used = set()
    for fname, data in named_bytes:
        stem = re.sub(r"\W+", "_", Path(fname).stem).strip("_") or "table1"
        if stem[0].isdigit():
            stem = "t_" + stem
        base, n = stem, 2
        while stem.lower() in used:  # two files with the same name
            stem, n = f"{base}_{n}", n + 1
        used.add(stem.lower())
        _read_csv_bytes(fname, data).to_sql(stem, conn, index=False, if_exists="replace")
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
    with _DB_LOCK:
        deadline = time.time() + timeout
        conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10000)
        try:
            cur = conn.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(max_rows + 1)
            return cols, rows[:max_rows], len(rows) > max_rows
        finally:
            conn.set_progress_handler(None, 0)


def display_df(columns, rows):
    """DataFrame that Streamlit can always show: unique column names, no raw bytes / huge ints."""
    import pandas as pd

    seen, names = {}, []
    for c in columns:
        c = str(c)
        seen[c] = seen.get(c, 0) + 1
        names.append(c if seen[c] == 1 else f"{c}_{seen[c]}")

    def fix(v):
        if isinstance(v, (bytes, bytearray)):
            return v.hex()
        if isinstance(v, int) and abs(v) >= 2 ** 63:
            return str(v)
        return v

    return pd.DataFrame([[fix(v) for v in r] for r in rows], columns=names)


# ----------------------------------------------------------------------------- model
def chat(host, model, messages, timeout=180):
    payload = {
        "model": model, "stream": False, "messages": messages,
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 300,
                    "repeat_penalty": 1.0},  # no penalty, like the HF evaluation
    }
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        try:
            body = json.loads(body).get("error", body)
        except Exception:
            pass
        hint = " Model install nahi hai: `ollama list` se naam check karo." if e.code == 404 else ""
        raise RuntimeError(f"Ollama ne error diya ({e.code}): {body}.{hint}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama se connect nahi hua ({host}): {e.reason}. "
                           "Ollama chal raha hai?") from None
    except TimeoutError:
        raise RuntimeError("Model ne time par jawab nahi diya. Dobara try karo.") from None


def _first_statement(text):
    """Cut at the first ';' that is not inside quotes."""
    quote = None
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"', "`"):
            quote = ch
        elif ch == ";":
            return text[:i]
    return text


def clean_model_sql(text):
    """Pull one SQL statement out of a model reply (code fences, extra text after ';')."""
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    if m:
        text = m.group(1)
    return _first_statement(text.strip()).strip()


def prompt_may_be_too_long(schema, question, max_tokens=4096, reserve=400):
    """Rough check (about 3 characters per token) that schema + question fit the model's context."""
    return (len(SYSTEM_PROMPT) + len(schema) + len(question) + 60) / 3 > max_tokens - reserve


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
        sql = clean_model_sql(chat(host, model, messages))
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
