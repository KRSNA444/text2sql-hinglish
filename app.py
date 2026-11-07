"""Streamlit demo: ask a question in English or Hinglish, get SQL + results.

Run (from C:\\text2sql):
    python -m streamlit run app.py
Needs: Ollama running with your model, and `pip install streamlit pandas`.
"""
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from sql_utils import memory_db_from_csvs, open_readonly, schema_from_connection, text_to_sql

st.set_page_config(page_title="Text-to-SQL (English + Hinglish)", page_icon="🗄️", layout="wide")
st.title("Text-to-SQL assistant")
st.caption("English ya Hinglish me sawaal poochho. Sirf SELECT queries chalti hain, "
           "aur database read-only khulta hai.")


@st.cache_resource(show_spinner=False)
def db_from_sqlite_bytes(name: str, data: bytes):
    path = Path(tempfile.mkdtemp()) / name
    path.write_bytes(data)
    return open_readonly(path)


@st.cache_resource(show_spinner=False)
def db_from_csvs(files: tuple):
    return memory_db_from_csvs(list(files))


@st.cache_resource(show_spinner=False)
def db_from_path(path: str):
    return open_readonly(path)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Model")
    host = st.text_input("Ollama host", "http://localhost:11434")
    model = st.text_input("Model name", "qwen2.5-coder:3b",
                          help="Fine-tuned model ka naam yahan daalo (jaise text2sql-hinglish).")
    retry = st.checkbox("Error aane par ek baar retry karo", value=True)
    st.divider()
    st.header("Database")
    source = st.radio("Source", ["Upload SQLite / CSV", "Spider database folder"])

conn = None
if source == "Upload SQLite / CSV":
    files = st.file_uploader("SQLite file (.sqlite/.db) ya ek ya zyada CSV files",
                             type=["sqlite", "sqlite3", "db", "csv"], accept_multiple_files=True)
    sqlite_files = [f for f in files if f.name.lower().endswith((".sqlite", ".sqlite3", ".db"))]
    csv_files = [f for f in files if f.name.lower().endswith(".csv")]
    try:
        if sqlite_files:
            conn = db_from_sqlite_bytes(sqlite_files[0].name, sqlite_files[0].getvalue())
        elif csv_files:
            conn = db_from_csvs(tuple((f.name, f.getvalue()) for f in csv_files))
    except Exception as e:
        st.error(f"Database load nahi hua: {e}")
else:
    root = Path(st.sidebar.text_input("Spider database folder", "spider/database"))
    names = sorted(p.name for p in root.glob("*") if (p / f"{p.name}.sqlite").exists()) if root.exists() else []
    if names:
        choice = st.sidebar.selectbox("Database", names, index=names.index("concert_singer") if "concert_singer" in names else 0)
        conn = db_from_path(str(root / choice / f"{choice}.sqlite"))
    else:
        st.sidebar.warning("Is folder me koi database nahi mila.")

if conn is None:
    st.info("Pehle ek database chuno ya upload karo.")
    st.stop()

schema = schema_from_connection(conn)
with st.expander("Database schema (model ko ye dikhaya jata hai)"):
    st.code(schema, language="sql")

# ------------------------------------------------------------------ question
question = st.text_input("Sawaal", placeholder="Har singer ka naam aur age batao, age ke descending order me")
if st.button("SQL banao", type="primary") and question.strip():
    try:
        with st.spinner("Model soch raha hai..."):
            attempts = text_to_sql(conn, schema, question.strip(), host, model, retry=retry)
    except Exception as e:
        st.error(f"Model se baat nahi ho paayi: {e}\n\nCheck karo ki Ollama chal raha hai aur "
                 f"model '{model}' installed hai.")
        st.stop()

    for i, a in enumerate(attempts, 1):
        if len(attempts) > 1:
            st.markdown(f"**Attempt {i}**" + (" (retry)" if i > 1 else ""))
        st.code(a["sql"], language="sql")
        if a["error"]:
            st.warning(a["error"])
        else:
            df = pd.DataFrame(a["rows"], columns=a["columns"])
            st.dataframe(df)
            st.caption(f"{len(df)} rows" + (" (pehli 200 dikhayi gayi)" if a["truncated"] else ""))
