from langchain_community.utilities import SQLDatabase
from dataclasses import dataclass
from langgraph.runtime import get_runtime
from langchain.tools import tool
from typing import Optional, Callable
import re


@dataclass
class RuntimeContext:
    db: SQLDatabase


DENY_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|ALTER|DROP|CREATE|REPLACE|TRUNCATE)\b", re.I
)
HAS_LIMIT_TAIL_RE = re.compile(r"(?is)\blimit\b\s+\d+(\s*,\s*\d+)?\s*;?\s*$")


def _safe_sql(q: str) -> str:
    q = q.strip()
    if q.count(";") > 1 or (q.endswith(";") and ";" in q[:-1]):
        return "Error: multiple statements are not allowed."
    q = q.rstrip(";").strip()
    if not q.lower().startswith("select"):
        return "Error: only SELECT statements are allowed."
    if DENY_RE.search(q):
        return "Error: DML/DDL detected. Only read-only queries are permitted."
    if not HAS_LIMIT_TAIL_RE.search(q):
        q += " LIMIT 5"
    return q


def sql_factory(db: Optional[SQLDatabase] = None) -> Callable:
    """
    Create a SQL execution tool, either offline with a provided database
    or runtime with a database from the runtime context.
    """

    if db is not None:
        # -------- OFFLINE TOOL --------
        @tool
        def execute_sql(query: str) -> str:
            """Execute a SQL query on the provided offline database."""
            q = _safe_sql(query)
            try:
                return str(db.run(q))
            except Exception as e:
                return f"Error: {e}"

        return execute_sql

    else:
        # -------- RUNTIME TOOL --------
        @tool
        def execute_sql(query: str) -> str:
            """Execute a SQL query using runtime context database."""
            q = _safe_sql(query)
            runtime = get_runtime(RuntimeContext)
            rdb = runtime.context.db
            try:
                return str(rdb.run(q))
            except Exception as e:
                return f"Error: {e}"

        return execute_sql
