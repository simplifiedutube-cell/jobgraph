"""SQLite: the application ledger, plus the caps that make unattended runs safe.

The same database file also holds LangGraph's checkpoints (separate tables),
so one file is the entire durable state of the system.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterator

from jobgraph.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    url           TEXT NOT NULL,
    url_key       TEXT NOT NULL,
    company       TEXT NOT NULL,
    title         TEXT NOT NULL,
    board         TEXT,
    ats_vendor    TEXT,
    fit_score     INTEGER,
    status        TEXT NOT NULL,
    outcome       TEXT,
    run_id        TEXT,
    run_dir       TEXT,
    resume_pdf    TEXT,
    notes         TEXT,
    errors        TEXT,
    created_at    TEXT NOT NULL,
    submitted_at  TEXT
);

-- One row per posting, ever. This is what stops the graph applying twice.
CREATE UNIQUE INDEX IF NOT EXISTS idx_applications_url_key
    ON applications(url_key);
CREATE INDEX IF NOT EXISTS idx_applications_status  ON applications(status);
CREATE INDEX IF NOT EXISTS idx_applications_created ON applications(created_at);

CREATE TABLE IF NOT EXISTS run_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    at         TEXT NOT NULL,
    level      TEXT NOT NULL,
    event      TEXT NOT NULL,
    detail     TEXT
);
CREATE INDEX IF NOT EXISTS idx_run_log_run ON run_log(run_id);
"""


def normalise_url(url: str) -> str:
    """Dedupe key. Job URLs pick up tracking params, trailing slashes and
    protocol differences between boards; the same posting must collapse to one
    key or the graph will happily apply twice."""
    u = url.strip().lower()
    for prefix in ("https://", "http://"):
        if u.startswith(prefix):
            u = u[len(prefix) :]
    if u.startswith("www."):
        u = u[4:]
    u = u.split("#", 1)[0]
    u = u.split("?", 1)[0]
    return u.rstrip("/")


_initialised: set[str] = set()


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """A ledger connection.

    LangGraph's checkpointer holds its own connection to this same file for the
    duration of a graph run, so a naive second connection hits
    `database is locked` the moment a node writes. WAL lets readers and a
    writer coexist, and busy_timeout absorbs the brief overlap when the
    checkpointer is mid-commit.
    """
    p = path or settings.db_path
    p.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(p, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")

    try:
        key = str(p)
        if key not in _initialised:
            conn.executescript(SCHEMA)
            _initialised.add(key)
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Path | None = None) -> None:
    with connect(path):
        pass


# --------------------------------------------------------------------------
# Dedupe and caps
# --------------------------------------------------------------------------


def seen_url_keys(path: Path | None = None) -> set[str]:
    with connect(path) as conn:
        return {r["url_key"] for r in conn.execute("SELECT url_key FROM applications")}


def already_applied(url: str, path: Path | None = None) -> bool:
    with connect(path) as conn:
        row = conn.execute(
            "SELECT status FROM applications WHERE url_key = ?", (normalise_url(url),)
        ).fetchone()
    return row is not None and row["status"] in {"submitted", "manual"}


def submissions_today(path: Path | None = None) -> int:
    today = date.today().isoformat()
    with connect(path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM applications "
            "WHERE status = 'submitted' AND substr(submitted_at, 1, 10) = ?",
            (today,),
        ).fetchone()
    return int(row["n"])


def remaining_daily_budget(path: Path | None = None) -> int:
    return max(0, settings.daily_submission_cap - submissions_today(path))


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def record_application(state: dict[str, Any], path: Path | None = None) -> int:
    """Upsert one application. Returns the row id."""
    job = state["job"]
    jd = state.get("jd")
    fit = state.get("fit")
    submission = state.get("submission")
    now = datetime.now(UTC).isoformat()

    submitted_at = None
    if submission is not None and submission.outcome == "submitted":
        submitted_at = now

    payload = (
        job.url,
        normalise_url(job.url),
        job.company,
        job.title,
        str(job.board) if job.board else None,
        str(jd.ats_vendor) if jd else None,
        fit.score if fit else None,
        state.get("status", "scraped"),
        str(submission.outcome) if submission else None,
        state.get("run_id"),
        state.get("run_dir"),
        state.get("resume_pdf"),
        json.dumps(state.get("notes", [])),
        json.dumps(state.get("errors", [])),
        now,
        submitted_at,
    )

    with connect(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO applications
                (url, url_key, company, title, board, ats_vendor, fit_score,
                 status, outcome, run_id, run_dir, resume_pdf, notes, errors,
                 created_at, submitted_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(url_key) DO UPDATE SET
                status       = excluded.status,
                outcome      = excluded.outcome,
                fit_score    = excluded.fit_score,
                ats_vendor   = excluded.ats_vendor,
                run_id       = excluded.run_id,
                run_dir      = excluded.run_dir,
                resume_pdf   = excluded.resume_pdf,
                notes        = excluded.notes,
                errors       = excluded.errors,
                submitted_at = COALESCE(excluded.submitted_at, applications.submitted_at)
            """,
            payload,
        )
        if cur.lastrowid:
            return int(cur.lastrowid)
        row = conn.execute(
            "SELECT id FROM applications WHERE url_key = ?", (normalise_url(job.url),)
        ).fetchone()
        return int(row["id"])


def log(run_id: str, level: str, event: str, detail: str = "", path: Path | None = None) -> None:
    """Append to the run log. This is the file you read in the morning."""
    with connect(path) as conn:
        conn.execute(
            "INSERT INTO run_log (run_id, at, level, event, detail) VALUES (?,?,?,?,?)",
            (run_id, datetime.now(UTC).isoformat(), level, event, detail),
        )


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def recent_applications(limit: int = 25, path: Path | None = None) -> list[dict[str, Any]]:
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM applications ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def status_counts(path: Path | None = None) -> dict[str, int]:
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM applications GROUP BY status"
        ).fetchall()
    return {r["status"]: int(r["n"]) for r in rows}


def run_events(run_id: str, path: Path | None = None) -> list[dict[str, Any]]:
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM run_log WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
    return [dict(r) for r in rows]
