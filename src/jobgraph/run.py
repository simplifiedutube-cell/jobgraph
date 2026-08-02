"""Run orchestration: checkpointing, run directories, and error isolation.

Written for unattended operation. Two rules drive the design:

  * One bad job must never kill a run. Every per-job failure is caught,
    recorded and stepped over.
  * A fatal condition (quota exhausted, kill switch) must stop the *whole*
    run immediately rather than burning through the remaining jobs producing
    identical errors.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncIterator

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from jobgraph import storage
from jobgraph.config import StopRequested, settings
from jobgraph.llm import QuotaExhausted
from jobgraph.schemas import JobPosting
from jobgraph.state import new_application_state


def new_run_id() -> str:
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def run_dir_for(run_id: str, job: JobPosting | None = None) -> Path:
    """Per-job directory inside a run.

    Keyed on the URL, not the company: when a run starts from a bare URL the
    company is still a placeholder at this point, and the directory would be
    named after it.
    """
    base = settings.runs_dir / run_id
    if job is None:
        return base
    key = storage.normalise_url(job.url)
    slug = "".join(c if c.isalnum() else "_" for c in key).strip("_")
    # Keep the tail — the leading host is identical across a whole board.
    return base / slug[-60:].strip("_").lower()


@asynccontextmanager
async def checkpointer() -> AsyncIterator[AsyncSqliteSaver]:
    """Checkpoints live in their own SQLite file — see config.checkpoint_db_path."""
    path = settings.checkpoint_db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        yield saver


class FatalRunError(RuntimeError):
    """Abort the whole run, not just this job."""


async def run_one_application(
    job: JobPosting,
    *,
    run_id: str,
    graph: Any,
    resume: bool = True,
) -> dict[str, Any]:
    """Drive one job through the application subgraph.

    Returns the final state. Raises FatalRunError only for conditions where
    continuing is pointless.
    """
    rd = run_dir_for(run_id, job)
    rd.mkdir(parents=True, exist_ok=True)

    state = new_application_state(job, run_id, str(rd))
    # Thread id keyed on the posting, so an interrupted job resumes where it
    # stopped rather than re-paying for every LLM call.
    config = {
        "configurable": {"thread_id": f"app:{storage.normalise_url(job.url)}"},
        "recursion_limit": 60,
    }

    storage.log(run_id, "info", "job_start", f"{job.company} — {job.title} :: {job.url}")

    try:
        final = await graph.ainvoke(state, config=config)
    except QuotaExhausted as exc:
        storage.log(run_id, "fatal", "quota", str(exc))
        raise FatalRunError(f"quota/auth exhausted: {exc}") from exc
    except StopRequested as exc:
        storage.log(run_id, "fatal", "stopped", str(exc))
        raise FatalRunError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - isolate this job, continue the run
        storage.log(run_id, "error", "job_failed", f"{job.url} :: {type(exc).__name__}: {exc}")
        failed = {**state, "status": "failed", "errors": [f"{type(exc).__name__}: {exc}"]}
        try:
            storage.record_application(failed)
        except Exception:  # noqa: BLE001
            pass
        return failed

    storage.log(
        run_id,
        "info",
        "job_done",
        f"{job.company} [{final.get('status')}] "
        f"fit={getattr(final.get('fit'), 'score', None)}",
    )
    return final


def summarise(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.get("status", "unknown")] = counts.get(r.get("status", "unknown"), 0) + 1
    return counts
