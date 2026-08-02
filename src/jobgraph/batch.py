"""Discover, then run each surviving posting through the application graph.

This is the unattended entry point, so the failure semantics are the whole
point of the module:

  * A per-job failure is caught, recorded, and stepped over. One posting with
    a broken form must not cost you the other nine.
  * A fatal condition — quota exhausted, kill switch, auth gone — stops the
    whole batch immediately. Continuing would produce N identical failures and
    burn the rest of the night.
  * Every run writes a summary you can read in the morning without opening
    SQLite.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jobgraph import storage
from jobgraph.config import settings
from jobgraph.graph_application import compile_application_graph
from jobgraph.llm import TELEMETRY
from jobgraph.nodes.discover import discover_listings, filter_and_select
from jobgraph.run import FatalRunError, checkpointer, new_run_id, run_one_application
from jobgraph.schemas import JobPosting
from jobgraph.state import DiscoveryState


async def discover(
    limit: int | None = None, *, source: str = "boards"
) -> tuple[list[JobPosting], list[str]]:
    """Find postings and apply the free filters. No LLM calls.

    `source` is "boards" (Greenhouse/Lever/Ashby public APIs), "linkedin"
    (scraped, remote + Easy Apply only), or "all".
    """
    prefix: list[str] = []
    seed: list[JobPosting] = []

    if source in ("linkedin", "all"):
        from jobgraph.nodes.discover_linkedin import discover_linkedin

        li, li_notes = await discover_linkedin(limit=(limit or 25) * 3)
        seed.extend(li)
        prefix.extend(f"linkedin: {n}" for n in li_notes)

    state: DiscoveryState = {
        "listings": list(seed),
        "results": [],
        "errors": [],
        "notes": [],
    }

    if source in ("boards", "all"):
        found = await discover_listings(state)
        state["listings"] = [*seed, *found.get("listings", [])]
        state["errors"] = found.get("errors", [])
        found = {"errors": state["errors"]}
    else:
        found = {"errors": []}

    picked = await filter_and_select(state)
    selected: list[JobPosting] = picked.get("selected", [])
    notes: list[str] = [*prefix, *found.get("errors", []), *picked.get("notes", [])]

    if limit is not None:
        if len(selected) > limit:
            notes.append(f"limited to {limit} of {len(selected)} selected")
        selected = selected[:limit]

    return selected, notes


async def run_batch(
    jobs: list[JobPosting],
    *,
    submit: bool,
    run_id: str | None = None,
    on_result=None,
) -> dict[str, Any]:
    """Drive each job through the graph. Returns a summary dict."""
    run_id = run_id or new_run_id()
    started = datetime.now(UTC)

    storage.init_db()
    storage.log(
        run_id,
        "info",
        "batch_start",
        f"{len(jobs)} jobs, submit={submit}, dry_run={settings.dry_run}",
    )

    results: list[dict[str, Any]] = []
    fatal: str | None = None

    async with checkpointer() as cp:
        graph = compile_application_graph(cp, with_submission=submit)

        for i, job in enumerate(jobs, 1):
            try:
                final = await run_one_application(job, run_id=run_id, graph=graph)
            except FatalRunError as exc:
                fatal = str(exc)
                storage.log(run_id, "fatal", "batch_abort", fatal)
                break

            results.append(
                {
                    "url": job.url,
                    "company": final.get("job", job).company,
                    "title": final.get("job", job).title,
                    "status": final.get("status"),
                    "fit": getattr(final.get("fit"), "score", None),
                    "outcome": (
                        str(final["submission"].outcome) if final.get("submission") else None
                    ),
                    "run_dir": final.get("run_dir"),
                }
            )
            if on_result is not None:
                on_result(i, len(jobs), results[-1])

            # Stop as soon as the live budget is spent rather than churning
            # through the remaining jobs only to queue them all.
            if submit and not settings.dry_run:
                submitted = sum(1 for r in results if r["status"] == "submitted")
                if submitted >= settings.max_submissions_per_run:
                    storage.log(
                        run_id,
                        "info",
                        "cap_reached",
                        f"{submitted} submitted this run",
                    )
                    break

    summary = _summarise(run_id, jobs, results, fatal, started)
    _write_summary(run_id, summary)
    return summary


def _summarise(
    run_id: str,
    jobs: list[JobPosting],
    results: list[dict[str, Any]],
    fatal: str | None,
    started: datetime,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"] or "unknown"] = counts.get(r["status"] or "unknown", 0) + 1

    return {
        "run_id": run_id,
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "dry_run": settings.dry_run,
        "considered": len(jobs),
        "processed": len(results),
        "counts": counts,
        "submitted": counts.get("submitted", 0),
        "fatal": fatal,
        "llm": TELEMETRY.summary(),
        "results": results,
    }


def _write_summary(run_id: str, summary: dict[str, Any]) -> None:
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"# Run {run_id}",
        "",
        f"- Started: {summary['started']}  →  {summary['finished']}",
        f"- Mode: {'DRY RUN' if summary['dry_run'] else 'LIVE'}",
        f"- Considered {summary['considered']}, processed {summary['processed']}",
        f"- **Submitted: {summary['submitted']}**",
        "",
    ]
    if summary["fatal"]:
        lines += [f"> **Run aborted:** {summary['fatal']}", ""]

    if summary["counts"]:
        lines += ["| status | n |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in sorted(summary["counts"].items())]
        lines += [""]

    if summary["results"]:
        lines += [
            "| company | title | fit | status | outcome |",
            "|---|---|---|---|---|",
        ]
        for r in summary["results"]:
            lines.append(
                f"| {r['company']} | {r['title'][:44]} | {r['fit'] or '-'} | "
                f"{r['status']} | {r['outcome'] or '-'} |"
            )
        lines += [""]

    llm = summary["llm"]
    lines += [
        "## Engine",
        "",
        f"- {llm['calls']} calls, {llm['failed']} failed, retry rate {llm['retry_rate']}",
        f"- ${llm['cost_usd']}, {llm['avg_duration_s']}s average",
        f"- cache: {llm['cache_read_tokens']:,} read / {llm['cache_write_tokens']:,} written",
        "",
        "Per-application detail is in each `package.md` under this directory.",
    ]

    (run_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def read_urls(path: Path) -> list[JobPosting]:
    """One URL per line; blank lines and `#` comments ignored."""
    jobs: list[JobPosting] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        jobs.append(
            JobPosting(url=line, title="(from posting)", company="(from posting)")
        )
    return jobs
