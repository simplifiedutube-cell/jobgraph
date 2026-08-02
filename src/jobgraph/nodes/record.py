"""Terminal node: persist the application and write a human-readable trail.

The Markdown file is the point. After an unattended run you want to open one
file and see what happened, not query SQLite.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from jobgraph import storage
from jobgraph.state import ApplicationState


def _package_markdown(state: ApplicationState) -> str:
    job = state["job"]
    jd = state.get("jd")
    fit = state.get("fit")
    tailored = state.get("tailored")
    letter = state.get("cover_letter")
    screening = state.get("screening")
    critique = state.get("critique")
    submission = state.get("submission")

    out: list[str] = [
        f"# {job.title} — {job.company}",
        "",
        f"- **URL**: {job.url}",
        f"- **Status**: `{state.get('status', 'unknown')}`",
        f"- **Recorded**: {datetime.now(UTC).isoformat(timespec='seconds')}",
    ]
    if fit:
        out += [
            f"- **Fit**: {fit.score}/100 ({fit.recommendation})",
        ]
    if jd:
        out += [f"- **ATS**: {jd.ats_vendor}"]
    if critique:
        out += [
            f"- **Critique**: {critique.verdict}, "
            f"keyword coverage {critique.keyword_coverage}%, "
            f"revisions {state.get('revision_count', 0)}"
        ]
    if submission:
        out += [f"- **Submission**: {submission.outcome}"]
        if submission.confirmation:
            out += [f"- **Confirmation**: {submission.confirmation}"]
    if state.get("resume_pdf"):
        out += [f"- **Resume**: `{state['resume_pdf']}`"]

    if fit:
        out += ["", "## Fit", "", "**Strengths**"]
        out += [f"- {s}" for s in fit.strengths] or ["- (none)"]
        out += ["", "**Gaps**"]
        out += [f"- {g}" for g in fit.gaps] or ["- (none)"]
        out += ["", "**Requirements**", ""]
        for m in fit.matches:
            mark = "x" if m.met else " "
            ev = f" `{', '.join(m.evidence_bullet_ids)}`" if m.evidence_bullet_ids else ""
            out += [f"- [{mark}] {m.requirement} — {m.note}{ev}"]

    if tailored:
        out += ["", "## Tailored resume", "", f"**Summary**: {tailored.summary}", ""]
        out += [f"**Highlighted**: {', '.join(tailored.highlighted_skills)}", ""]
        for section in tailored.sections:
            out += [f"### {section.heading}", ""]
            for b in section.bullets:
                out += [f"- {b.text}  <sub>`{b.source_id}`</sub>"]
            out += [""]

    if letter:
        out += ["## Cover letter", "", letter.body, ""]
        out += [f"<sub>sources: `{', '.join(letter.source_ids)}`</sub>", ""]

    if screening:
        out += ["## Screening answers", ""]
        for a in screening.answers:
            src = f"  <sub>`{', '.join(a.source_ids)}`</sub>" if a.source_ids else ""
            out += [f"**{a.question}**", "", f"{a.answer}{src}", ""]

    if critique and critique.issues:
        out += ["## Critique issues", ""]
        out += [f"- {i}" for i in critique.issues]
        out += [""]

    notes = state.get("notes") or []
    if notes:
        out += ["## Trace", ""]
        out += [f"- {n}" for n in notes]

    errors = state.get("errors") or []
    if errors:
        out += ["", "## Errors", ""]
        out += [f"- `{e}`" for e in errors]

    return "\n".join(out) + "\n"


async def record(state: ApplicationState) -> dict:
    """Write SQLite row, package.md and package.json. Never raises."""
    job = state["job"]
    run_dir = Path(state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    status = state.get("status")
    if status in (None, "scraped"):
        status = "drafted" if state.get("tailored") else "failed"

    persisted = {**state, "status": status}

    notes: list[str] = []
    try:
        (run_dir / "package.md").write_text(_package_markdown(persisted), encoding="utf-8")

        serialisable = {
            k: (v.model_dump(mode="json") if hasattr(v, "model_dump") else v)
            for k, v in persisted.items()
        }
        (run_dir / "package.json").write_text(
            json.dumps(serialisable, indent=2, default=str), encoding="utf-8"
        )

        row_id = storage.record_application(persisted)
        storage.log(
            state.get("run_id", "?"),
            "info",
            "recorded",
            f"{job.company} — {job.title} [{status}] row={row_id}",
        )
        notes.append(f"recorded row {row_id} -> {run_dir / 'package.md'}")
    except Exception as exc:  # noqa: BLE001 - recording must never lose a run
        notes.append(f"record failed: {type(exc).__name__}: {exc}")

    return {"status": status, "notes": notes}
