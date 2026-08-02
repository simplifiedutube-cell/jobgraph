"""The fallback that makes auto-submission safe to run unattended.

Anything the graph does not confidently understand — an unsupported vendor, a
required question with no matching answer, a form that changed shape — lands
here instead of being guessed at. It writes a folder you can act on in two
minutes: the PDF, the letter, the answers, and the exact URL.

An agent that queues work it doesn't understand is far more useful than one
that submits a half-filled form under your name.
"""

from __future__ import annotations

from pathlib import Path

from jobgraph.schemas import AtsVendor, SubmissionOutcome, SubmissionResult


def queue(state: dict, run_dir: Path, *, vendor: AtsVendor, reason: str) -> SubmissionResult:
    job = state["job"]
    run_dir.mkdir(parents=True, exist_ok=True)

    letter = state.get("cover_letter")
    screening = state.get("screening")
    resume_pdf = state.get("resume_pdf")

    lines = [
        f"# Apply by hand: {job.title} — {job.company}",
        "",
        f"**{reason}**",
        "",
        f"- URL: {job.url}",
        f"- Detected ATS: {vendor}",
        f"- Resume: `{resume_pdf or '(not generated)'}`",
        "",
        "Everything below is ready to paste.",
        "",
    ]

    if letter is not None:
        lines += ["## Cover letter", "", letter.body, ""]

    if screening is not None:
        lines += ["## Screening answers", ""]
        for a in screening.answers:
            lines += [f"### {a.question}", "", a.answer, ""]

    (run_dir / "APPLY_BY_HAND.md").write_text("\n".join(lines), encoding="utf-8")

    return SubmissionResult(
        outcome=SubmissionOutcome.QUEUED_MANUAL,
        vendor=vendor,
        confirmation=str(run_dir / "APPLY_BY_HAND.md"),
        error=reason,
    )
