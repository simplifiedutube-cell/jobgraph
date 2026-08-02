"""The reflection node and the edge that sends work back for revision.

Two layers, deliberately different in kind:

  1. Provenance — deterministic Python. Did every claim cite a real corpus id?
     An LLM judging its own faithfulness is precisely the failure mode this
     project exists to prevent, so this half never asks a model anything.

  2. Quality — an LLM pass over keyword coverage, tone and length.

Layer 1 failing is a hard revise. Layer 2 failing is a soft revise, bounded by
`max_revisions` so a stubborn package can't loop forever.
"""

from __future__ import annotations

import re

from jobgraph.config import check_stop, settings
from jobgraph.llm import ENGINE
from jobgraph.profile import as_prompt_block, verify_provenance
from jobgraph.schemas import CritiqueReport
from jobgraph.state import ApplicationState


def _keyword_coverage(state: ApplicationState) -> tuple[int, list[str]]:
    """What fraction of the JD's tech keywords appear in the package?

    Computed in code so the number is reproducible and the model can't flatter
    itself. The model still gets asked for its own read, and disagreement
    between the two is interesting signal.
    """
    jd = state.get("jd")
    if jd is None or not jd.tech_keywords:
        return 100, []

    tailored = state.get("tailored")
    letter = state.get("cover_letter")

    blob = " ".join(
        [
            tailored.summary if tailored else "",
            " ".join(tailored.highlighted_skills) if tailored else "",
            " ".join(b.text for s in tailored.sections for b in s.bullets) if tailored else "",
            letter.body if letter else "",
        ]
    ).lower()

    missing = [
        kw
        for kw in jd.tech_keywords
        if not re.search(rf"(?<![a-z0-9]){re.escape(kw.lower())}(?![a-z0-9])", blob)
    ]
    covered = len(jd.tech_keywords) - len(missing)
    return round(100 * covered / len(jd.tech_keywords)), missing


async def critique(state: ApplicationState) -> dict:
    check_stop()
    tailored = state.get("tailored")
    letter = state.get("cover_letter")
    screening = state.get("screening")
    jd = state.get("jd")

    if tailored is None or jd is None:
        return {"errors": ["critique: missing package"], "status": "failed"}

    # --- layer 1: provenance, in code -------------------------------------
    violations = verify_provenance(
        tailored=tailored, cover_letter=letter, screening=screening
    )
    prov_errors = [f"{v.where} cited unknown id {v.claimed_id!r}" for v in violations]

    # --- coverage, in code ------------------------------------------------
    coverage, missing = _keyword_coverage(state)

    # --- layer 2: quality, via the model ----------------------------------
    revisions = state.get("revision_count", 0)
    budget_note = (
        f"\nThis package has ALREADY been revised {revisions} time(s) at your "
        "request. Each revision costs real time and can lose ground elsewhere. "
        "The bar for asking again is correspondingly higher — if the remaining "
        "objections are things you would not personally hold up a submission "
        "for, pass it.\n"
        if revisions
        else ""
    )

    prompt = f"""\
Review this job application package as a hiring manager who has read a thousand
of them. Be blunt. You are the last check before it is submitted.

Judge only what is here. Do not suggest adding experience the candidate does
not have — the package is deliberately constrained to their real history.
{budget_note}
THE DECISION YOU ARE MAKING
You are not grading this package, you are deciding one thing: is it better to
send this, or to hold it back and regenerate?

Regeneration is not free and not reliably an improvement — a previous revision
of this system fixed one issue while dropping a required keyword. So:

  blocking_issues  = defects that make sending this worse than not sending it.
                     A wrong or misleading claim. A qualification the JD
                     requires and the candidate demonstrably has, that appears
                     nowhere. Writing generic enough to be discarded on sight.
                     A resume that runs past two pages.

  issues           = everything else. "This bullet could be sharper", "I'd
                     reorder these two", "Docker is thinly evidenced". Real
                     observations, logged, not worth another pass.

Most competent packages have zero blocking_issues and several issues. That is
the expected outcome, not a failure to find problems. Set verdict to "revise"
if and only if blocking_issues is non-empty.

length_ok is false if the resume would run past two pages: roughly, more than
~22 bullets or bullets averaging over 30 words.

A deterministic checker already computed keyword coverage as {coverage}%.
Missing keywords: {", ".join(missing) or "none"}. Report your own read; if you
disagree with the checker, say so in issues.

--- ROLE ---
{jd.model_dump_json(indent=2)}

--- TAILORED RESUME ---
{as_prompt_block(tailored)}

--- COVER LETTER ---
{letter.body if letter else "(none)"}

--- SCREENING ANSWERS ---
{as_prompt_block(screening) if screening else "(none)"}
"""

    try:
        report = await ENGINE.call(
            node="critique",
            prompt=prompt,
            schema=CritiqueReport,
            profile=settings.profile_critique,
        )
    except Exception as exc:  # noqa: BLE001
        # A failed critique must not block submission of an otherwise valid
        # package — but a provenance failure still must.
        report = CritiqueReport(
            keyword_coverage=coverage,
            tone_ok=True,
            length_ok=True,
            issues=[f"critique unavailable: {exc}"],
            verdict="pass",
        )

    # Code-computed coverage is authoritative over the model's own read.
    report.keyword_coverage = coverage

    # Provenance failures are blocking regardless of what the model concluded.
    if prov_errors:
        report.blocking_issues = [*prov_errors, *report.blocking_issues]

    # Coverage below the floor is blocking; the model is inconsistent about
    # treating it that way on its own.
    if coverage < settings.min_keyword_coverage:
        report.blocking_issues.append(
            f"keyword coverage {coverage}% is below the {settings.min_keyword_coverage}% "
            f"floor; missing: {', '.join(missing[:10])}"
        )

    # The gate trusts blocking_issues, not the verdict string. Models say
    # "revise" while listing nothing blocking far more often than the reverse.
    report.verdict = "revise" if report.blocking_issues else "pass"

    notes = [
        f"critique: {report.verdict}, coverage {coverage}%, "
        f"{len(report.blocking_issues)} blocking / {len(report.issues)} minor"
    ]
    if missing:
        notes.append(f"missing keywords: {', '.join(missing[:8])}")
    for b in report.blocking_issues[:3]:
        notes.append(f"blocking: {b[:120]}")

    return {
        "critique": report,
        "provenance_errors": prov_errors,
        "notes": notes,
    }


def critique_gate(state: ApplicationState) -> str:
    """revise -> back to tailor_resume; pass -> on to submission."""
    if state.get("status") == "failed":
        return "abandon"

    report = state.get("critique")
    if report is None:
        return "abandon"

    revisions = state.get("revision_count", 0)

    if report.verdict == "pass":
        return "proceed"

    if revisions >= settings.max_revisions:
        # Out of retries. A package with unsourced claims must never ship;
        # one that is merely mediocre can.
        if state.get("provenance_errors"):
            return "abandon"
        return "proceed"

    return "revise"


async def bump_revision(state: ApplicationState) -> dict:
    """Increment the counter on the way back round the cycle."""
    n = state.get("revision_count", 0) + 1
    return {"revision_count": n, "notes": [f"revision {n} requested"]}


async def abandon(state: ApplicationState) -> dict:
    report = state.get("critique")
    prov = state.get("provenance_errors") or []
    if prov:
        reason = f"{len(prov)} unsourced claim(s) after {state.get('revision_count', 0)} revisions"
    elif report is not None:
        reason = "; ".join(report.issues[:3]) or "critique failed"
    else:
        reason = "; ".join(state.get("errors", [])[:3]) or "unknown"
    return {"status": "failed", "notes": [f"abandoned: {reason}"]}
