"""Vendor routing and the safety gates that sit in front of every submission.

Order matters. Cheapest and most absolute checks first, so a run that should
not be submitting anything discovers that before opening a browser:

    kill switch -> already applied -> daily cap -> review interrupt
                -> vendor supported? -> adapter -> dry_run gate (inside adapter)
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from jobgraph import storage
from jobgraph.browser import browser_context
from jobgraph.config import ReviewMode, check_stop, settings
from jobgraph.nodes.submit import greenhouse, lever, manual
from jobgraph.schemas import AtsVendor, SubmissionOutcome, SubmissionResult
from jobgraph.state import ApplicationState

ADAPTERS = {
    AtsVendor.GREENHOUSE: greenhouse.submit,
    AtsVendor.LEVER: lever.submit,
}

# Ashby's form is close enough to Greenhouse's that the generic filler handles
# it; keeping it explicit documents that this is a deliberate reuse, not an
# accident of dict ordering.
ADAPTERS[AtsVendor.ASHBY] = greenhouse.submit


async def route_and_submit(state: ApplicationState) -> dict:
    check_stop()

    job = state["job"]
    jd = state.get("jd")
    run_dir = Path(state["run_dir"])
    vendor = jd.ats_vendor if jd else AtsVendor.UNKNOWN

    # --- gates ------------------------------------------------------------
    if storage.already_applied(job.url):
        return {
            "submission": SubmissionResult(
                outcome=SubmissionOutcome.QUEUED_MANUAL,
                vendor=vendor,
                error="already applied to this posting",
            ),
            "status": "manual",
            "notes": ["skipped: already applied"],
        }

    remaining = storage.remaining_daily_budget()
    if remaining <= 0 and not settings.dry_run:
        return {
            "submission": SubmissionResult(
                outcome=SubmissionOutcome.QUEUED_MANUAL,
                vendor=vendor,
                error=f"daily cap of {settings.daily_submission_cap} reached",
            ),
            "status": "manual",
            "notes": [f"daily cap reached ({settings.daily_submission_cap})"],
        }

    if _needs_review():
        # A LangGraph interrupt() would suspend here for approval. review_mode
        # defaults to AUTO, so this path is opt-in.
        from langgraph.errors import NodeInterrupt

        raise NodeInterrupt(
            f"review_mode={settings.review_mode}: approve submission to "
            f"{job.company} — {job.title} ({job.url})"
        )

    adapter = ADAPTERS.get(vendor)
    if adapter is None:
        result = manual.queue(
            dict(state),
            run_dir,
            vendor=vendor,
            reason=f"no adapter for ATS vendor '{vendor}'",
        )
        return {
            "submission": result,
            "status": "manual",
            "notes": [f"queued for manual apply ({vendor})"],
        }

    if not state.get("resume_pdf"):
        result = manual.queue(
            dict(state), run_dir, vendor=vendor, reason="no resume PDF was generated"
        )
        return {"submission": result, "status": "manual", "notes": ["queued: no resume"]}

    # --- drive the form ---------------------------------------------------
    try:
        async with browser_context() as ctx:
            page = await ctx.new_page()
            try:
                await page.goto(job.url, wait_until="domcontentloaded", timeout=45_000)
                await page.wait_for_timeout(2_000)
                result = await adapter(page, dict(state), run_dir, dry_run=settings.dry_run)
            finally:
                await page.close()
    except Exception as exc:  # noqa: BLE001
        result = manual.queue(
            dict(state),
            run_dir,
            vendor=vendor,
            reason=f"submission raised {type(exc).__name__}: {exc}",
        )
        return {
            "submission": result,
            "status": "manual",
            "errors": [f"submit: {type(exc).__name__}: {exc}"],
            "notes": ["queued for manual apply after error"],
        }

    result.at = datetime.now(UTC)

    status = {
        SubmissionOutcome.SUBMITTED: "submitted",
        SubmissionOutcome.DRY_RUN: "drafted",
        SubmissionOutcome.QUEUED_MANUAL: "manual",
        SubmissionOutcome.FAILED: "failed",
    }[result.outcome]

    notes = [f"submit: {result.outcome} via {vendor} ({len(result.fields_filled)} fields)"]
    if result.error:
        notes.append(f"submit note: {result.error}")

    return {"submission": result, "status": status, "notes": notes}


def _needs_review() -> bool:
    if settings.dry_run:
        return False
    if settings.review_mode is ReviewMode.ALWAYS:
        return True
    if settings.review_mode is ReviewMode.FIRST_N:
        return storage.submissions_today() < settings.review_first_n
    return False


async def verify_submission(state: ApplicationState) -> dict:
    """Second opinion on whether the click actually landed.

    The adapter checks for a success marker immediately. This node exists so
    that a slow confirmation page, or a redirect that arrives late, is still
    caught before the run is recorded as a failure.
    """
    check_stop()
    result = state.get("submission")
    if result is None or result.outcome is not SubmissionOutcome.FAILED:
        return {}

    job = state["job"]
    try:
        async with browser_context() as ctx:
            page = await ctx.new_page()
            try:
                await page.goto(job.url, wait_until="domcontentloaded", timeout=30_000)
                await page.wait_for_timeout(2_000)
                body = (await page.locator("body").inner_text(timeout=10_000)).lower()
            finally:
                await page.close()
    except Exception as exc:  # noqa: BLE001
        return {"notes": [f"verify: could not re-check ({type(exc).__name__})"]}

    if any(
        p in body
        for p in ("already applied", "application received", "thank you for applying")
    ):
        result.outcome = SubmissionOutcome.SUBMITTED
        result.confirmation = "confirmed on re-check"
        return {
            "submission": result,
            "status": "submitted",
            "notes": ["verify: confirmed on re-check"],
        }

    return {"notes": ["verify: still no confirmation"]}
