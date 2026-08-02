"""Lever (jobs.lever.co). Flat form, predictable `name` attributes."""

from __future__ import annotations

from pathlib import Path

from playwright.async_api import Page

from jobgraph.nodes.submit.base import FormFiller, contact_fields, find_first
from jobgraph.schemas import AtsVendor, SubmissionOutcome, SubmissionResult

VENDOR = AtsVendor.LEVER

APPLY_BUTTON = ["a:has-text('Apply for this job')", "a.postings-btn", "a:has-text('Apply')"]

FIELDS = {
    "full_name": ["input[name='name']", "#name"],
    "email": ["input[name='email']", "#email"],
    "phone": ["input[name='phone']", "#phone"],
    "location": ["input[name='location']", "#location"],
    "linkedin": ["input[name='urls[LinkedIn]']", "input[name*='LinkedIn']"],
    "github": ["input[name='urls[GitHub]']", "input[name*='GitHub']"],
}

RESUME_INPUT = ["input[name='resume']", "input[type='file']"]
COVER_LETTER = ["textarea[name='comments']", "textarea[name*='cover' i]"]
SUBMIT_BUTTON = [
    "button:has-text('Submit application')",
    "button[type='submit']",
    "input[type='submit']",
]
SUCCESS_MARKERS = [
    "text=Thank you for applying",
    "text=Application received",
    "text=Thanks for applying",
]


async def submit(page: Page, state: dict, run_dir: Path, *, dry_run: bool) -> SubmissionResult:
    filler = FormFiller(page)
    shots: list[str] = []
    contact = contact_fields()

    apply_btn = await find_first(page, APPLY_BUTTON)
    if apply_btn:
        try:
            await page.locator(apply_btn).first.click(timeout=5_000)
            await page.wait_for_timeout(1_500)
        except Exception:  # noqa: BLE001
            pass

    for key, selectors in FIELDS.items():
        value = contact.get(key, "")
        if not value:
            continue
        sel = await find_first(page, selectors)
        if sel:
            await filler.fill_text(sel, value, label=key)

    resume_pdf = state.get("resume_pdf")
    if resume_pdf:
        sel = await find_first(page, RESUME_INPUT)
        if sel:
            await filler.upload(sel, Path(resume_pdf), label="resume")

    letter = state.get("cover_letter")
    if letter is not None:
        sel = await find_first(page, COVER_LETTER)
        if sel:
            await filler.fill_text(sel, letter.body, label="cover_letter")

    await filler.answer_free_text_fields(
        state.get("screening"), company=state["job"].company
    )

    shot = await filler.screenshot(run_dir / "lever_filled.png")
    if shot:
        shots.append(shot)

    if filler.unanswered_required:
        return SubmissionResult(
            outcome=SubmissionOutcome.QUEUED_MANUAL,
            vendor=VENDOR,
            screenshots=shots,
            fields_filled=filler.filled,
            error="required questions with no matching answer: "
            + "; ".join(filler.unanswered_required[:5]),
        )

    if dry_run:
        return SubmissionResult(
            outcome=SubmissionOutcome.DRY_RUN,
            vendor=VENDOR,
            screenshots=shots,
            fields_filled=filler.filled,
            confirmation="form filled, submit not clicked",
        )

    sel = await find_first(page, SUBMIT_BUTTON)
    if not sel:
        return SubmissionResult(
            outcome=SubmissionOutcome.QUEUED_MANUAL,
            vendor=VENDOR,
            screenshots=shots,
            fields_filled=filler.filled,
            error="no submit button found",
        )

    await page.locator(sel).first.click(timeout=10_000)
    await page.wait_for_timeout(4_000)
    after = await filler.screenshot(run_dir / "lever_after.png")
    if after:
        shots.append(after)

    confirmed = False
    for marker in SUCCESS_MARKERS:
        try:
            if await page.locator(marker).first.count() > 0:
                confirmed = True
                break
        except Exception:  # noqa: BLE001
            continue

    return SubmissionResult(
        outcome=SubmissionOutcome.SUBMITTED if confirmed else SubmissionOutcome.FAILED,
        vendor=VENDOR,
        screenshots=shots,
        fields_filled=filler.filled,
        confirmation="success marker found" if confirmed else None,
        error=None if confirmed else "clicked submit but saw no confirmation",
    )
