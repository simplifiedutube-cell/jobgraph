"""Greenhouse. The most stable ATS DOM of the four, so it's the reference.

Two generations of markup are in the wild: the older `boards.greenhouse.io`
form with `#first_name`-style ids, and the newer `job-boards.greenhouse.io`
React app with generated ids. Each selector list covers both.
"""

from __future__ import annotations

from pathlib import Path

from playwright.async_api import Page

from jobgraph.nodes.submit.base import FormFiller, contact_fields, find_first
from jobgraph.schemas import AtsVendor, SubmissionOutcome, SubmissionResult

VENDOR = AtsVendor.GREENHOUSE

APPLY_BUTTON = [
    "button:has-text('Apply for this job')",
    "a:has-text('Apply for this job')",
    "button:has-text('Apply')",
    "#apply_button",
]

FIELDS = {
    "first_name": ["#first_name", "input[name='first_name']", "input[autocomplete='given-name']"],
    "last_name": ["#last_name", "input[name='last_name']", "input[autocomplete='family-name']"],
    "email": ["#email", "input[name='email']", "input[type='email']"],
    "phone": ["#phone", "input[name='phone']", "input[type='tel']"],
}

RESUME_INPUT = [
    "input[type='file'][name*='resume']",
    "#resume_upload_input",
    "input[type='file']",
]

COVER_LETTER = [
    "#cover_letter_text",
    "textarea[name*='cover_letter']",
    "textarea[aria-label*='Cover' i]",
]

SUBMIT_BUTTON = [
    "#submit_app",
    "button:has-text('Submit application')",
    "button[type='submit']:has-text('Submit')",
    "input[type='submit']",
]

SUCCESS_MARKERS = [
    "text=Thank you for applying",
    "text=Your application has been submitted",
    "text=Application submitted",
    "text=Thanks for applying",
]


async def submit(page: Page, state: dict, run_dir: Path, *, dry_run: bool) -> SubmissionResult:
    filler = FormFiller(page)
    shots: list[str] = []
    contact = contact_fields()

    # Newer boards hide the form behind an Apply button.
    apply_btn = await find_first(page, APPLY_BUTTON)
    if apply_btn:
        try:
            await page.locator(apply_btn).first.click(timeout=5_000)
            await page.wait_for_timeout(1_500)
        except Exception:  # noqa: BLE001
            pass  # form may already be inline

    for key, selectors in FIELDS.items():
        sel = await find_first(page, selectors)
        if sel:
            await filler.fill_text(sel, contact[key], label=key)

    resume_pdf = state.get("resume_pdf")
    if resume_pdf:
        sel = await find_first(page, RESUME_INPUT)
        if sel:
            await filler.upload(sel, Path(resume_pdf), label="resume")
        else:
            filler.skipped.append("no resume file input found")

    letter = state.get("cover_letter")
    if letter is not None:
        sel = await find_first(page, COVER_LETTER)
        if sel:
            await filler.fill_text(sel, letter.body, label="cover_letter")

    await filler.answer_free_text_fields(
        state.get("screening"), company=state["job"].company
    )

    shot = await filler.screenshot(run_dir / "greenhouse_filled.png")
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

    after = await filler.screenshot(run_dir / "greenhouse_after.png")
    if after:
        shots.append(after)

    confirmed = await _confirmed(page)
    return SubmissionResult(
        outcome=SubmissionOutcome.SUBMITTED if confirmed else SubmissionOutcome.FAILED,
        vendor=VENDOR,
        screenshots=shots,
        fields_filled=filler.filled,
        confirmation="success marker found" if confirmed else None,
        error=None if confirmed else "clicked submit but saw no confirmation",
    )


async def _confirmed(page: Page) -> bool:
    for marker in SUCCESS_MARKERS:
        try:
            if await page.locator(marker).first.count() > 0:
                return True
        except Exception:  # noqa: BLE001
            continue
    try:
        body = (await page.locator("body").inner_text(timeout=5_000)).lower()
    except Exception:  # noqa: BLE001
        return False
    return any(
        p in body
        for p in ("thank you for applying", "application submitted", "thanks for applying")
    )
