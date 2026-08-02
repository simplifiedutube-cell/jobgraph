"""LinkedIn Easy Apply.

Structurally different from the other adapters: not one form but a modal that
walks 2-6 steps (contact → resume → questions → review → submit), where each
step must be filled before "Next" appears, and the question set differs per
employer.

That shape drives two rules:

  * Fill, then look for Next/Review/Submit. Never click Submit unless the
    review step is actually reached — a mis-clicked "Next" that happens to be
    "Submit" is unrecoverable.
  * If any step has a required field we cannot answer, close the modal and
    queue for manual. Abandoning a half-filled Easy Apply is free; LinkedIn
    discards it. Guessing is not.

Note on terms: LinkedIn's ToS prohibits automated interaction. This adapter
exists because it was explicitly requested; the account risk is real and sits
with whoever runs it.
"""

from __future__ import annotations

from pathlib import Path

from playwright.async_api import Page

from jobgraph.browser import human_pause
from jobgraph.nodes.submit.base import FormFiller, contact_fields
from jobgraph.profile import load_preferences
from jobgraph.schemas import AtsVendor, SubmissionOutcome, SubmissionResult

VENDOR = AtsVendor.LINKEDIN_EASY_APPLY

EASY_APPLY_BUTTON = [
    "button.jobs-apply-button",
    "button:has-text('Easy Apply')",
    "button[aria-label*='Easy Apply']",
]
MODAL = "div.jobs-easy-apply-modal, div[role='dialog']"

NEXT_BUTTON = [
    "button[aria-label='Continue to next step']",
    "button:has-text('Next')",
    "button:has-text('Continue')",
]
REVIEW_BUTTON = [
    "button[aria-label='Review your application']",
    "button:has-text('Review')",
]
SUBMIT_BUTTON = [
    "button[aria-label='Submit application']",
    "button:has-text('Submit application')",
]
DISMISS = ["button[aria-label='Dismiss']", "button[aria-label='Cancel']"]

MAX_STEPS = 8


async def _click_first(page: Page, selectors: list[str], *, timeout: int = 4_000) -> bool:
    for sel in selectors:
        loc = page.locator(sel).first
        try:
            if await loc.count() and await loc.is_visible() and await loc.is_enabled():
                await loc.click(timeout=timeout)
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


async def _present(page: Page, selectors: list[str]) -> bool:
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if await loc.count() and await loc.is_visible():
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


async def _abandon(page: Page) -> None:
    """Close the modal and discard. LinkedIn drops the partial application."""
    await _click_first(page, DISMISS, timeout=3_000)
    await page.wait_for_timeout(800)
    # A confirmation dialog sometimes asks whether to save a draft.
    await _click_first(page, ["button:has-text('Discard')"], timeout=3_000)


async def submit(page: Page, state: dict, run_dir: Path, *, dry_run: bool) -> SubmissionResult:
    filler = FormFiller(page)
    shots: list[str] = []
    prefs = load_preferences()
    contact = contact_fields()
    company = state["job"].company

    if not await _click_first(page, EASY_APPLY_BUTTON, timeout=8_000):
        return SubmissionResult(
            outcome=SubmissionOutcome.QUEUED_MANUAL,
            vendor=VENDOR,
            error="no Easy Apply button — this posting applies on the company site",
        )

    await page.wait_for_timeout(2_000)
    if not await _present(page, [MODAL]):
        return SubmissionResult(
            outcome=SubmissionOutcome.QUEUED_MANUAL,
            vendor=VENDOR,
            error="Easy Apply modal did not open (login expired?)",
        )

    reached_review = False

    for step in range(1, MAX_STEPS + 1):
        # --- fill whatever this step shows ---------------------------------
        for key, sels in (
            ("email", ["input[type='email']", "select[id*='email']"]),
            ("phone", ["input[id*='phoneNumber']", "input[type='tel']"]),
        ):
            for sel in sels:
                if await filler.fill_text(f"{MODAL} {sel}", contact[key], label=key):
                    break

        resume_pdf = state.get("resume_pdf")
        if resume_pdf:
            await filler.upload(
                f"{MODAL} input[type='file']", Path(resume_pdf), label="resume"
            )

        await filler.answer_free_text_fields(
            state.get("screening"), container=MODAL, company=company
        )
        await filler.answer_choice_fields(container=MODAL, prefs=prefs)

        shot = await filler.screenshot(run_dir / f"linkedin_step{step}.png", full_page=False)
        if shot:
            shots.append(shot)

        # --- anything required we could not answer? ------------------------
        if filler.unanswered_required:
            await _abandon(page)
            return SubmissionResult(
                outcome=SubmissionOutcome.QUEUED_MANUAL,
                vendor=VENDOR,
                screenshots=shots,
                fields_filled=filler.filled,
                error="required questions with no derivable answer: "
                + "; ".join(filler.unanswered_required[:5]),
            )

        # --- advance --------------------------------------------------------
        if await _present(page, SUBMIT_BUTTON):
            reached_review = True
            break

        if await _click_first(page, REVIEW_BUTTON):
            await page.wait_for_timeout(1_500)
            continue

        if await _click_first(page, NEXT_BUTTON):
            await page.wait_for_timeout(1_500)
            continue

        # No way forward and no submit button — the step has something we do
        # not understand.
        await _abandon(page)
        return SubmissionResult(
            outcome=SubmissionOutcome.QUEUED_MANUAL,
            vendor=VENDOR,
            screenshots=shots,
            fields_filled=filler.filled,
            error=f"stuck on Easy Apply step {step}: no Next/Review/Submit available",
        )

    if not reached_review:
        await _abandon(page)
        return SubmissionResult(
            outcome=SubmissionOutcome.QUEUED_MANUAL,
            vendor=VENDOR,
            screenshots=shots,
            fields_filled=filler.filled,
            error=f"did not reach the review step within {MAX_STEPS} steps",
        )

    if dry_run:
        await _abandon(page)
        return SubmissionResult(
            outcome=SubmissionOutcome.DRY_RUN,
            vendor=VENDOR,
            screenshots=shots,
            fields_filled=filler.filled,
            confirmation="reached review step; modal discarded without submitting",
        )

    await human_pause()
    if not await _click_first(page, SUBMIT_BUTTON, timeout=8_000):
        await _abandon(page)
        return SubmissionResult(
            outcome=SubmissionOutcome.FAILED,
            vendor=VENDOR,
            screenshots=shots,
            fields_filled=filler.filled,
            error="submit button present but not clickable",
        )

    await page.wait_for_timeout(4_000)
    after = await filler.screenshot(run_dir / "linkedin_after.png", full_page=False)
    if after:
        shots.append(after)

    confirmed = await _confirmed(page)
    # Close whatever dialog remains so the next job starts clean.
    await _click_first(page, DISMISS, timeout=3_000)

    return SubmissionResult(
        outcome=SubmissionOutcome.SUBMITTED if confirmed else SubmissionOutcome.FAILED,
        vendor=VENDOR,
        screenshots=shots,
        fields_filled=filler.filled,
        confirmation="LinkedIn confirmed the application" if confirmed else None,
        error=None if confirmed else "clicked submit but saw no confirmation",
    )


async def _confirmed(page: Page) -> bool:
    for marker in (
        "text=Your application was sent",
        "text=Application sent",
        "text=Applied",
        "h2:has-text('Your application was sent')",
    ):
        try:
            if await page.locator(marker).first.count() > 0:
                return True
        except Exception:  # noqa: BLE001
            continue
    return False
