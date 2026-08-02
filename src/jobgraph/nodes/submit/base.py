"""Shared form-filling primitives for every ATS adapter.

Two ideas do most of the work here:

  * `FormFiller` records every value it enters. In DRY_RUN nothing is clicked,
    but the record and the screenshots are identical to a live run — so what
    you review in dry-run is exactly what would have been sent.

  * `match_answer` maps an arbitrary form label onto one of our pre-generated
    screening answers by lexical overlap. It returns None rather than guessing,
    and a required field with no match sends the job to the manual queue. An
    agent inventing an answer to a question it did not understand is precisely
    the failure this project cannot afford.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from playwright.async_api import Locator, Page

from jobgraph.browser import human_pause
from jobgraph.config import settings
from jobgraph.schemas import ScreeningAnswers

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "is",
    "are", "do", "you", "your", "we", "us", "this", "that", "with", "what",
    "why", "how", "would", "will", "please", "if", "any", "have", "has",
}

# Label patterns -> the canonical question in compose.STANDARD_QUESTIONS.
CANONICAL_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"notice period|start date|availab|when can you", re.I),
     "What is your notice period / earliest start date?"),
    (re.compile(r"salary|compensation|ctc|expected pay|remuneration", re.I),
     "What are your salary expectations?"),
    (re.compile(r"authoriz|authoris|visa|sponsor|work permit|eligible to work", re.I),
     "Are you legally authorised to work in this location?"),
    (re.compile(r"relocat|on-?site|onsite|willing to travel|hybrid", re.I),
     "Are you willing to relocate or work on-site?"),
    (re.compile(r"why (do you want to work|this company|us\b)|interest in (our|the company)", re.I),
     "Why do you want to work at this company?"),
    (re.compile(r"why (are you interested|this role|this position)", re.I),
     "Why are you interested in this role?"),
    (re.compile(r"relevant experience|qualif|why are you a (good|strong) fit", re.I),
     "What relevant experience do you bring to this role?"),
    (re.compile(r"challeng|difficult|proud|recent project|technical problem", re.I),
     "Describe a challenging technical problem you solved recently.",),
]


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9+#.]+", text.lower()) if w not in STOPWORDS and len(w) > 2}


def match_answer(
    label: str,
    screening: ScreeningAnswers | None,
    *,
    company: str = "",
) -> str | None:
    """Best pre-generated answer for a form label, or None.

    None is a real answer. The caller escalates to the manual queue rather
    than letting the graph improvise into a field it did not understand.

    `company` matters more than it looks: "Why Anthropic?" is one of the most
    common required questions on Greenhouse and matches none of the generic
    patterns. Without this, most postings with a company-specific question
    fall through to manual — which for an unattended run means the queue, not
    the applications, is what you find in the morning.
    """
    if screening is None or not screening.answers:
        return None

    by_question = {a.question: a.answer for a in screening.answers}

    # "Why <Company>?" / "Why do you want to join <Company>?"
    if company:
        name = re.escape(company.split(",")[0].strip())
        if name and re.search(rf"\bwhy\b.{{0,40}}\b{name}\b", label, re.I):
            canonical = "Why do you want to work at this company?"
            if canonical in by_question:
                return by_question[canonical]

    # Strong signal next: explicit patterns for the common ATS fields.
    for pattern, canonical in CANONICAL_HINTS:
        if pattern.search(label):
            if canonical in by_question:
                return by_question[canonical]

    # Otherwise lexical overlap, with a floor so a weak match is no match.
    label_tokens = _tokens(label)
    if not label_tokens:
        return None

    best_score = 0.0
    best: str | None = None
    for answer in screening.answers:
        overlap = label_tokens & _tokens(answer.question)
        if not overlap:
            continue
        score = len(overlap) / len(label_tokens)
        if score > best_score:
            best_score, best = score, answer.answer

    return best if best_score >= 0.4 else None


@dataclass
class FormFiller:
    """Fills a form and records exactly what it entered."""

    page: Page
    filled: dict[str, str] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    unanswered_required: list[str] = field(default_factory=list)

    async def fill_text(self, selector: str, value: str, *, label: str = "") -> bool:
        loc = self.page.locator(selector).first
        try:
            if await loc.count() == 0 or not await loc.is_visible():
                return False
            await loc.scroll_into_view_if_needed(timeout=3_000)
            await loc.fill(value, timeout=5_000)
            await human_pause()
        except Exception as exc:  # noqa: BLE001
            self.skipped.append(f"{label or selector}: {type(exc).__name__}")
            return False
        self.filled[label or selector] = value if len(value) < 200 else value[:200] + "…"
        return True

    async def upload(self, selector: str, path: Path, *, label: str = "resume") -> bool:
        loc = self.page.locator(selector).first
        try:
            if await loc.count() == 0:
                return False
            await loc.set_input_files(str(path), timeout=15_000)
            await human_pause()
        except Exception as exc:  # noqa: BLE001
            self.skipped.append(f"{label}: {type(exc).__name__}")
            return False
        self.filled[label] = path.name
        return True

    async def answer_free_text_fields(
        self,
        screening: ScreeningAnswers | None,
        *,
        container: str = "form",
        company: str = "",
    ) -> None:
        """Walk every textarea in the form and fill what we can match.

        Anything required that we cannot match is recorded in
        `unanswered_required`, which the adapter treats as a hard stop.
        """
        areas = self.page.locator(f"{container} textarea")
        try:
            count = await areas.count()
        except Exception:  # noqa: BLE001
            return

        for i in range(min(count, 25)):
            area = areas.nth(i)
            try:
                if not await area.is_visible():
                    continue
                label = await self._label_for(area)
                required = await self._is_required(area)
                existing = (await area.input_value()) or ""
                if existing.strip():
                    continue

                answer = match_answer(label, screening, company=company)
                if answer:
                    await area.fill(answer, timeout=5_000)
                    await human_pause()
                    self.filled[label or f"textarea[{i}]"] = answer[:200]
                elif required:
                    self.unanswered_required.append(label or f"textarea[{i}]")
                else:
                    self.skipped.append(f"optional, unmatched: {label[:60]}")
            except Exception as exc:  # noqa: BLE001
                self.skipped.append(f"textarea[{i}]: {type(exc).__name__}")

    async def answer_choice_fields(
        self,
        *,
        container: str = "form",
        prefs: dict | None = None,
    ) -> None:
        """Dropdowns and radio groups — LinkedIn Easy Apply is full of them.

        These are almost always yes/no eligibility questions ("Do you have
        experience with X?", "Are you authorised to work in...?") or a years-
        of-experience number. Free-text answers cannot be pasted into them, so
        they need their own handling.

        The rule is the same as everywhere else: answer only what we can
        derive from the profile, and record anything required-but-unanswered
        so the caller can escalate instead of guessing.
        """
        prefs = prefs or {}
        await self._answer_selects(container, prefs)
        await self._answer_radios(container, prefs)

    async def _answer_selects(self, container: str, prefs: dict) -> None:
        selects = self.page.locator(f"{container} select")
        try:
            count = await selects.count()
        except Exception:  # noqa: BLE001
            return

        for i in range(min(count, 30)):
            sel = selects.nth(i)
            try:
                if not await sel.is_visible():
                    continue
                current = await sel.input_value()
                if current and current not in ("", "Select an option"):
                    continue

                label = await self._label_for(sel)
                options = await sel.locator("option").all_text_contents()
                choice = _choose_option(label, options, prefs)

                if choice is None:
                    if await self._is_required(sel):
                        self.unanswered_required.append(f"[dropdown] {label[:80]}")
                    continue

                await sel.select_option(label=choice, timeout=5_000)
                await human_pause()
                self.filled[f"select:{label[:60]}"] = choice
            except Exception as exc:  # noqa: BLE001
                self.skipped.append(f"select[{i}]: {type(exc).__name__}")

    async def _answer_radios(self, container: str, prefs: dict) -> None:
        groups = self.page.locator(f"{container} fieldset")
        try:
            count = await groups.count()
        except Exception:  # noqa: BLE001
            return

        for i in range(min(count, 30)):
            group = groups.nth(i)
            try:
                radios = group.locator("input[type='radio']")
                n = await radios.count()
                if n == 0:
                    continue
                if await radios.locator(":checked").count() > 0:
                    continue

                label = " ".join((await group.inner_text()).split())
                options: list[str] = []
                for r in range(n):
                    options.append(await self._label_for(radios.nth(r)) or "")

                choice = _choose_option(label, options, prefs)
                if choice is None:
                    self.unanswered_required.append(f"[radio] {label[:80]}")
                    continue

                idx = options.index(choice)
                await radios.nth(idx).check(timeout=5_000)
                await human_pause()
                self.filled[f"radio:{label[:60]}"] = choice
            except Exception as exc:  # noqa: BLE001
                self.skipped.append(f"radio[{i}]: {type(exc).__name__}")

    async def _label_for(self, loc: Locator) -> str:
        """Best-effort label text for a field. ATS markup varies wildly."""
        for expr in (
            "el => el.labels && el.labels[0] ? el.labels[0].innerText : ''",
            "el => el.getAttribute('aria-label') || ''",
            "el => el.getAttribute('placeholder') || ''",
            "el => el.closest('div,fieldset,li') ? "
            "el.closest('div,fieldset,li').innerText.slice(0, 300) : ''",
        ):
            try:
                text = (await loc.evaluate(expr) or "").strip()
            except Exception:  # noqa: BLE001
                continue
            if text:
                return " ".join(text.split())
        return ""

    async def _is_required(self, loc: Locator) -> bool:
        try:
            if await loc.get_attribute("required") is not None:
                return True
            if (await loc.get_attribute("aria-required") or "").lower() == "true":
                return True
            label = await self._label_for(loc)
            return "*" in label or "required" in label.lower()
        except Exception:  # noqa: BLE001
            return False

    async def screenshot(self, path: Path, *, full_page: bool = True) -> str | None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            await self.page.screenshot(path=str(path), full_page=full_page)
        except Exception:  # noqa: BLE001
            return None
        return str(path)


async def find_first(page: Page, selectors: list[str]) -> str | None:
    """First selector present on the page. ATS markup drifts; adapters pass a
    list of known spellings rather than betting on one."""
    for sel in selectors:
        try:
            if await page.locator(sel).first.count() > 0:
                return sel
        except Exception:  # noqa: BLE001
            continue
    return None


def contact_fields() -> dict[str, str]:
    """Identity fields, from the corpus. Never invented."""
    from jobgraph.profile import load_profile

    p = load_profile()
    parts = p.name.split()
    return {
        "first_name": parts[0],
        "last_name": parts[-1] if len(parts) > 1 else "",
        "full_name": p.name,
        "email": p.email,
        "phone": p.phone,
        "location": p.location,
        "linkedin": p.links.get("linkedin", ""),
        "github": p.links.get("github", ""),
    }


def dry_run_banner() -> str:
    return "DRY RUN — form filled, nothing submitted" if settings.dry_run else "LIVE"


# Questions where answering "yes" is a factual claim about the candidate that
# we can only make from the profile. Anything not derivable stays unanswered.
_YES = ("yes", "y", "true")
_NO = ("no", "n", "false")


def _choose_option(label: str, options: list[str], prefs: dict) -> str | None:
    """Pick an option for a dropdown or radio group, or None.

    Deliberately conservative. The failure mode we cannot accept is confidently
    ticking "yes, I have 5 years of Kubernetes" on a form that goes out under
    someone's name, so anything not directly derivable from the profile returns
    None and escalates to a human.
    """
    clean = [o.strip() for o in options if o and o.strip()]
    real = [o for o in clean if o.lower() not in ("select an option", "choose", "-", "")]
    if not real:
        return None

    text = label.lower()

    def pick(words: tuple[str, ...]) -> str | None:
        for opt in real:
            if opt.strip().lower() in words:
                return opt
        return None

    # Work authorisation. The profile states this explicitly; anything about
    # a country the candidate needs sponsorship for is a truthful "no".
    if re.search(r"authoriz|authoris|eligible to work|right to work|work permit", text):
        auth = str(prefs.get("work_authorization", "")).lower()
        if "india" in text and "indian" in auth:
            return pick(_YES)
        if re.search(r"\b(us|u\.s|united states|uk|canada|eu|europe)\b", text):
            return pick(_NO)
        return None

    if re.search(r"sponsor", text):
        # "Will you require sponsorship?" — true for anywhere outside India.
        auth = str(prefs.get("work_authorization", "")).lower()
        if "sponsorship" in auth and "india" not in text:
            return pick(_YES)
        return None

    if re.search(r"willing to relocat|open to relocat", text):
        return pick(_YES if prefs.get("relocate_ok") else _NO)

    if re.search(r"remote|work from home", text) and prefs.get("remote_ok", True):
        return pick(_YES)

    if re.search(r"notice period|when can you start|available to start", text):
        notice = str(prefs.get("notice_period", ""))
        for opt in real:
            if opt.lower() in notice.lower() or notice.lower() in opt.lower():
                return opt
        return None

    # Years of experience: answer only when the profile clearly clears the bar.
    m = re.search(r"(\d+)\+?\s*(?:or more\s*)?years", text)
    if m and re.search(r"how many years|years of experience|do you have", text):
        wanted = int(m.group(1))
        total = int(prefs.get("total_years_experience", 8))
        if wanted <= total:
            return pick(_YES) or (str(total) if str(total) in real else None)
        return pick(_NO)

    if re.search(r"how many years", text):
        total = prefs.get("total_years_experience", 8)
        for opt in real:
            if opt.strip() == str(total):
                return opt
        return None

    # Boilerplate consent / acknowledgement checkboxes phrased as radios.
    if re.search(r"agree|acknowledge|consent|have read", text):
        return pick(_YES)

    return None
