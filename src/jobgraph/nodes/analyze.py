"""fetch_jd -> parse_jd -> score_fit, plus the gate that ends cheap runs early."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

from jobgraph.browser import detect_vendor, fetch_page, html_to_text
from jobgraph.config import check_stop, settings
from jobgraph.llm import ENGINE
from jobgraph.profile import corpus_for_prompt, load_preferences, load_profile
from jobgraph.schemas import AtsVendor, FitReport, ParsedJD
from jobgraph.state import ApplicationState

MAX_JD_CHARS = 18_000


def _as_local_path(url: str) -> Path | None:
    """A file:// URL or a plain filesystem path, else None."""
    if url.startswith("file://"):
        return Path(unquote(urlparse(url).path))
    if "://" not in url:
        p = Path(url).expanduser()
        if p.suffix or p.exists():
            return p
    return None


async def fetch_jd(state: ApplicationState) -> dict:
    """Pull the JD body. No LLM — this is a page load.

    A local path or file:// URL is read straight off disk. That's how JDs that
    arrive by email get in, and it's what makes the eval suite runnable offline
    against golden fixtures.
    """
    check_stop()
    job = state["job"]

    local = _as_local_path(job.url)
    if local is not None:
        if not local.exists():
            return {"errors": [f"fetch_jd: no such file {local}"], "status": "failed"}
        raw = local.read_text(encoding="utf-8", errors="replace")
        text = html_to_text(raw) if local.suffix.lower() in {".html", ".htm"} else raw
        return {
            "jd_text": text[:MAX_JD_CHARS],
            "notes": [f"read {len(text)} chars from {local.name}"],
        }

    try:
        html, text = await fetch_page(job.url)
    except Exception as exc:  # noqa: BLE001 - one bad page must not kill a run
        return {
            "errors": [f"fetch_jd: {type(exc).__name__}: {exc}"],
            "status": "failed",
        }

    if len(text) < 200:
        return {
            "errors": [f"fetch_jd: page yielded only {len(text)} chars — likely a login wall"],
            "status": "failed",
        }

    vendor = detect_vendor(job.url, html)
    return {
        "jd_text": text[:MAX_JD_CHARS],
        "notes": [f"fetched {len(text)} chars; vendor guess = {vendor}"],
    }


async def parse_jd(state: ApplicationState) -> dict:
    """JD text -> structured requirements."""
    check_stop()
    job = state["job"]
    jd_text = state.get("jd_text", "")
    if not jd_text:
        return {"errors": ["parse_jd: no jd_text"], "status": "failed"}

    prompt = f"""\
You are extracting structured data from a job posting. Be literal: record what
the posting says, not what you infer a role like this usually wants.

Known from the listing:
  url: {job.url}
  title: {job.title}
  company: {job.company}
  location: {job.location or "unstated"}

Split requirements into must_have and nice_to_have using the posting's own
framing — "required", "must have", "you have" are must_have; "bonus",
"preferred", "nice to have", "a plus" are nice_to_have. If the posting does not
distinguish them, treat the first list as must_have.

For ats_vendor, judge from the URL and any form markup below. Do not guess from
the company name.

--- JOB POSTING ---
{jd_text}
--- END ---
"""

    try:
        parsed = await ENGINE.call(
            node="parse_jd",
            prompt=prompt,
            schema=ParsedJD,
            profile=settings.profile_parse_jd,
        )
    except Exception as exc:  # noqa: BLE001
        return {"errors": [f"parse_jd: {exc}"], "status": "failed"}

    # The URL is authoritative for vendor detection; the model only fills gaps.
    url_vendor = detect_vendor(job.url)
    if url_vendor is not AtsVendor.UNKNOWN:
        parsed.ats_vendor = url_vendor

    # Backfill title/company from the posting itself. When a run starts from a
    # bare URL these are placeholders, and they end up in the ledger, the run
    # directory name and the PDF footer.
    updated = job.model_copy(
        update={
            "title": parsed.title or job.title,
            "company": parsed.company or job.company,
            "location": job.location or parsed.location,
        }
    )

    return {
        "job": updated,
        "jd": parsed,
        "notes": [
            f"parsed {len(parsed.requirements)} requirements "
            f"({updated.title} @ {updated.company})"
        ],
    }


async def score_fit(state: ApplicationState) -> dict:
    """Score the match, citing corpus bullet ids as evidence.

    Evidence ids matter beyond the score: they are what tailor_resume starts
    from, so a well-grounded fit report makes the tailoring better too.
    """
    check_stop()
    jd = state.get("jd")
    if jd is None:
        return {"errors": ["score_fit: no parsed jd"], "status": "failed"}

    prefs = load_preferences()
    corpus = corpus_for_prompt()

    prompt = f"""\
Score how well this candidate fits this role, then justify it per requirement.

Be calibrated, not generous. This score gates whether an application is sent,
so inflation wastes the candidate's time and the employer's.

Guidance for the 0-100 score:
  85-100  meets every must-have with direct, demonstrable evidence
  70-84   meets most must-haves; gaps are adjacent skills they could pick up
  50-69   meaningful gaps in must-haves, or a seniority/domain mismatch
  0-49    wrong role, wrong level, or wrong domain

For every requirement, cite the corpus bullet ids that evidence it. Cite only
ids that appear in the corpus below. If nothing in the corpus supports a
requirement, set met=false and leave evidence_bullet_ids empty — do not stretch
an unrelated bullet to cover it.

Set recommendation to "apply" (strong), "stretch" (worth a shot) or "skip".

--- CANDIDATE PREFERENCES ---
Target titles: {", ".join(prefs.get("titles", []))}
Locations: {", ".join(prefs.get("locations", []))}
Seniority: {", ".join(prefs.get("seniority", []))}
Notes: {prefs.get("narrative", "")}

--- ROLE ---
{jd.model_dump_json(indent=2)}

{corpus}
"""

    try:
        fit = await ENGINE.call(
            node="score_fit",
            prompt=prompt,
            schema=FitReport,
            profile=settings.profile_score_fit,
        )
    except Exception as exc:  # noqa: BLE001
        return {"errors": [f"score_fit: {exc}"], "status": "failed"}

    # Evidence must be real. A hallucinated id here would silently poison
    # tailoring downstream, so drop unknown ids and say so.
    valid = load_profile().bullet_ids()
    dropped: list[str] = []
    for match in fit.matches:
        keep = [i for i in match.evidence_bullet_ids if i in valid]
        dropped.extend(i for i in match.evidence_bullet_ids if i not in valid)
        match.evidence_bullet_ids = keep

    notes = [f"fit={fit.score} ({fit.recommendation})"]
    if dropped:
        notes.append(f"dropped {len(dropped)} hallucinated evidence ids: {sorted(set(dropped))}")

    return {"fit": fit, "notes": notes}


def fit_gate(state: ApplicationState) -> str:
    """Conditional edge: is this worth generating a package for?"""
    if state.get("status") == "failed":
        return "rejected"
    fit = state.get("fit")
    if fit is None:
        return "rejected"
    if fit.score < settings.min_fit_score:
        return "rejected"
    return "proceed"


async def log_rejected(state: ApplicationState) -> dict:
    """Terminal node for the cheap path."""
    fit = state.get("fit")
    job = state["job"]
    if fit is not None:
        reason = f"fit {fit.score} < {settings.min_fit_score}; gaps: {'; '.join(fit.gaps[:3])}"
    else:
        reason = "no fit report — " + "; ".join(state.get("errors", [])[:2])
    return {
        "status": "rejected",
        "notes": [f"rejected {job.company} — {reason}"],
    }
