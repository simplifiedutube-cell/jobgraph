"""Discovery over the official Greenhouse and Lever board APIs.

Deliberately not HTML scraping. Both vendors publish public, unauthenticated
JSON endpoints listing every open role on a board:

    https://boards-api.greenhouse.io/v1/boards/{slug}/jobs
    https://api.lever.co/v0/postings/{slug}?mode=json

That is strictly better than driving a browser at a job board: no login, no
bot detection, no selectors to break, no terms-of-service friction, and the
results are structured. It also lines up with what the submit adapters
actually support — there is no point discovering a posting we cannot fill.

Filtering is a cheap deterministic pass (title keywords, exclusions, dedupe
against the ledger) *before* any LLM call, so a 400-role board costs nothing
until a posting genuinely looks worth scoring.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx
import yaml

from jobgraph import storage
from jobgraph.config import check_stop, settings
from jobgraph.profile import load_preferences
from jobgraph.schemas import JobPosting
from jobgraph.state import DiscoveryState

GREENHOUSE_API = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
LEVER_API = "https://api.lever.co/v0/postings/{slug}?mode=json"

TIMEOUT = httpx.Timeout(20.0)


def load_targets() -> dict[str, list[str]]:
    """Company boards to poll, from data/profile/companies.yaml."""
    path = settings.profile_dir / "companies.yaml"
    if not path.exists():
        return {"greenhouse": [], "lever": []}
    data = yaml.safe_load(path.read_text()) or {}
    return {
        "greenhouse": list(data.get("greenhouse") or []),
        "lever": list(data.get("lever") or []),
    }


async def _get_json(client: httpx.AsyncClient, url: str) -> Any | None:
    try:
        r = await client.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:  # noqa: BLE001 - one dead board must not kill discovery
        return None


async def _greenhouse(client: httpx.AsyncClient, slug: str) -> list[JobPosting]:
    data = await _get_json(client, GREENHOUSE_API.format(slug=slug))
    if not data:
        return []
    out = []
    for j in data.get("jobs", []):
        out.append(
            JobPosting(
                url=j.get("absolute_url", ""),
                title=j.get("title", ""),
                company=j.get("company_name") or slug,
                location=(j.get("location") or {}).get("name"),
                posted_at=j.get("updated_at"),
            )
        )
    return [p for p in out if p.url]


async def _lever(client: httpx.AsyncClient, slug: str) -> list[JobPosting]:
    data = await _get_json(client, LEVER_API.format(slug=slug))
    if not isinstance(data, list):
        return []
    out = []
    for j in data:
        cats = j.get("categories") or {}
        out.append(
            JobPosting(
                url=j.get("hostedUrl", ""),
                title=j.get("text", ""),
                company=slug,
                location=cats.get("location"),
                raw_snippet=(j.get("descriptionPlain") or "")[:400],
            )
        )
    return [p for p in out if p.url]


ROLE_WORDS = ("engineer", "developer", "scientist", "architect", "lead", "mle")

# Strong domain signals in a title, weighted. A title is only worth an LLM call
# if it names the actual work, not just "engineer".
DOMAIN_SIGNALS: list[tuple[str, int]] = [
    ("machine learning", 5), ("ml engineer", 5), ("applied ai", 5),
    ("generative ai", 5), ("genai", 5), ("llm", 5), ("rag", 5),
    ("ai engineer", 5), ("ai/ml", 5), ("agent", 4), ("nlp", 4),
    ("artificial intelligence", 4), (" ai ", 3), ("ai,", 3), ("ai)", 3),
    ("data scientist", 2), ("backend", 2), ("platform", 1), ("python", 3),
    ("full stack", 1), ("fullstack", 1), ("full-stack", 1),
]

# Titles that contain a domain word but are not the job. Cheap precision win.
NEGATIVE_TITLE = (
    "recruiter", "sales", "account executive", "marketing", "designer",
    "counsel", "attorney", "paralegal", "accountant", "controller",
    "customer success", "support", "manager, people", "chief of staff",
    "technical writer", "hardware", "silicon", "chip", "datacenter",
    "data center", "gpu kernel", "compiler", "firmware", "security engineer",
    "site reliability", "network engineer", "solutions consultant",
    "intern", "internship", "director", "vp,", "head of",
    # People-management tracks. This candidate is looking for senior IC work;
    # an EM role is a different job with a different resume.
    "engineering manager", "manager, engineering", "people manager",
    "engineering lead, people",
)

# Remote postings usually name the region they are hiring into. "Remote - US"
# is not remote-from-anywhere; it needs US work authorisation. Treating those
# as a match fills the shortlist with roles that reject on paperwork.
GLOBAL_REMOTE = ("global", "worldwide", "anywhere", "any location", "emea", "apac")


def title_score(title: str, prefs: dict) -> int:
    """0 means 'not worth an LLM call'. Higher is a better prima facie match.

    Scoring rather than a boolean filter, because 5,700 listings collapse to
    ~1,200 under any permissive rule — far more than a run can process. The
    score decides which ones get looked at first.
    """
    t = f" {title.lower()} "

    for bad in prefs.get("exclude_keywords", []):
        if bad.lower() in t:
            return 0
    if any(bad in t for bad in NEGATIVE_TITLE):
        return 0
    if not any(w in t for w in ROLE_WORDS):
        return 0

    score = sum(weight for signal, weight in DOMAIN_SIGNALS if signal in t)
    if score == 0:
        return 0

    # Seniority nudge — this candidate is senior, and a mis-levelled match
    # wastes a scoring call.
    if any(s in t for s in ("senior", "staff", "principal", "lead", "sr.")):
        score += 2
    if any(s in t for s in ("junior", "associate", "i)", " i ", "entry")):
        score -= 3

    return max(score, 0)


def location_ok(location: str | None, prefs: dict) -> bool:
    """Would this candidate actually take the role?

    Without this the shortlist fills with on-site roles in cities they are not
    in — the fit scorer would reject them one expensive call at a time.
    """
    if not location:
        return True  # unstated; let the fit scorer judge

    loc = location.lower()

    # An explicitly wanted city or country always wins, remote or not.
    wanted = [w.lower() for w in prefs.get("locations", []) if w.lower() != "remote"]
    if any(w in loc for w in wanted):
        return True

    if prefs.get("remote_ok", True) and "remote" in loc:
        # Remote-from-anywhere, or remote into a region we can work in.
        if any(g in loc for g in GLOBAL_REMOTE):
            return True
        if any(w in loc for w in wanted):
            return True
        # Bare "Remote" with no region named — worth letting the fit scorer see.
        stripped = re.sub(r"[^a-z]", "", loc.replace("remote", ""))
        if not stripped:
            return True
        # Region-locked remote we cannot work in.
        return False

    if prefs.get("relocate_ok", False):
        return any(w in loc for w in ("india", "anywhere"))

    return False


async def discover_listings(state: DiscoveryState) -> dict:
    """Poll every configured board concurrently."""
    check_stop()
    targets = load_targets()
    if not any(targets.values()):
        return {
            "errors": [
                "no boards configured — add slugs to data/profile/companies.yaml"
            ],
            "listings": [],
        }

    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        tasks = [_greenhouse(client, s) for s in targets["greenhouse"]]
        tasks += [_lever(client, s) for s in targets["lever"]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    listings: list[JobPosting] = []
    errors: list[str] = []
    for r in results:
        if isinstance(r, BaseException):
            errors.append(f"board fetch failed: {type(r).__name__}")
        else:
            listings.extend(r)

    return {
        "listings": listings,
        "errors": errors,
        "results": [],
    }


async def filter_and_select(state: DiscoveryState) -> dict:
    """Deterministic funnel: title filter -> dedupe -> caps.

    Every stage here is free. Nothing reaches an LLM until it has survived all
    of them, which is what makes polling a 400-role board affordable.
    """
    check_stop()
    prefs = load_preferences()
    listings = state.get("listings", [])

    seen_keys = storage.seen_url_keys()

    scored: dict[str, tuple[int, JobPosting]] = {}
    dropped_title = 0
    dropped_location = 0
    dropped_seen = 0

    for job in listings:
        key = storage.normalise_url(job.url)
        if key in seen_keys:
            dropped_seen += 1
            continue
        if key in scored:
            continue

        score = title_score(job.title, prefs)
        if score == 0:
            dropped_title += 1
            continue
        if not location_ok(job.location, prefs):
            dropped_location += 1
            continue

        scored[key] = (score, job)

    candidates = sorted(scored.values(), key=lambda t: -t[0])

    cap = settings.max_submissions_per_run
    if not settings.dry_run:
        cap = min(cap, storage.remaining_daily_budget())

    selected = _diversify([j for _, j in candidates], cap)

    notes = [
        f"discovered {len(listings)} listings",
        f"dropped {dropped_title} on title, {dropped_location} on location, "
        f"{dropped_seen} already seen",
        f"{len(candidates)} candidates, taking {len(selected)} (cap={cap})",
    ]
    if candidates:
        top = candidates[0]
        notes.append(f"best match: {top[1].title} @ {top[1].company} (score {top[0]})")
    if len(candidates) > len(selected):
        # Never silently truncate — say what was left behind.
        notes.append(
            f"NOT processed this run: {len(candidates) - len(selected)} candidates "
            "(raise MAX_SUBMISSIONS_PER_RUN or run again)"
        )

    return {"selected": selected, "results": [], "errors": [], "notes": notes}


def _diversify(ranked: list[JobPosting], cap: int) -> list[JobPosting]:
    """Round-robin across companies, preserving rank within each.

    Straight rank ordering produces five roles at one company, because a
    company that titles its jobs well dominates the top of the list. Spreading
    across employers is both a better use of a limited daily cap and a better
    portfolio of applications.
    """
    if cap <= 0:
        return []

    by_company: dict[str, list[JobPosting]] = {}
    for job in ranked:
        by_company.setdefault(job.company, []).append(job)

    out: list[JobPosting] = []
    while len(out) < cap and any(by_company.values()):
        for queue in by_company.values():
            if not queue:
                continue
            out.append(queue.pop(0))
            if len(out) >= cap:
                break
    return out
