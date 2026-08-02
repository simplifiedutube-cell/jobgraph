"""LinkedIn job-search discovery, filtered to remote + Easy Apply.

Unlike the board APIs this is genuine scraping: it needs a logged-in session,
the DOM is obfuscated and changes without notice, and LinkedIn's terms
prohibit automated access. It is here because it was explicitly asked for.
Expect it to need selector repair more often than everything else combined.

The search URL does most of the filtering server-side, which is both more
reliable and gentler than pulling everything and filtering locally:

    f_WT=2        remote only
    f_AL=true     Easy Apply only  (anything else we cannot submit to)
    f_TPR=r604800 posted in the last week
    f_E=4,5       mid-senior and director levels
"""

from __future__ import annotations

import re
from urllib.parse import quote_plus

from playwright.async_api import BrowserContext

from jobgraph.browser import browser_context, human_pause
from jobgraph.config import check_stop
from jobgraph.profile import load_preferences
from jobgraph.schemas import Board, JobPosting

SEARCH_URL = (
    "https://www.linkedin.com/jobs/search/"
    "?keywords={keywords}&f_WT=2&f_AL=true&f_TPR=r604800&sortBy=DD"
)

CARD = "div.job-card-container, li.jobs-search-results__list-item, div.base-card"
LOGIN_WALL = ("sign in", "join now", "authwall", "please log in")


def search_urls(prefs: dict, max_queries: int = 4) -> list[str]:
    """One search per target title. LinkedIn matches literally, so a handful of
    distinct phrasings beats one broad query."""
    titles = prefs.get("titles", [])[:max_queries]
    return [SEARCH_URL.format(keywords=quote_plus(t)) for t in titles]


async def _scrape_search(ctx: BrowserContext, url: str, limit: int) -> list[JobPosting]:
    page = await ctx.new_page()
    out: list[JobPosting] = []
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        await page.wait_for_timeout(3_000)

        body = (await page.locator("body").inner_text(timeout=10_000)).lower()
        if any(w in body for w in LOGIN_WALL) and "jobs" not in body[:200]:
            raise PermissionError(
                "LinkedIn showed a login wall — run `jobgraph login linkedin` first"
            )

        # Lazy list; scroll to pull more cards in.
        for _ in range(3):
            await page.mouse.wheel(0, 2500)
            await page.wait_for_timeout(1_200)

        cards = page.locator(CARD)
        n = min(await cards.count(), limit)

        for i in range(n):
            card = cards.nth(i)
            try:
                link = card.locator("a[href*='/jobs/view/']").first
                if not await link.count():
                    continue
                href = await link.get_attribute("href") or ""
                if not href:
                    continue
                if href.startswith("/"):
                    href = "https://www.linkedin.com" + href
                href = href.split("?")[0]

                text = " ".join((await card.inner_text()).split())
                title = (await link.inner_text()).strip() or text[:70]

                company = ""
                for sel in (
                    "span.job-card-container__primary-description",
                    "a.job-card-container__company-name",
                    "h4.base-search-card__subtitle",
                    "div.artdeco-entity-lockup__subtitle",
                ):
                    c = card.locator(sel).first
                    if await c.count():
                        company = " ".join((await c.inner_text()).split())
                        break

                location = ""
                for sel in (
                    "li.job-card-container__metadata-item",
                    "span.job-search-card__location",
                ):
                    lo = card.locator(sel).first
                    if await lo.count():
                        location = " ".join((await lo.inner_text()).split())
                        break

                out.append(
                    JobPosting(
                        url=href,
                        title=re.sub(r"\s+", " ", title)[:120],
                        company=company or "(unknown)",
                        location=location or "Remote",
                        board=Board.LINKEDIN,
                        raw_snippet=text[:300],
                    )
                )
            except Exception:  # noqa: BLE001 - one bad card is not fatal
                continue

            await human_pause()
    finally:
        await page.close()

    return out


async def discover_linkedin(limit: int = 25) -> tuple[list[JobPosting], list[str]]:
    """Remote + Easy Apply postings across the candidate's target titles."""
    check_stop()
    prefs = load_preferences()
    urls = search_urls(prefs)

    listings: list[JobPosting] = []
    notes: list[str] = []

    async with browser_context() as ctx:
        for url in urls:
            try:
                found = await _scrape_search(ctx, url, limit)
            except PermissionError as exc:
                notes.append(str(exc))
                break
            except Exception as exc:  # noqa: BLE001
                notes.append(f"search failed: {type(exc).__name__}: {exc}")
                continue
            notes.append(f"{len(found)} from {url.split('keywords=')[1].split('&')[0]}")
            listings.extend(found)
            await human_pause()

    # Dedupe by URL; the same posting shows up under several title searches.
    seen: set[str] = set()
    unique: list[JobPosting] = []
    for job in listings:
        if job.url in seen:
            continue
        seen.add(job.url)
        unique.append(job)

    notes.append(f"{len(unique)} unique remote Easy Apply postings")
    return unique[:limit], notes
