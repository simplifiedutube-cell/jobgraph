"""Playwright: one persistent, logged-in browser profile shared by every use.

Three jobs run through here — scraping boards, fetching JD pages, and filling
ATS forms — and all three want the same authenticated session. Credentials are
never stored by this project; you log in by hand once via `jobgraph login` and
Chromium keeps the cookies in `data/browser/`.
"""

from __future__ import annotations

import asyncio
import random
import re
from contextlib import asynccontextmanager
from typing import AsyncIterator

from playwright.async_api import BrowserContext, Page, async_playwright
from selectolax.parser import HTMLParser

from jobgraph.config import settings
from jobgraph.schemas import AtsVendor

# A real UA; the default Playwright one is trivially fingerprinted.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"
)

LOGIN_URLS = {
    "linkedin": "https://www.linkedin.com/login",
    "naukri": "https://www.naukri.com/nlogin/login",
    "wellfound": "https://wellfound.com/login",
}

_VENDOR_PATTERNS: list[tuple[re.Pattern[str], AtsVendor]] = [
    (re.compile(r"greenhouse\.io|boards\.greenhouse", re.I), AtsVendor.GREENHOUSE),
    (re.compile(r"lever\.co|jobs\.lever", re.I), AtsVendor.LEVER),
    (re.compile(r"ashbyhq\.com|jobs\.ashby", re.I), AtsVendor.ASHBY),
    (re.compile(r"myworkdayjobs\.com|workday", re.I), AtsVendor.WORKDAY),
    (re.compile(r"linkedin\.com/jobs", re.I), AtsVendor.LINKEDIN_EASY_APPLY),
]


async def human_pause() -> None:
    lo, hi = settings.action_delay_ms
    await asyncio.sleep(random.uniform(lo, hi) / 1000)


@asynccontextmanager
async def browser_context(headless: bool | None = None) -> AsyncIterator[BrowserContext]:
    """The shared browser context. Two modes.

    CDP attach (`CHROME_CDP_URL` set) reuses a Chrome *you* started, with the
    sessions you are already logged into — no separate login, and this project
    never reads your cookie store. Start Chrome yourself with:

        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \\
            --remote-debugging-port=9222

    Note Chrome must be fully quit first: a running instance holds a lock on
    its profile directory and will ignore the flag.

    Otherwise the project launches its own persistent profile in
    `data/browser/`, which you log into once via `jobgraph login`.
    """
    if settings.chrome_cdp_url:
        async with async_playwright() as pw:
            browser = await pw.chromium.connect_over_cdp(settings.chrome_cdp_url)
            ctx = browser.contexts[0] if browser.contexts else await browser.new_context()
            try:
                yield ctx
            finally:
                # Detach only. Closing would kill the user's own browser.
                await browser.close()
        return

    settings.browser_profile_dir.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            user_data_dir=str(settings.browser_profile_dir),
            headless=settings.headless if headless is None else headless,
            user_agent=USER_AGENT,
            viewport={"width": 1440, "height": 900},
            locale="en-IN",
            timezone_id="Asia/Kolkata",
            args=["--disable-blink-features=AutomationControlled"],
        )
        # navigator.webdriver is the cheapest bot tell there is.
        await ctx.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
        )
        try:
            yield ctx
        finally:
            await ctx.close()


def detect_vendor(url: str, html: str = "") -> AtsVendor:
    """URL first, then markup. Cheap and deterministic — no LLM needed."""
    for pattern, vendor in _VENDOR_PATTERNS:
        if pattern.search(url):
            return vendor
    for pattern, vendor in _VENDOR_PATTERNS:
        if pattern.search(html[:20000]):
            return vendor
    return AtsVendor.UNKNOWN


def html_to_text(html: str) -> str:
    """Readable text from a job page.

    Strips chrome that would otherwise dominate the JD and burn tokens: nav,
    cookie banners, script/style, footers.
    """
    tree = HTMLParser(html)
    for sel in ("script", "style", "noscript", "nav", "footer", "header", "svg"):
        for node in tree.css(sel):
            node.decompose()

    body = tree.body or tree.root
    text = body.text(separator="\n") if body else ""

    lines = [ln.strip() for ln in text.splitlines()]
    out: list[str] = []
    blank = 0
    for ln in lines:
        if not ln:
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(ln)
    return "\n".join(out).strip()


async def fetch_page(
    url: str,
    *,
    ctx: BrowserContext | None = None,
    wait_selector: str | None = None,
    timeout_ms: int = 45_000,
) -> tuple[str, str]:
    """Return (html, text) for a URL, reusing the logged-in session."""

    async def _do(context: BrowserContext) -> tuple[str, str]:
        page: Page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            if wait_selector:
                try:
                    await page.wait_for_selector(wait_selector, timeout=8_000)
                except Exception:
                    pass  # selector is a hint, not a requirement
            # Job pages hydrate late; networkidle is unreliable on boards that
            # poll, so settle on a short fixed wait instead.
            await page.wait_for_timeout(1_200)
            html = await page.content()
            return html, html_to_text(html)
        finally:
            await page.close()

    if ctx is not None:
        return await _do(ctx)
    async with browser_context() as owned:
        return await _do(owned)


async def interactive_login(board: str) -> None:
    """Open a real window at the board's login page and wait for the user.

    Deliberately manual: this project never handles your credentials. You log
    in, solve whatever MFA or captcha appears, and press Enter here.
    """
    url = LOGIN_URLS.get(board)
    if not url:
        raise ValueError(f"unknown board {board!r}; known: {', '.join(LOGIN_URLS)}")

    async with browser_context(headless=False) as ctx:
        page = await ctx.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        print(f"\nA browser window is open at {url}")
        print("Log in there (including any MFA), then come back here.")
        await asyncio.get_event_loop().run_in_executor(
            None, input, "Press Enter once you are logged in... "
        )
        # Touch the cookie jar so it flushes before the context closes.
        cookies = await ctx.cookies()
        print(f"Saved {len(cookies)} cookies to {settings.browser_profile_dir}")
