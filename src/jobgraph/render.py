"""Jinja2 -> HTML -> PDF via Playwright.

Playwright is already a dependency for scraping and form-fill, so it renders
the PDF too. That's why there's no LaTeX or wkhtmltopdf here — one browser
does all three jobs and there is no extra toolchain to install.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from playwright.async_api import async_playwright

from jobgraph.profile import load_profile
from jobgraph.schemas import Profile, TailoredResume, TailoredSection

TEMPLATE_DIR = Path(__file__).parent / "templates"

_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    undefined=StrictUndefined,
    autoescape=True,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _role_lookup(profile: Profile) -> dict[str, dict[str, str | None]]:
    """Map a section heading back to the role it came from.

    The model emits one section per role with a free-text heading, so match on
    several plausible spellings rather than demanding an exact string.
    """
    meta: dict[str, dict[str, str | None]] = {}
    for role in profile.roles:
        record = {
            "title": role.title,
            "company": role.company,
            "start": role.start,
            "end": role.end,
            "context": role.context,
        }
        for key in (
            f"{role.title} | {role.company}",
            f"{role.title} — {role.company}",
            f"{role.title} - {role.company}",
            f"{role.company} | {role.title}",
            role.company,
            role.title,
        ):
            meta.setdefault(key.strip().lower(), record)
    return meta


def _blocks(tailored: TailoredResume, profile: Profile) -> list[dict]:
    """Classify each section as a role entry or a standalone section.

    Doing this here keeps the template declarative — the alternative is nested
    conditionals in Jinja that are much harder to read and to change.
    """
    lookup = _role_lookup(profile)
    out: list[dict] = []
    for section in tailored.sections:
        meta = lookup.get(section.heading.strip().lower())
        out.append(
            {
                "kind": "role" if meta else "other",
                "heading": section.heading,
                "meta": meta,
                "bullets": section.bullets,
            }
        )
    return out


def render_html(
    tailored: TailoredResume,
    *,
    profile: Profile | None = None,
    job: str | None = None,
) -> str:
    prof = profile or load_profile()
    template = _env.get_template("resume.html.j2")
    return template.render(
        profile=prof,
        tailored=tailored,
        blocks=_blocks(tailored, prof),
        job=job,
    )


async def render_pdf(
    tailored: TailoredResume,
    out_path: Path,
    *,
    profile: Profile | None = None,
    job: str | None = None,
) -> Path:
    """Write a PDF and return its path. Also writes the HTML beside it, which
    is what you actually want when a layout looks wrong."""
    prof = profile or load_profile()
    html = render_html(tailored, profile=prof, job=job)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    html_path = out_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            # file:// so relative assets and @page rules behave as in a real load
            await page.goto(html_path.as_uri(), wait_until="load")
            await page.pdf(
                path=str(out_path),
                format="A4",
                print_background=True,
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                prefer_css_page_size=True,
            )
        finally:
            await browser.close()

    return out_path


def baseline_resume(profile: Profile | None = None) -> TailoredResume:
    """The untailored resume: every bullet, in corpus order.

    Used by `jobgraph render` to check the template, and as the fallback if
    tailoring fails but the run should still produce something submittable.
    """
    prof = profile or load_profile()
    summary = " ".join(" ".join(b.text.split()) for b in prof.summary_bullets)

    sections: list[TailoredSection] = []
    for role in prof.roles:
        if not role.bullets:
            continue
        sections.append(
            TailoredSection.model_validate(
                {
                    "heading": f"{role.title} | {role.company}",
                    "bullets": [
                        {"source_id": b.id, "text": " ".join(b.text.split())}
                        for b in role.bullets
                    ],
                }
            )
        )

    project_bullets = [
        {"source_id": b.id, "text": f"{p.name}: {' '.join(b.text.split())}"}
        for p in prof.projects
        for b in p.bullets
    ]
    if project_bullets:
        sections.append(
            TailoredSection.model_validate(
                {"heading": "Selected Projects", "bullets": project_bullets}
            )
        )

    skills = [s for group in prof.skills.values() for s in group][:12]
    return TailoredResume(
        summary=summary,
        highlighted_skills=skills,
        sections=sections,
    )
