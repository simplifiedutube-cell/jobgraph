"""The three generation nodes. All three run in parallel after the fit gate.

Every one of them is bound by PROVENANCE_RULE: they may reword corpus content
but may not add to it. `critique` enforces that with code, not vibes.
"""

from __future__ import annotations

from pathlib import Path

from jobgraph.config import check_stop, settings
from jobgraph.llm import ENGINE
from jobgraph.profile import (
    PROVENANCE_RULE,
    corpus_for_prompt,
    load_preferences,
    load_qa_bank,
)
from jobgraph.render import render_pdf
from jobgraph.schemas import CoverLetter, ScreeningAnswers, TailoredResume
from jobgraph.state import ApplicationState

# Screening questions worth pre-answering. ATS forms ask these constantly and
# having them ready means the submit nodes rarely block on generation.
STANDARD_QUESTIONS = [
    "Why are you interested in this role?",
    "Why do you want to work at this company?",
    "What relevant experience do you bring to this role?",
    "Describe a challenging technical problem you solved recently.",
    "What is your notice period / earliest start date?",
    "What are your salary expectations?",
    "Are you legally authorised to work in this location?",
    "Are you willing to relocate or work on-site?",
]


def _revision_feedback(state: ApplicationState) -> str:
    """Feed the previous critique back in on a revision pass.

    The important half is the *preserve* instruction. Without it the model
    treats "these bullets are filler" as licence to delete freely and
    regresses keyword coverage in the process — observed going 100% -> 82%
    by dropping the only React/Next.js mentions while removing filler.
    """
    critique = state.get("critique")
    prov = state.get("provenance_errors") or []
    if critique is None and not prov:
        return ""

    n = state.get("revision_count", 0) + 1
    parts = [
        "\n--- REVISION REQUESTED ---",
        f"This is revision {n}. Your previous attempt was rejected.",
        "",
        "Make targeted fixes. Do not regenerate from scratch and do not rewrite "
        "the parts that were already working — a revision that fixes one issue "
        "while losing ground elsewhere is a worse package, not a better one.",
    ]

    if prov:
        parts += [
            "",
            "UNSOURCED CLAIMS — hard failure. Every source_id must appear in the "
            "corpus above:",
            *(f"  - {p}" for p in prov[-10:]),
        ]

    if critique is not None:
        prev = critique.keyword_coverage
        parts += [
            "",
            f"Keyword coverage on the previous attempt: {prev}%.",
            f"DO NOT let coverage fall below {prev}%. If a keyword is currently "
            "covered, it must still be covered when you are done — including via "
            "highlighted_skills, which counts toward coverage.",
        ]
        missing = _missing_keywords(state)
        if missing:
            parts += [
                "",
                "Keywords the JD asks for that are still missing: "
                + ", ".join(missing)
                + ".",
                "Add each one ONLY where the corpus genuinely supports it "
                "(highlighted_skills counts, and the skills list is part of the "
                "corpus). If the corpus does not support a keyword, leave it out "
                "— do not manufacture evidence to hit a number.",
            ]
        if critique.issues:
            parts += ["", "Issues to address:", *(f"  - {i}" for i in critique.issues)]

    return "\n".join(parts)


def _missing_keywords(state: ApplicationState) -> list[str]:
    """Recompute here rather than threading it through state — same function
    the critique node uses, so the two can never drift apart."""
    from jobgraph.nodes.critique import _keyword_coverage

    try:
        _, missing = _keyword_coverage(state)
    except Exception:  # noqa: BLE001
        return []
    return missing


async def tailor_resume(state: ApplicationState) -> dict:
    check_stop()
    jd = state.get("jd")
    fit = state.get("fit")
    if jd is None or fit is None:
        return {"errors": ["tailor_resume: missing jd or fit"], "status": "failed"}

    evidence = sorted({i for m in fit.matches for i in m.evidence_bullet_ids})

    # Corpus first, deliberately. The corpus + provenance rule block is
    # byte-identical across tailor / cover_letter / screening and across
    # revisions, so leading with it lets those ~5k tokens serve as a cache read
    # after the first call instead of being reprocessed every time. Everything
    # that varies per call goes after it.
    prompt = f"""\
{corpus_for_prompt()}

{PROVENANCE_RULE}

--- TASK ---
Tailor this candidate's resume for the role below.

HOW TO TAILOR
- Select the bullets that matter for THIS role and drop the ones that don't.
  A focused one-page-equivalent beats a complete two-page one.
- Order sections newest-first by role, then a projects section if the role's
  technologies show up better there than in the day job.
- Rewrite each selected bullet to lead with the outcome that this JD cares
  about. Same facts, different emphasis.
- Section headings for work history MUST be exactly "<Title> | <Company>" so
  the renderer can attach dates. For anything else use a plain heading such as
  "Selected Projects".
- highlighted_skills: 8-12 skills, most JD-relevant first, drawn only from the
  candidate's skills and bullets.
- The summary is 2-3 sentences positioning this candidate for this role.

- Keywords the JD names are worth surfacing wherever the corpus genuinely
  supports them. highlighted_skills is the natural home for a technology the
  candidate knows but has no dedicated bullet about.

The fit analysis already identified these bullets as relevant evidence — start
there, but use your own judgement about what else belongs:
  {", ".join(evidence) or "(none identified)"}

--- ROLE ---
{jd.model_dump_json(indent=2)}
{_revision_feedback(state)}
"""

    try:
        tailored = await ENGINE.call(
            node="tailor_resume",
            prompt=prompt,
            schema=TailoredResume,
            profile=settings.profile_tailor,
        )
    except Exception as exc:  # noqa: BLE001
        return {"errors": [f"tailor_resume: {exc}"], "status": "failed"}

    n = sum(len(s.bullets) for s in tailored.sections)
    return {"tailored": tailored, "notes": [f"tailored {n} bullets in {len(tailored.sections)} sections"]}


async def write_cover_letter(state: ApplicationState) -> dict:
    check_stop()
    jd = state.get("jd")
    if jd is None:
        return {"errors": ["write_cover_letter: missing jd"], "status": "failed"}

    prefs = load_preferences()

    prompt = f"""\
{corpus_for_prompt()}

{PROVENANCE_RULE}

--- TASK ---
Write the body of a cover letter for this role.

STYLE
- 3-4 short paragraphs. No greeting, no signoff — the form adds those.
- Open with why this specific role, referencing something concrete from the
  posting. Never "I am writing to apply for...".
- One paragraph of the most relevant evidence, with the real numbers from the
  corpus.
- Close with what the candidate wants to do there. One sentence.
- Plain, direct sentences. No "passionate", "leverage", "synergy",
  "I am excited to", "proven track record".
- Do not restate the resume line by line. Pick the two things that matter here.

List in source_ids every corpus bullet id whose content you drew on.

--- CANDIDATE'S OWN FRAMING ---
{prefs.get("narrative", "")}

--- ROLE ---
{jd.model_dump_json(indent=2)}
{_revision_feedback(state)}
"""

    try:
        letter = await ENGINE.call(
            node="write_cover_letter",
            prompt=prompt,
            schema=CoverLetter,
            profile=settings.profile_cover_letter,
        )
    except Exception as exc:  # noqa: BLE001
        # Non-fatal. Most ATS forms treat the cover letter as optional, and
        # failing a well-scored application because one of three parallel
        # branches hit a transient error is the wrong trade.
        return {"errors": [f"write_cover_letter: {exc}"], "notes": ["no cover letter"]}

    return {"cover_letter": letter, "notes": [f"cover letter {len(letter.body)} chars"]}


def _relevant_guidance(jd_text: str, limit: int = 12) -> str:
    """Pull the most on-topic entries out of the 100-entry Q&A bank.

    Cheap lexical overlap rather than embeddings: the bank is small, and this
    keeps the node dependency-free and deterministic.
    """
    bank = load_qa_bank().get("entries", [])
    if not bank:
        return "(no question bank available)"

    words = {w for w in jd_text.lower().split() if len(w) > 4}
    scored = []
    for entry in bank:
        blob = (entry.get("question", "") + " " + " ".join(entry.get("guidance", []))).lower()
        overlap = sum(1 for w in words if w in blob)
        scored.append((overlap, entry))

    scored.sort(key=lambda t: t[0], reverse=True)
    chosen = [e for score, e in scored[:limit] if score > 0] or [e for _, e in scored[:4]]

    out = []
    for e in chosen:
        out.append(f"- ({e['id']}) Q: {e['question']}")
        for g in e.get("guidance", [])[:2]:
            out.append(f"    {g[:400]}")
    return "\n".join(out)


async def answer_screening(state: ApplicationState) -> dict:
    check_stop()
    jd = state.get("jd")
    if jd is None:
        return {"errors": ["answer_screening: missing jd"], "status": "failed"}

    prefs = load_preferences()
    guidance = _relevant_guidance(state.get("jd_text", "") or jd.model_dump_json())

    questions = "\n".join(f"  - {q}" for q in STANDARD_QUESTIONS)

    prompt = f"""\
{corpus_for_prompt()}

{PROVENANCE_RULE}

--- TASK ---
Draft answers to the screening questions an application form is likely to ask
for this role.

RULES
- 2-4 sentences per answer. These go in small textareas, not essays.
- First person, plain language, specific. No filler openers.
- Factual fields (notice period, work authorisation, relocation, salary) come
  from the CANDIDATE FACTS below verbatim — do not invent or soften them.
- source_ids: corpus bullet ids for experience claims, or qa bank ids
  (they look like qa.3.1) where you followed that coaching. Factual answers
  may have an empty source_ids.

QUESTIONS
{questions}

--- CANDIDATE FACTS (authoritative, use verbatim) ---
Notice period: {prefs.get("notice_period", "unstated")}
Work authorisation: {prefs.get("work_authorization", "unstated")}
Willing to relocate: {prefs.get("relocate_ok", False)}
Remote acceptable: {prefs.get("remote_ok", True)}
Compensation floor: {prefs.get("compensation", {}).get("annual_floor", "unstated")} \
{prefs.get("compensation", {}).get("currency", "")}
Own framing: {prefs.get("narrative", "")}

--- ROLE ---
{jd.model_dump_json(indent=2)}

--- INTERVIEW COACHING NOTES (tone and structure only, not facts) ---
{guidance}
{_revision_feedback(state)}
"""

    try:
        answers = await ENGINE.call(
            node="answer_screening",
            prompt=prompt,
            schema=ScreeningAnswers,
            profile=settings.profile_screening,
        )
    except Exception as exc:  # noqa: BLE001
        # Non-fatal, same reasoning as the cover letter. A submit adapter that
        # meets an unanswered required question will queue the job to manual
        # rather than guess.
        return {"errors": [f"answer_screening: {exc}"], "notes": ["no screening answers"]}

    return {"screening": answers, "notes": [f"{len(answers.answers)} screening answers"]}


async def render_resume_pdf(state: ApplicationState) -> dict:
    """Join point for the parallel branches — needs `tailored` to exist."""
    check_stop()
    tailored = state.get("tailored")
    if tailored is None:
        return {"errors": ["render_resume_pdf: no tailored resume"], "status": "failed"}

    job = state["job"]
    run_dir = Path(state["run_dir"])
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in job.company)[:40]
    out = run_dir / f"resume_{safe}.pdf"

    try:
        await render_pdf(tailored, out, job=f"{job.title} at {job.company}")
    except Exception as exc:  # noqa: BLE001
        return {"errors": [f"render_resume_pdf: {exc}"], "status": "failed"}

    return {"resume_pdf": str(out), "notes": [f"pdf {out.name} ({out.stat().st_size // 1024} KB)"]}
