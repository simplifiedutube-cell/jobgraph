"""Pydantic models for every structured LLM output.

Each model's `model_json_schema()` is injected verbatim into the prompt, so the
field descriptions here *are* the spec the model reads. Write them for the
model, not for a human skimming the file.

Keep shapes shallow. Deeply nested optionals produce more retries.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Profile corpus (loaded from YAML, never LLM-generated)
# --------------------------------------------------------------------------


class Bullet(BaseModel):
    """One achievement from the resume corpus. The atom of provenance."""

    id: str
    """Stable identifier, e.g. 'kns.rag.relevance'. Never regenerate these —
    tailored output references them and evals assert against them."""
    text: str
    tags: list[str] = Field(default_factory=list)
    """Free-form skill tags used for retrieval, e.g. ['rag', 'evals']."""
    metric: str | None = None
    """The quantified claim, if any, e.g. '+35% answer relevance'."""


class Role(BaseModel):
    id: str
    title: str
    company: str
    start: str
    end: str
    context: str | None = None
    bullets: list[Bullet] = Field(default_factory=list)


class Project(BaseModel):
    id: str
    name: str
    stack: list[str] = Field(default_factory=list)
    url: str | None = None
    bullets: list[Bullet] = Field(default_factory=list)


class Profile(BaseModel):
    """The bullet bank. Source of truth — nothing may be claimed that is not
    traceable to a Bullet.id in here."""

    name: str
    headline: str
    email: str
    phone: str
    location: str
    links: dict[str, str] = Field(default_factory=dict)
    summary_bullets: list[Bullet] = Field(default_factory=list)
    skills: dict[str, list[str]] = Field(default_factory=dict)
    roles: list[Role] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)

    def all_bullets(self) -> list[Bullet]:
        out = list(self.summary_bullets)
        for r in self.roles:
            out.extend(r.bullets)
        for p in self.projects:
            out.extend(p.bullets)
        return out

    def bullet_ids(self) -> set[str]:
        return {b.id for b in self.all_bullets()}


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


class Board(StrEnum):
    LINKEDIN = "linkedin"
    NAUKRI = "naukri"
    WELLFOUND = "wellfound"


class SearchQuery(BaseModel):
    board: Board
    keywords: str = Field(description="Search string as typed into the board's search box.")
    location: str = Field(description="Location filter, or 'Remote'.")
    rationale: str = Field(description="One sentence: why this query fits the candidate.")


class SearchPlan(BaseModel):
    queries: list[SearchQuery] = Field(
        description="4-8 queries spanning the candidate's target titles and locations. "
        "Vary phrasing across queries; boards match literally."
    )


class JobPosting(BaseModel):
    """A listing as scraped, before the JD body is fetched."""

    url: str
    title: str
    company: str
    location: str | None = None
    board: Board | None = None
    posted_at: str | None = None
    raw_snippet: str | None = None


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


class AtsVendor(StrEnum):
    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    ASHBY = "ashby"
    LINKEDIN_EASY_APPLY = "linkedin_easy_apply"
    WORKDAY = "workday"
    UNKNOWN = "unknown"


class Requirement(BaseModel):
    text: str = Field(description="The requirement, quoted or lightly normalised from the JD.")
    kind: str = Field(description="One of: must_have, nice_to_have.")


class ParsedJD(BaseModel):
    title: str
    company: str
    seniority: str = Field(description="e.g. 'Senior', 'Staff', 'Mid'. 'Unspecified' if absent.")
    location: str = Field(description="Location as stated. 'Remote' if fully remote.")
    remote: bool = Field(description="True if the role can be done fully remotely.")
    requirements: list[Requirement]
    tech_keywords: list[str] = Field(
        description="Concrete technologies, frameworks and platforms named in the JD. "
        "Lowercase, no duplicates."
    )
    responsibilities: list[str] = Field(description="Up to 8 short phrases.")
    comp: str | None = Field(default=None, description="Compensation as stated, or null.")
    ats_vendor: AtsVendor = Field(
        description="Which applicant tracking system hosts the application form, "
        "inferred from the URL and page markup. 'unknown' if unclear."
    )


class RequirementMatch(BaseModel):
    requirement: str
    met: bool
    evidence_bullet_ids: list[str] = Field(
        description="Ids of bullets from the candidate corpus that support this. "
        "MUST be ids that appear in the supplied corpus. Empty list if not met."
    )
    note: str = Field(description="One short sentence justifying the verdict.")


class FitReport(BaseModel):
    score: int = Field(ge=0, le=100, description="Overall fit, 0-100.")
    matches: list[RequirementMatch]
    strengths: list[str] = Field(description="Up to 4 short phrases.")
    gaps: list[str] = Field(description="Up to 4 short phrases. Be honest.")
    recommendation: str = Field(description="One of: apply, stretch, skip.")


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------


class TailoredBullet(BaseModel):
    source_id: str = Field(
        description="The id of the corpus bullet this is derived from. "
        "MUST exist in the supplied corpus. Never invent an id."
    )
    text: str = Field(
        description="The rewritten bullet. You may reword, reorder emphasis and "
        "adjust length. You may NOT introduce any fact, technology, metric or "
        "claim absent from the source bullet."
    )


class TailoredSection(BaseModel):
    heading: str
    bullets: list[TailoredBullet]


class TailoredResume(BaseModel):
    summary: str = Field(description="2-3 sentences, grounded only in corpus content.")
    highlighted_skills: list[str] = Field(
        description="Skills to surface first, ordered by relevance to this JD. "
        "Only skills present in the candidate's corpus."
    )
    sections: list[TailoredSection]


class ScreeningAnswer(BaseModel):
    question: str
    answer: str
    source_ids: list[str] = Field(
        description="Corpus bullet ids or qa_bank entry ids grounding this answer. "
        "Empty only for purely factual fields such as notice period."
    )


class ScreeningAnswers(BaseModel):
    answers: list[ScreeningAnswer]


class CoverLetter(BaseModel):
    body: str = Field(description="3-4 short paragraphs. No greeting, no signoff.")
    source_ids: list[str] = Field(
        description="Every corpus bullet id whose content appears in the body."
    )


# --------------------------------------------------------------------------
# Critique
# --------------------------------------------------------------------------


class CritiqueReport(BaseModel):
    keyword_coverage: int = Field(
        ge=0, le=100, description="Percent of the JD's tech_keywords surfaced in the package."
    )
    tone_ok: bool = Field(description="False if the writing is inflated, generic or padded.")
    length_ok: bool = Field(description="False if the resume would exceed two pages.")

    blocking_issues: list[str] = Field(
        default_factory=list,
        description=(
            "Defects that make this package WORSE THAN NOT APPLYING, or that "
            "would get it filtered out. Only these trigger a revision. A defect "
            "is blocking if it is one of: a factually wrong or misleading claim; "
            "a required JD qualification the candidate demonstrably has but the "
            "package never mentions; writing so generic a screener would discard "
            "it; or a resume that would run past two pages. "
            "An improvement that would merely make a good package slightly "
            "better is NOT blocking. Leave this empty unless you would genuinely "
            "rather send nothing than send this."
        ),
    )
    issues: list[str] = Field(
        default_factory=list,
        description=(
            "Non-blocking observations worth recording but NOT worth another "
            "generation pass. These are logged and ignored by the gate."
        ),
    )
    verdict: str = Field(
        description=(
            "'revise' if and only if blocking_issues is non-empty. "
            "Otherwise 'pass'."
        )
    )


class ProvenanceViolation(BaseModel):
    """Produced by code, not the model. A hard failure."""

    where: str
    claimed_id: str
    detail: str


# --------------------------------------------------------------------------
# Submission
# --------------------------------------------------------------------------


class SubmissionOutcome(StrEnum):
    SUBMITTED = "submitted"
    DRY_RUN = "dry_run"
    FAILED = "failed"
    QUEUED_MANUAL = "queued_manual"


class SubmissionResult(BaseModel):
    outcome: SubmissionOutcome
    vendor: AtsVendor
    confirmation: str | None = None
    screenshots: list[str] = Field(default_factory=list)
    fields_filled: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    at: datetime | None = None
