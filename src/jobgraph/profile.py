"""Loading the profile corpus and enforcing provenance against it.

`verify_provenance` is the function that makes the whole anti-fabrication
design real. Everything else is plumbing.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from jobgraph.config import settings
from jobgraph.schemas import (
    CoverLetter,
    Profile,
    ProvenanceViolation,
    ScreeningAnswers,
    TailoredResume,
)


@functools.cache
def load_profile(path: Path | None = None) -> Profile:
    p = path or settings.resume_path
    if not p.exists():
        raise FileNotFoundError(
            f"No resume corpus at {p}. This is the source of truth for every "
            "claim the graph may make; it cannot run without one."
        )
    data = yaml.safe_load(p.read_text())
    profile = Profile.model_validate(data)
    _assert_unique_ids(profile)
    return profile


def _assert_unique_ids(profile: Profile) -> None:
    seen: set[str] = set()
    dupes: set[str] = set()
    for b in profile.all_bullets():
        if b.id in seen:
            dupes.add(b.id)
        seen.add(b.id)
    if dupes:
        raise ValueError(
            f"Duplicate bullet ids in the corpus: {sorted(dupes)}. Ids are the "
            "provenance key and must be unique."
        )


@functools.cache
def load_preferences(path: Path | None = None) -> dict[str, Any]:
    p = path or settings.preferences_path
    if not p.exists():
        raise FileNotFoundError(f"No preferences at {p}.")
    return yaml.safe_load(p.read_text())


@functools.cache
def load_qa_bank(path: Path | None = None) -> dict[str, Any]:
    p = path or settings.qa_bank_path
    if not p.exists():
        return {"entries": []}
    return yaml.safe_load(p.read_text()) or {"entries": []}


def qa_bank_ids() -> set[str]:
    bank = load_qa_bank()
    return {e["id"] for e in bank.get("entries", []) if "id" in e}


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def verify_provenance(
    *,
    tailored: TailoredResume | None = None,
    cover_letter: CoverLetter | None = None,
    screening: ScreeningAnswers | None = None,
    profile: Profile | None = None,
) -> list[ProvenanceViolation]:
    """Every claimed source id must exist in the corpus.

    Deliberately dumb and deterministic: an LLM judging whether a rewrite is
    faithful is exactly the failure mode we're guarding against. This checks
    that the model *cited* something real. `critique` handles the softer
    question of whether the rewrite drifted from what the source says.
    """
    prof = profile or load_profile()
    valid = prof.bullet_ids()
    valid_answers = valid | qa_bank_ids()
    violations: list[ProvenanceViolation] = []

    if tailored is not None:
        for section in tailored.sections:
            for bullet in section.bullets:
                if bullet.source_id not in valid:
                    violations.append(
                        ProvenanceViolation(
                            where=f"resume/{section.heading}",
                            claimed_id=bullet.source_id,
                            detail=f"unknown source id; text was {bullet.text[:80]!r}",
                        )
                    )

    if cover_letter is not None:
        for sid in cover_letter.source_ids:
            if sid not in valid:
                violations.append(
                    ProvenanceViolation(
                        where="cover_letter",
                        claimed_id=sid,
                        detail="unknown source id",
                    )
                )

    if screening is not None:
        for ans in screening.answers:
            for sid in ans.source_ids:
                if sid not in valid_answers:
                    violations.append(
                        ProvenanceViolation(
                            where=f"screening/{ans.question[:40]}",
                            claimed_id=sid,
                            detail="unknown source id",
                        )
                    )

    return violations


# --------------------------------------------------------------------------
# Prompt-side rendering of the corpus
# --------------------------------------------------------------------------


def corpus_for_prompt(profile: Profile | None = None) -> str:
    """The bullet bank as the model sees it: id, text, metric, tags.

    Kept stable and compact — this block is the bulk of every composition
    prompt, so churn here costs tokens on every call.
    """
    prof = profile or load_profile()
    lines: list[str] = []

    def emit(bullets: list[Any], header: str) -> None:
        if not bullets:
            return
        lines.append(f"\n## {header}")
        for b in bullets:
            metric = f"  [metric: {b.metric}]" if b.metric else ""
            tags = f"  [tags: {', '.join(b.tags)}]" if b.tags else ""
            text = " ".join(b.text.split())
            lines.append(f"- ({b.id}) {text}{metric}{tags}")

    lines.append(f"# CANDIDATE CORPUS — {prof.name}")
    lines.append(f"{prof.headline}")
    lines.append(f"{prof.location}")

    emit(prof.summary_bullets, "Summary")

    for role in prof.roles:
        header = f"{role.title} | {role.company} ({role.start} – {role.end})"
        if role.context:
            lines.append(f"\n## {header}")
            lines.append(f"_context_: {' '.join(role.context.split())}")
            emit(role.bullets, f"{role.company} achievements")
        else:
            emit(role.bullets, header)

    for proj in prof.projects:
        stack = f" — stack: {', '.join(proj.stack)}" if proj.stack else ""
        emit(proj.bullets, f"Project: {proj.name}{stack}")

    if prof.skills:
        lines.append("\n## Skills (not citable as bullets)")
        for group, items in prof.skills.items():
            lines.append(f"- {group}: {', '.join(items)}")

    if prof.education:
        lines.append("\n## Education")
        for e in prof.education:
            lines.append(f"- {e}")

    return "\n".join(lines)


PROVENANCE_RULE = """\
GROUNDING RULES — these are hard constraints, not style guidance.

1. Every bullet you emit must carry a `source_id` that appears verbatim in the
   CANDIDATE CORPUS above. Never invent an id. Never cite an id you did not see.
2. You may reword, compress, re-emphasise and reorder. You may NOT introduce a
   technology, employer, metric, date, scale or responsibility that is absent
   from the source bullet you are citing.
3. If the job asks for something the candidate has not done, leave it out. Do
   not imply it. A shorter honest resume beats a padded one, and a downstream
   validator will reject unsourced claims anyway.
4. Numbers are exact or absent. If the source says "around 35%", you may write
   "~35%" but never "40%", "significant" or "dramatic".
"""


class ProvenanceError(ValueError):
    def __init__(self, violations: list[ProvenanceViolation]) -> None:
        self.violations = violations
        detail = "; ".join(f"{v.where}:{v.claimed_id}" for v in violations)
        super().__init__(f"{len(violations)} unsourced claim(s): {detail}")


def as_prompt_block(model: BaseModel) -> str:
    """Compact JSON for embedding a prior node's output in a later prompt."""
    return model.model_dump_json(indent=2, exclude_none=True)
