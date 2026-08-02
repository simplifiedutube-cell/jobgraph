"""Graph state.

Two states: one per job (the application subgraph) and one per run (the
discovery graph). Reducers on the accumulating fields let parallel branches
merge without clobbering each other.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from jobgraph.schemas import (
    CoverLetter,
    CritiqueReport,
    FitReport,
    JobPosting,
    ParsedJD,
    ScreeningAnswers,
    SubmissionResult,
    TailoredResume,
)

Status = Literal[
    "scraped",
    "rejected",
    "drafted",
    "submitted",
    "failed",
    "manual",
]


def _last_wins(a: Any, b: Any) -> Any:
    """Reducer for fields written by exactly one branch of a parallel fan-out.

    LangGraph requires a reducer on any key more than one concurrent node
    could write. These keys are each written by a single branch, but the
    branches run in parallel, so a reducer must exist.
    """
    return b if b is not None else a


class ApplicationState(TypedDict, total=False):
    """One job, start to finish."""

    # inputs
    job: JobPosting
    run_id: str
    run_dir: str

    # analysis
    jd_text: str
    jd: ParsedJD
    fit: FitReport

    # composition (these three are produced by parallel branches)
    tailored: Annotated[TailoredResume | None, _last_wins]
    cover_letter: Annotated[CoverLetter | None, _last_wins]
    screening: Annotated[ScreeningAnswers | None, _last_wins]

    resume_pdf: str

    # critique cycle
    critique: CritiqueReport
    revision_count: int
    provenance_errors: Annotated[list[str], operator.add]

    # submission
    submission: SubmissionResult

    # bookkeeping
    errors: Annotated[list[str], operator.add]
    notes: Annotated[list[str], operator.add]
    status: Status


class DiscoveryState(TypedDict, total=False):
    """One search run across boards."""

    run_id: str
    run_dir: str
    queries: list[dict[str, Any]]
    listings: Annotated[list[JobPosting], operator.add]
    selected: list[JobPosting]
    results: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]


def new_application_state(job: JobPosting, run_id: str, run_dir: str) -> ApplicationState:
    return ApplicationState(
        job=job,
        run_id=run_id,
        run_dir=run_dir,
        revision_count=0,
        errors=[],
        notes=[],
        provenance_errors=[],
        status="scraped",
    )
