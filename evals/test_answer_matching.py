"""The answer matcher decides what gets typed into a real form.

Two properties matter, and the second matters more:
  1. Common ATS question phrasings resolve to the right pre-generated answer.
  2. An unrecognised question resolves to None, so the adapter queues the job
     for a human instead of improvising into a required field.
"""

from __future__ import annotations

import pytest

from jobgraph.nodes.submit.base import match_answer
from jobgraph.schemas import ScreeningAnswer, ScreeningAnswers

ANSWERS = ScreeningAnswers(
    answers=[
        ScreeningAnswer(
            question="Why do you want to work at this company?",
            answer="COMPANY",
            source_ids=[],
        ),
        ScreeningAnswer(
            question="Why are you interested in this role?", answer="ROLE", source_ids=[]
        ),
        ScreeningAnswer(
            question="What is your notice period / earliest start date?",
            answer="NOTICE",
            source_ids=[],
        ),
        ScreeningAnswer(
            question="What are your salary expectations?", answer="SALARY", source_ids=[]
        ),
        ScreeningAnswer(
            question="Are you legally authorised to work in this location?",
            answer="AUTH",
            source_ids=[],
        ),
        ScreeningAnswer(
            question="Are you willing to relocate or work on-site?",
            answer="RELOCATE",
            source_ids=[],
        ),
        ScreeningAnswer(
            question="What relevant experience do you bring to this role?",
            answer="EXPERIENCE",
            source_ids=[],
        ),
    ]
)


@pytest.mark.parametrize(
    ("label", "company", "expected"),
    [
        # Company-specific phrasings — the single most common required question
        # on Greenhouse, and the one that matches no generic pattern.
        ("Why Anthropic?*", "Anthropic", "COMPANY"),
        ("Why do you want to join Northwind Labs?", "Northwind Labs", "COMPANY"),
        ("Why are you excited about Stripe?", "Stripe, Inc.", "COMPANY"),
        # Generic phrasings
        ("Why are you interested in this position?", "Acme", "ROLE"),
        ("Expected CTC (INR)", "Acme", "SALARY"),
        ("What is your expected compensation?", "Acme", "SALARY"),
        ("Notice period *", "Acme", "NOTICE"),
        ("When can you start?", "Acme", "NOTICE"),
        ("Are you legally authorized to work in the US?", "Acme", "AUTH"),
        ("Do you require visa sponsorship?", "Acme", "AUTH"),
        ("Are you willing to relocate to Bangalore?", "Acme", "RELOCATE"),
        ("Tell us about your relevant experience", "Acme", "EXPERIENCE"),
    ],
)
def test_known_questions_resolve(label: str, company: str, expected: str) -> None:
    assert match_answer(label, ANSWERS, company=company) == expected


@pytest.mark.parametrize(
    "label",
    [
        "Describe your favourite colour",
        "What is your T-shirt size?",
        "Please upload a portfolio of generative art",
        "",
    ],
)
def test_unknown_questions_return_none(label: str) -> None:
    """The important one. None routes the job to a human."""
    assert match_answer(label, ANSWERS, company="Acme") is None


def test_no_screening_answers_returns_none() -> None:
    assert match_answer("Why Acme?", None, company="Acme") is None
    assert match_answer("Why Acme?", ScreeningAnswers(answers=[]), company="Acme") is None


def test_company_pattern_is_scoped_to_the_company_being_applied_to() -> None:
    """A question naming a *different* company must not match.

    If we are applying to Northwind Labs and the form asks "Why Anthropic?",
    something is wrong — a stale cached page, the wrong tab, a mis-scraped
    URL. Declining to answer sends it to a human, which is the right outcome.
    """
    assert match_answer("Why Anthropic?", ANSWERS, company="Northwind Labs") is None

    # And a question that merely mentions the company is not the same question
    # as "why do you want to work here".
    assert match_answer("Which Anthropic products have you used?", ANSWERS, company="Anthropic") is None
