"""The per-job subgraph.

    fetch_jd -> parse_jd -> score_fit -> [gate]
                                          |-- reject --> log_rejected --> record
                                          `-- proceed -> tailor_resume
                                                            |
                                    +-----------------------+------------------+
                                    v                       v                  v
                            render_resume_pdf      write_cover_letter   answer_screening
                                    +-----------------------+------------------+
                                                            v
                                                        critique -- revise --> bump_revision
                                                            |                       |
                                                            |<----------------------+
                                                            |
                                                            |-- abandon --> abandon --> record
                                                            `-- proceed --> submit... -> record

The three composition nodes run concurrently. `render_resume_pdf` depends on
tailor_resume rather than on the other two, so it starts immediately instead of
waiting for the letter.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from jobgraph.nodes import analyze, compose, critique as crit, record as rec
from jobgraph.state import ApplicationState

# Phase 4 replaces this with route_ats + the vendor adapters.
SUBMISSION_ENTRY = "record"


def build_application_graph(*, with_submission: bool = True):
    g = StateGraph(ApplicationState)

    g.add_node("fetch_jd", analyze.fetch_jd)
    g.add_node("parse_jd", analyze.parse_jd)
    g.add_node("score_fit", analyze.score_fit)
    g.add_node("log_rejected", analyze.log_rejected)

    g.add_node("tailor_resume", compose.tailor_resume)
    g.add_node("write_cover_letter", compose.write_cover_letter)
    g.add_node("answer_screening", compose.answer_screening)
    g.add_node("render_resume_pdf", compose.render_resume_pdf)

    g.add_node("critique", crit.critique)
    g.add_node("bump_revision", crit.bump_revision)
    g.add_node("abandon", crit.abandon)

    g.add_node("record", rec.record)

    # --- analysis ---------------------------------------------------------
    g.add_edge(START, "fetch_jd")
    g.add_edge("fetch_jd", "parse_jd")
    g.add_edge("parse_jd", "score_fit")
    g.add_conditional_edges(
        "score_fit",
        analyze.fit_gate,
        {"proceed": "tailor_resume", "rejected": "log_rejected"},
    )
    g.add_edge("log_rejected", "record")

    # --- composition, in parallel -----------------------------------------
    g.add_edge("tailor_resume", "render_resume_pdf")
    g.add_edge("tailor_resume", "write_cover_letter")
    g.add_edge("tailor_resume", "answer_screening")

    # critique waits for all three branches
    g.add_edge("render_resume_pdf", "critique")
    g.add_edge("write_cover_letter", "critique")
    g.add_edge("answer_screening", "critique")

    # --- the cycle --------------------------------------------------------
    submission_entry = "submit" if with_submission else "record"
    g.add_conditional_edges(
        "critique",
        crit.critique_gate,
        {
            "revise": "bump_revision",
            "abandon": "abandon",
            "proceed": submission_entry,
        },
    )
    g.add_edge("bump_revision", "tailor_resume")
    g.add_edge("abandon", "record")

    # --- submission -------------------------------------------------------
    if with_submission:
        from jobgraph.nodes import submit as sub

        g.add_node("submit", sub.route_and_submit)
        g.add_node("verify_submission", sub.verify_submission)
        g.add_edge("submit", "verify_submission")
        g.add_edge("verify_submission", "record")

    g.add_edge("record", END)
    return g


def compile_application_graph(checkpointer=None, *, with_submission: bool = True):
    return build_application_graph(with_submission=with_submission).compile(
        checkpointer=checkpointer
    )
