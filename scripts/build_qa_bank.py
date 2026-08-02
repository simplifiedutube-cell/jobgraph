"""Seed data/profile/qa_bank.yaml from the hand-written interview question bank.

The source HTML holds coaching notes ("what they're testing", "strong answer
contains"), not verbatim answers. They're stored here as *guidance*: the
composer uses them to shape a screening answer, grounded in resume.yaml
bullets for anything factual.

Usage:
    uv run python scripts/build_qa_bank.py \
        "../Interview prep/resume-question-bank.html"
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml
from selectolax.parser import HTMLParser

DEFAULT_SRC = Path.home() / "Documents" / "Interview prep" / "resume-question-bank.html"
OUT = Path(__file__).resolve().parents[1] / "data" / "profile" / "qa_bank.yaml"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def build(src: Path) -> dict:
    tree = HTMLParser(src.read_text(encoding="utf-8"))
    entries: list[dict] = []

    # NB: css("h2, details") returns all h2s *then* all details, not document
    # order, which silently files every question under the last heading.
    # Scope the walk to each <section> instead.
    containers = tree.css("section") or [tree.body]

    for container in containers:
        h2 = container.css_first("h2")
        section = _clean(h2.text()) if h2 else "General"

        for node in container.css("details"):
            summary = node.css_first("summary")
            if summary is None:
                continue

            idx_node = summary.css_first(".q-ix")
            idx = _clean(idx_node.text()) if idx_node else ""

            tags = [_clean(t.text()).lower() for t in summary.css(".tag")]

            # The question is the summary text minus the index and tag chips.
            question = _clean(summary.text())
            for strip in [idx, *(t.upper() for t in tags)]:
                if strip:
                    question = question.replace(strip, " ", 1)
            question = _clean(question.replace("›", " "))
            if not question:
                continue

            body = node.css_first(".a")
            guidance: list[str] = []
            if body is not None:
                for lbl in body.css(".lbl"):
                    label = _clean(lbl.text())
                    # Text between this label and the next belongs to it.
                    chunk = []
                    sib = lbl.next
                    while sib is not None and not (
                        sib.attributes and "lbl" in (sib.attributes.get("class") or "")
                    ):
                        text = _clean(sib.text())
                        if text:
                            chunk.append(text)
                        sib = sib.next
                    if chunk:
                        guidance.append(f"{label}: {' '.join(chunk)}")
                if not guidance:
                    guidance = [_clean(body.text())]

            entries.append(
                {
                    "id": f"qa.{idx}" if idx else f"qa.{len(entries) + 1}",
                    "section": section,
                    "question": question,
                    "tags": tags,
                    "guidance": guidance,
                }
            )

    return {
        "source": str(src),
        "note": (
            "Coaching notes, not verbatim answers. answer_screening_questions "
            "uses these to shape tone and structure; every factual claim must "
            "still cite a resume.yaml bullet id."
        ),
        "entries": entries,
    }


def main() -> None:
    src = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_SRC
    if not src.exists():
        raise SystemExit(f"source not found: {src}")

    bank = build(src)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        yaml.safe_dump(bank, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )

    sections = {e["section"] for e in bank["entries"]}
    print(f"wrote {OUT}")
    print(f"  {len(bank['entries'])} entries across {len(sections)} sections")


if __name__ == "__main__":
    main()
