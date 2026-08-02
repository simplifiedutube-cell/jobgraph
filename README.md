# JobGraph

A LangGraph state-graph agent that finds jobs, scores fit against a structured
resume corpus, tailors a resume and cover letter per posting, and submits to
ATS forms — with provenance enforced in code so it cannot claim anything you
have not actually done.

Runs entirely on the **Claude Code CLI**. There is no `ANTHROPIC_API_KEY`
anywhere in this project.

```
                    ┌──────────── discovery ────────────┐
                    │  45 board APIs (Greenhouse /      │
                    │  Lever / Ashby) → score → dedupe  │
                    │  → location filter → diversify    │
                    └────────────────┬──────────────────┘
                                     │ one subgraph per job
                                     ▼
  fetch_jd → parse_jd → score_fit ─┬─ <75 ─→ log_rejected ─→ record
                                   │
                                   └─ ≥75 ─→ tailor_resume
                                                 │
                        ┌────────────────────────┼────────────────────┐
                        ▼                        ▼                    ▼
                 render_resume_pdf      write_cover_letter    answer_screening
                        └────────────────────────┼────────────────────┘
                                                 ▼
                                             critique ──revise──→ bump_revision
                                                 │                      │
                                                 │◄─────────────────────┘
                                                 │
                                    ┌────────────┴───────────┐
                                    ▼                        ▼
                             route_ats → submit          abandon
                              (greenhouse/lever/           │
                               ashby/manual)               │
                                    └──────→ record ◄──────┘
```

## Quick start

```bash
uv sync && uv run playwright install chromium

uv run jobgraph check      # validate the resume corpus
uv run jobgraph doctor     # verify the engine + show cache behaviour
uv run jobgraph render     # render the untailored resume to PDF

uv run jobgraph discover   # what would tonight work on? (free, no LLM calls)
uv run jobgraph apply --limit 5 --no-submit      # generate packages, no forms
uv run jobgraph apply --limit 5                  # dry-run submission
uv run jobgraph status                           # the ledger
```

Everything defaults to `DRY_RUN=true`: forms are filled and screenshotted,
nothing is clicked. See `.env.example`.

---

## Three things worth reading the code for

### 1. Provenance is enforced by code, not by a model

`data/profile/resume.yaml` is a bullet bank where every achievement carries a
stable id. The tailoring node may reword, reorder and re-emphasise — it may not
invent. Every emitted bullet must cite a `source_id` that exists in the corpus,
and `profile.verify_provenance()` checks that deterministically before anything
is submitted.

Asking an LLM whether its own output is faithful is precisely the failure mode
this guards against, so that half never asks a model anything.

**Known limitation, stated honestly:** the check verifies the cited id exists,
not that the rewrite stayed faithful to that specific bullet. A tailored bullet
that merges two real achievements passes. Catching *drift* is the critique
layer's job, and it is softer than the id check.

### 2. The prompt cache finding

Every `claude -p` call carries Claude Code's own system prompt as a prefix.
Measured on this machine:

| | default flags | stable flag set |
|---|---|---|
| prefix | 8.1k write + 23k read | **1.2k write + 19.2k read** |
| cost/call | $0.057 | **$0.014** |
| latency | 3.9s | **~1.8s** |

A stable flag set serves that prefix as a cache *read* at 0.1× rate. Changing
`--model` or `--effort` mints a fresh ~20k cache *write* at 2× — a ~10× swing
on overhead alone.

So `config.Profile` exists: nodes pick from three fixed `(model, effort)` pairs
rather than tuning effort per node. **The number of distinct profiles is the
number of warm prefixes you maintain.** Prompts also lead with the resume
corpus, so those ~5k tokens sit inside the cached prefix too.

`jobgraph doctor` shows this live — if every call writes instead of reads, the
flag set has drifted.

### 3. The agent refuses to guess

`match_answer` maps an ATS form label onto a pre-generated screening answer,
and returns `None` when it cannot. A required question with no match sends the
whole job to `manual.queue()`, which writes an `APPLY_BY_HAND.md` with the PDF,
letter and answers ready to paste.

This was found by running it: against a live Greenhouse form it filled five
fields correctly, hit a required *"Why Anthropic?"*, and stopped. That drove
the company-aware matcher — `"Why <Company>?"` is one of the most common
required questions and matched none of the generic patterns.

An agent that queues work it does not understand is far more useful than one
that submits a half-filled form under your name.

---

## What running it actually taught us

Findings that changed the design, all from real output rather than reasoning:

- **The critic was never satisfied.** It demanded revision three times at 100%
  keyword coverage. Splitting `blocking_issues` from `issues` at the schema
  level — and having the gate read `blocking_issues` rather than the verdict
  string — made it converge in one pass. Models say "revise" while listing
  nothing blocking far more often than the reverse.
- **A revision made the package worse**, dropping coverage 100% → 82% by
  deleting bullets to address "filler" feedback. Revision prompts now carry an
  explicit preserve-floor.
- **The fit scorer counted boilerplate as gaps** — "passion for collaboration",
  "willing to travel" (the resume says *open to travel*), and "degree **or
  equivalent practical experience**". It now classifies each requirement before
  scoring: hard technical / boilerplate / logistics / credential-with-
  equivalence.
- **A boolean title filter left 1,218 candidates** out of 5,700 — far more than
  a run can process. Replaced with a weighted score, plus a location filter
  that understands region-locked remote (`"Remote - US"` needs US work
  authorisation and is not remote-from-anywhere), plus round-robin across
  employers so the shortlist isn't five roles at one company.

### The finding that isn't a bug

With the scorer correctly calibrated, most discovered roles still score below
the 75 gate — and inspection says that's **right**. The Anthropic role is
pre-sales wanting 10+ years customer-facing; GitLab's is a business-systems
role wanting Marketo/Workato/Glean; Twilio's wants PyTorch plus Kafka/Spark.
None are applied-LLM product engineering.

755 further candidates drop on location: most AI-product companies do not hire
remote-into-India.

Lowering `MIN_FIT_SCORE` would manufacture applications that get rejected on
qualifications, at the cost of your name at those companies. The honest levers
are widening `data/profile/companies.yaml` toward employers who hire where you
are, or deciding a stretch application is worth it — both of which are a
human's call, not the graph's.

---

## Safety

| Control | Default | Effect |
|---|---|---|
| `DRY_RUN` | `true` | Fill and screenshot, click nothing |
| `MIN_FIT_SCORE` | 75 | Below this, stop before generating |
| `MAX_SUBMISSIONS_PER_RUN` | 5 | Per-run cap |
| `DAILY_SUBMISSION_CAP` | 15 | Per-day cap, enforced from the ledger |
| `REVIEW_MODE` | `auto` | `first_n` / `always` interrupt before the click |
| `STOP` file | — | Kill switch, checked at every node entry |

Deduping is by normalised URL against the ledger, so the same posting is never
applied to twice. Every submission records the JD snapshot, the exact PDF sent,
every field value entered, and before/after screenshots.

`batch.py` is the unattended entry point: per-job failures are recorded and
stepped over, while fatal conditions (quota exhausted, kill switch) abort the
whole run rather than producing N identical failures. Each run writes a
`SUMMARY.md`.

---

## Layout

```
src/jobgraph/
  llm.py        ClaudeCliEngine — structured output over `claude -p`, no API key
  profile.py    corpus loading + verify_provenance (the anti-fabrication core)
  schemas.py    every Pydantic model; model_json_schema() goes into the prompt
  config.py     settings, Profile, the kill switch
  browser.py    Playwright — scraping, form-fill and PDF, one browser
  storage.py    SQLite ledger, dedupe, caps
  batch.py      unattended runner
  nodes/        analyze · compose · critique · discover · record · submit/
evals/          answer matching, golden JDs
```

## Why no API key

All inference shells out to `claude -p --output-format json`, reading the
prompt from stdin (prompts embed the full corpus; argv caps near 1MB). There is
no `messages.parse()` over the CLI, so schema conformance is enforced in
`llm.py`: inject the JSON Schema, validate the reply, and re-prompt with the
validation error on failure. Retry rate is tracked in `TELEMETRY` and surfaces
as an eval metric — 0–14% in practice.

`LLMEngine` is a ~40-line interface. An `AnthropicEngine` using the real SDK is
a drop-in swap if this ever needs to run without Claude Code installed.

## Still to build

- LinkedIn / Naukri discovery and Easy Apply (browser-driven, brittle; the
  board APIs cover Greenhouse, Lever and Ashby without any of that risk)
- Golden-set eval for fit-score agreement against hand-labelled JDs
- Ashby-specific submit adapter (currently reuses the Greenhouse filler)
