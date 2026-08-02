"""Runtime configuration.

Everything tunable lives here so the guardrails are visible in one place and
the CLI flag set stays byte-identical across every LLM call (see llm.py).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PROFILE_DIR = DATA_DIR / "profile"
RUNS_DIR = DATA_DIR / "runs"
DB_PATH = DATA_DIR / "jobgraph.db"
STOP_FILE = REPO_ROOT / "STOP"


class Model(StrEnum):
    """Aliases accepted by `claude --model`."""

    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"


class Effort(StrEnum):
    """Levels accepted by `claude --effort`."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class ReviewMode(StrEnum):
    AUTO = "auto"
    """Submit without stopping."""
    FIRST_N = "first_n"
    """Interrupt before the first `review_first_n` submissions, then go auto."""
    ALWAYS = "always"
    """Interrupt before every submission."""


class Profile(StrEnum):
    """A (model, effort) pair. Nodes pick a profile, never a raw model+effort.

    Measured on 2026-08-02: each `claude -p` call carries ~19-20k tokens of
    Claude Code system prompt. A stable flag set serves that as a cache *read*
    at 0.1x rate (995 write / 19,236 read, ~$0.0125, ~1.8s). Changing --model
    or --effort mints a fresh ~20k cache *write* at 2x rate (~$0.12, ~3.9s).

    So the number of distinct profiles is the number of warm prefixes you
    maintain, and each one costs a write the first time it is used in a run.
    Three is the sweet spot. Adding a fourth is a real, measurable cost --
    don't add one casually, and don't be tempted to hand-tune effort per node.
    """

    FAST = "fast"
    """haiku + low. Mechanical extraction where the answer is in the input."""
    STANDARD = "standard"
    """sonnet + medium. The default for judgement and generation."""
    DEEP = "deep"
    """opus + high. Reserved for nodes where quality visibly pays off."""


PROFILE_SPECS: dict[Profile, tuple[Model, Effort]] = {
    Profile.FAST: (Model.HAIKU, Effort.LOW),
    Profile.STANDARD: (Model.SONNET, Effort.MEDIUM),
    Profile.DEEP: (Model.OPUS, Effort.HIGH),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- safety -----------------------------------------------------------
    dry_run: bool = True
    """Fill and screenshot ATS forms but never click submit."""

    min_fit_score: int = 75
    """Below this, the graph terminates at log_rejected without generating."""

    max_submissions_per_run: int = 5
    daily_submission_cap: int = 15

    review_mode: ReviewMode = ReviewMode.AUTO
    review_first_n: int = 3
    """Only meaningful when review_mode is FIRST_N."""

    max_revisions: int = 2
    """Bound on the critique -> tailor_resume cycle."""

    min_keyword_coverage: int = 70
    """Below this, the package is sent back regardless of what the critic
    thought. Computed in code from the JD's tech_keywords, so it can't be
    flattered away."""

    # ---- llm engine -------------------------------------------------------
    llm_concurrency: int = 4
    """Semaphore over `claude` subprocesses. Send() fan-out will otherwise
    fork one process per job per node."""

    llm_timeout_s: int = 300
    llm_max_attempts: int = 3
    """Total tries per call, including the first. Retries re-prompt with the
    validation error attached."""

    llm_cli_retries: int = 3
    """Process-level retries with backoff, separate from the schema retries
    above. The CLI intermittently exits rc=1 with no output under concurrency;
    that is transient and a retry clears it. Auth and quota failures are never
    retried — those abort the whole run."""

    llm_fallback_model: str = "sonnet,haiku"
    """Passed to --fallback-model so an overloaded model doesn't kill a run."""

    # Per-node profile routing. Keep the set of distinct profiles small --
    # see Profile's docstring for why. Defaults put the two extraction nodes
    # on FAST and everything with judgement in it on STANDARD, which is two
    # warm prefixes per run.
    profile_plan_searches: Profile = Profile.FAST
    profile_parse_jd: Profile = Profile.FAST
    profile_score_fit: Profile = Profile.STANDARD
    profile_tailor: Profile = Profile.STANDARD
    profile_cover_letter: Profile = Profile.STANDARD
    profile_screening: Profile = Profile.STANDARD
    profile_critique: Profile = Profile.STANDARD

    # ---- browser ----------------------------------------------------------
    browser_profile_dir: Path = DATA_DIR / "browser"
    headless: bool = False
    """Scraping logged-in boards headless trips bot detection more often."""

    chrome_cdp_url: str = ""
    """Attach to a Chrome you started with --remote-debugging-port=9222 and
    reuse the sessions you are already signed into. Empty means use the
    project's own profile in data/browser/. See browser.browser_context."""

    action_delay_ms: tuple[int, int] = (400, 1400)
    """Uniform-random pause between browser actions."""

    # ---- paths ------------------------------------------------------------
    profile_dir: Path = PROFILE_DIR
    runs_dir: Path = RUNS_DIR
    db_path: Path = DB_PATH

    @property
    def checkpoint_db_path(self) -> Path:
        """Separate file from the ledger.

        LangGraph's checkpointer holds a long-lived write transaction for the
        duration of a graph run, so sharing one file with the ledger produced
        `database is locked` the moment a node wrote a row — WAL and a busy
        timeout only narrowed the window. Two files, no contention.
        """
        return self.db_path.with_name(self.db_path.stem + "_checkpoints.db")

    @property
    def resume_path(self) -> Path:
        return self.profile_dir / "resume.yaml"

    @property
    def preferences_path(self) -> Path:
        return self.profile_dir / "preferences.yaml"

    @property
    def qa_bank_path(self) -> Path:
        return self.profile_dir / "qa_bank.yaml"


settings = Settings()


class StopRequested(RuntimeError):
    """Raised at node entry when the STOP kill-switch file exists."""


def check_stop() -> None:
    """Kill switch. Called at the top of every node.

    `touch STOP` in the repo root halts the graph at the next node boundary;
    the SQLite checkpointer means you resume from there, not from scratch.
    """
    if STOP_FILE.exists():
        raise StopRequested(
            f"{STOP_FILE} exists — halting. Delete it to resume this run."
        )
