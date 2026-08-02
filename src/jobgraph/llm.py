"""The LLM engine: structured output over the `claude` CLI, no API key.

Design notes, all of them measured rather than assumed (2026-08-02, this machine):

* Every `claude -p` invocation carries Claude Code's own system prompt as a
  prefix. Default flags: ~31k tokens. With the flag set below (no MCP servers,
  no tool schemas, no per-machine sections): ~19.2k tokens.

* A *stable* flag set serves that prefix as a cache read at 0.1x rate:
  995 write / 19,236 read, ~$0.0125, ~1.8s per call. Changing --model or
  --effort mints a fresh ~20k cache write at 2x rate: ~$0.12, ~3.9s. That is a
  ~10x cost swing and a ~2x latency swing on the overhead alone.

  Hence `Profile`: nodes pick a (model, effort) pair from a fixed small set,
  and FLAGS below is assembled identically on every call. If you find yourself
  adding a flag that varies per call, you are paying for it.

* The prompt goes over **stdin**, not argv. Prompts here embed the full resume
  corpus plus a JD -- tens of KB, and argv is capped near 1MB.

* There is no `messages.parse()` over the CLI, so schema conformance is
  enforced here: inject the JSON Schema, validate the reply, and re-prompt with
  the validation error on failure. Every retry is recorded in `TELEMETRY` and
  surfaces as an eval metric.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from jobgraph.config import PROFILE_SPECS, Profile, settings

# --------------------------------------------------------------------------
# The flag set. Byte-identical on every call except --model / --effort, which
# come from the profile. Do not append to this casually.
# --------------------------------------------------------------------------

BASE_FLAGS: tuple[str, ...] = (
    "--output-format",
    "json",
    "--disable-slash-commands",
    "--no-session-persistence",
    "--strict-mcp-config",
    "--disallowed-tools",
    "Bash Edit Write Read Glob Grep WebFetch WebSearch Task NotebookEdit TodoWrite",
    "--exclude-dynamic-system-prompt-sections",
)

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


class LLMError(RuntimeError):
    """Non-retryable failure of the engine itself (CLI missing, timeout, quota)."""


class SchemaExhausted(LLMError):
    """The model never produced output matching the schema within max_attempts."""


class QuotaExhausted(LLMError):
    """Auth or usage-limit failure. In an unattended run this must stop the
    whole run rather than burn through every remaining job producing errors."""


@dataclass(slots=True)
class CallRecord:
    node: str
    schema: str
    profile: str
    attempts: int
    ok: bool
    duration_s: float
    cache_read: int = 0
    cache_write: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    failures: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Telemetry:
    """Process-wide call log. `jobgraph doctor` and the evals read this."""

    calls: list[CallRecord] = field(default_factory=list)

    def add(self, rec: CallRecord) -> None:
        self.calls.append(rec)

    @property
    def retry_rate(self) -> float:
        """Fraction of calls that needed more than one attempt. The headline
        reliability number for the CLI backend."""
        if not self.calls:
            return 0.0
        return sum(1 for c in self.calls if c.attempts > 1) / len(self.calls)

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def summary(self) -> dict[str, Any]:
        n = len(self.calls)
        return {
            "calls": n,
            "failed": sum(1 for c in self.calls if not c.ok),
            "retry_rate": round(self.retry_rate, 4),
            "avg_duration_s": round(sum(c.duration_s for c in self.calls) / n, 2) if n else 0.0,
            "cache_read_tokens": sum(c.cache_read for c in self.calls),
            "cache_write_tokens": sum(c.cache_write for c in self.calls),
            "output_tokens": sum(c.output_tokens for c in self.calls),
            "cost_usd": round(self.total_cost_usd, 4),
        }


TELEMETRY = Telemetry()


def _strip_fences(text: str) -> str:
    """The model usually obeys 'no fences'. Usually is not always."""
    return _FENCE.sub("", text).strip()


def _extract_json_object(text: str) -> str:
    """Pull the outermost JSON object out of a reply that may carry stray prose.

    Brace-counting rather than a regex, because nested objects and braces inside
    strings both break the naive version.
    """
    text = _strip_fences(text)
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text[start:], start):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _schema_instructions(model_cls: type[BaseModel]) -> str:
    schema = json.dumps(model_cls.model_json_schema(), indent=2)
    return (
        "\n\n---\n"
        "Respond with a single JSON object and nothing else. No prose before or "
        "after it, no markdown code fences, no explanation.\n\n"
        "It must validate against this JSON Schema:\n\n"
        f"{schema}\n"
    )


class ClaudeCliEngine:
    """Runs `claude -p` as a subprocess and returns validated Pydantic models."""

    def __init__(self, concurrency: int | None = None) -> None:
        self._sem = asyncio.Semaphore(concurrency or settings.llm_concurrency)

    async def _spawn(self, prompt: str, profile: Profile) -> tuple[int, str, str]:
        model, effort = PROFILE_SPECS[profile]
        argv = [
            "claude",
            "-p",
            *BASE_FLAGS,
            "--model",
            str(model),
            "--effort",
            str(effort),
            "--fallback-model",
            settings.llm_fallback_model,
        ]

        async with self._sem:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError as exc:
                raise LLMError(
                    "`claude` not found on PATH. This project runs entirely on the "
                    "Claude Code CLI; install it and run `claude` once to log in."
                ) from exc

            try:
                out, err = await asyncio.wait_for(
                    proc.communicate(prompt.encode()), timeout=settings.llm_timeout_s
                )
            except TimeoutError as exc:
                proc.kill()
                await proc.wait()
                raise LLMError(f"claude timed out after {settings.llm_timeout_s}s") from exc

        return (
            proc.returncode or 0,
            out.decode(errors="replace"),
            err.decode(errors="replace").strip(),
        )

    async def _run_cli(self, prompt: str, profile: Profile) -> dict[str, Any]:
        """One CLI turn, with backoff for transient process-level failures.

        Observed under 4-way concurrency: the CLI occasionally exits rc=1 with
        empty stdout *and* empty stderr. It is transient and a retry clears it,
        so it is absorbed here rather than surfacing as a node failure and
        failing an otherwise good application. Auth and quota failures are
        never retried — those abort the whole run.
        """
        last = ""
        for attempt in range(1, settings.llm_cli_retries + 1):
            rc, stdout, stderr = await self._spawn(prompt, profile)

            if rc == 0 and stdout.strip():
                try:
                    envelope = json.loads(stdout)
                except json.JSONDecodeError as exc:
                    last = f"non-JSON stdout: {exc}: {stdout[:300]}"
                else:
                    if not envelope.get("is_error") and envelope.get("subtype") == "success":
                        return envelope
                    detail = str(
                        envelope.get("result")
                        or envelope.get("api_error_status")
                        or stderr
                        or "unknown"
                    )
                    self._raise_for_cli_failure(rc, detail)  # fatal classes raise
                    last = detail
            else:
                self._raise_for_cli_failure(rc, stderr or stdout)  # fatal classes raise
                last = (
                    f"rc={rc} stdout={stdout[:200]!r} stderr={stderr[:200]!r}"
                    if (stdout or stderr)
                    else f"rc={rc} with no output (transient CLI failure)"
                )

            if attempt < settings.llm_cli_retries:
                await asyncio.sleep(2.0 * attempt)

        raise LLMError(f"claude failed after {settings.llm_cli_retries} attempts: {last}")

    @staticmethod
    def _raise_for_cli_failure(rc: int, detail: str) -> None:
        """Raise ONLY for failures where retrying is pointless.

        Separates 'this run is doomed' from 'this call failed'. Matters most
        overnight: a quota or auth failure must abort the whole run rather than
        marking twenty jobs failed one at a time, each after three retries.
        Anything else returns so the caller can back off and try again.
        """
        low = detail.lower()
        fatal = (
            "usage limit",
            "rate limit",
            "quota",
            "not logged in",
            "please log in",
            "authentication",
            "unauthorized",
            "invalid api key",
            "credit balance",
        )
        if any(t in low for t in fatal):
            raise QuotaExhausted(f"claude auth/usage failure (rc={rc}): {detail[:400]}")

    async def call(
        self,
        *,
        node: str,
        prompt: str,
        schema: type[BaseModel],
        profile: Profile = Profile.STANDARD,
    ) -> BaseModel:
        """Prompt -> validated instance of `schema`.

        Retries re-prompt with the validation error attached; blind retries are
        markedly worse at recovering than showing the model what it got wrong.
        """
        base = prompt.rstrip() + _schema_instructions(schema)
        attempt_prompt = base
        started = time.perf_counter()
        failures: list[str] = []
        rec_stats: dict[str, Any] = {}

        for attempt in range(1, settings.llm_max_attempts + 1):
            envelope = await self._run_cli(attempt_prompt, profile)

            usage = envelope.get("usage", {}) or {}
            rec_stats = {
                "cache_read": usage.get("cache_read_input_tokens", 0),
                "cache_write": usage.get("cache_creation_input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cost_usd": envelope.get("total_cost_usd", 0.0) or 0.0,
            }

            raw = envelope.get("result") or ""
            candidate = _extract_json_object(raw)

            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError as exc:
                failures.append(f"attempt {attempt}: not JSON — {exc}")
                attempt_prompt = (
                    f"{base}\n\n---\nYour previous reply was not valid JSON.\n"
                    f"Parser error: {exc}\n"
                    f"Your reply began: {raw[:300]!r}\n"
                    "Return only the JSON object."
                )
                continue

            try:
                validated = schema.model_validate(obj)
            except ValidationError as exc:
                failures.append(f"attempt {attempt}: schema — {exc.error_count()} error(s)")
                attempt_prompt = (
                    f"{base}\n\n---\nYour previous reply was valid JSON but did not "
                    f"match the schema.\n\nValidation errors:\n{exc}\n\n"
                    "Return a corrected JSON object."
                )
                continue

            TELEMETRY.add(
                CallRecord(
                    node=node,
                    schema=schema.__name__,
                    profile=str(profile),
                    attempts=attempt,
                    ok=True,
                    duration_s=time.perf_counter() - started,
                    failures=failures,
                    **rec_stats,
                )
            )
            return validated

        TELEMETRY.add(
            CallRecord(
                node=node,
                schema=schema.__name__,
                profile=str(profile),
                attempts=settings.llm_max_attempts,
                ok=False,
                duration_s=time.perf_counter() - started,
                failures=failures,
                **rec_stats,
            )
        )
        raise SchemaExhausted(
            f"{node}: no schema-valid reply for {schema.__name__} in "
            f"{settings.llm_max_attempts} attempts. " + " | ".join(failures)
        )


ENGINE = ClaudeCliEngine()
