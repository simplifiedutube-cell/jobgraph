"""Command line entry point."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from jobgraph.config import PROFILE_SPECS, Profile, settings

app = typer.Typer(
    name="jobgraph",
    help="Find, score, tailor for and apply to jobs.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def doctor(
    calls: int = typer.Option(3, help="Calls per profile to exercise the cache."),
    profiles: list[Profile] = typer.Option(
        [Profile.FAST, Profile.STANDARD],
        "--profile",
        help="Profiles to exercise. Each one maintains its own warm prefix.",
    ),
) -> None:
    """Verify the CLI engine and show what the prompt cache is actually doing.

    What you want to see: the first call for a profile shows a large
    cache_write, and every call after it shows cache_write near zero and
    cache_read around 19k. If every call writes, the flag set is varying
    between calls and you are paying ~10x on overhead.
    """
    from pydantic import BaseModel, Field

    from jobgraph.llm import ENGINE, TELEMETRY

    class Probe(BaseModel):
        n: int = Field(description="Echo the integer given in the prompt.")
        word: str = Field(description="Echo the word given in the prompt.")

    async def _run() -> None:
        for profile in profiles:
            model, effort = PROFILE_SPECS[profile]
            console.print(f"\n[bold]{profile}[/bold]  ({model} / effort={effort})")
            for i in range(1, calls + 1):
                try:
                    got = await ENGINE.call(
                        node="doctor",
                        prompt=f"n is {i}. word is 'probe{i}'.",
                        schema=Probe,
                        profile=profile,
                    )
                except Exception as exc:  # noqa: BLE001 - diagnostic surface
                    console.print(f"  [red]call {i} failed:[/red] {exc}")
                    continue
                rec = TELEMETRY.calls[-1]
                warm = "read" if rec.cache_read > rec.cache_write else "[yellow]WRITE[/yellow]"
                ok = "ok" if got.n == i else f"[yellow]echo mismatch: {got.n}[/yellow]"
                console.print(
                    f"  call {i}: {ok}  {rec.duration_s:5.2f}s  "
                    f"write={rec.cache_write:>6,}  read={rec.cache_read:>6,}  "
                    f"${rec.cost_usd:.4f}  [{warm}]  attempts={rec.attempts}"
                )

    asyncio.run(_run())

    s = TELEMETRY.summary()
    table = Table(title="engine summary", show_header=False)
    for k, v in s.items():
        table.add_row(k, str(v))
    console.print(table)

    if s["calls"] and s["cache_write_tokens"] > s["cache_read_tokens"]:
        console.print(
            "\n[yellow]More cache writes than reads.[/yellow] Expected on a first "
            "run (each profile warms once). If it persists, the flag set is not "
            "stable across calls — see BASE_FLAGS in llm.py."
        )
    if s["failed"]:
        raise typer.Exit(1)


@app.command()
def render(
    out: Path = typer.Option(Path("/tmp/jobgraph-baseline.pdf"), help="Output PDF path."),
    open_after: bool = typer.Option(False, "--open", help="Open the PDF when done."),
) -> None:
    """Render the untailored resume straight from the corpus.

    Checks the template and the corpus in one shot, with no LLM involved. If
    this doesn't look like your resume, fix it before running the graph.
    """
    from jobgraph.profile import load_profile
    from jobgraph.render import baseline_resume, render_pdf

    prof = load_profile()
    tailored = baseline_resume(prof)
    path = asyncio.run(render_pdf(tailored, out, profile=prof))

    n_bullets = sum(len(s.bullets) for s in tailored.sections)
    console.print(
        f"[green]rendered[/green] {path}  "
        f"({len(tailored.sections)} sections, {n_bullets} bullets, "
        f"{path.stat().st_size // 1024} KB)"
    )
    console.print(f"[dim]html: {path.with_suffix('.html')}[/dim]")
    if open_after:
        subprocess.run(["open", str(path)], check=False)


@app.command()
def check() -> None:
    """Validate the profile corpus without calling the model."""
    from jobgraph.profile import load_preferences, load_profile, load_qa_bank

    prof = load_profile()
    prefs = load_preferences()
    qa = load_qa_bank()

    table = Table(title="corpus", show_header=False)
    table.add_row("name", prof.name)
    table.add_row("roles", str(len(prof.roles)))
    table.add_row("projects", str(len(prof.projects)))
    table.add_row("bullets", str(len(prof.all_bullets())))
    table.add_row("unique ids", str(len(prof.bullet_ids())))
    table.add_row("skill groups", str(len(prof.skills)))
    table.add_row("target titles", str(len(prefs.get("titles", []))))
    table.add_row("qa bank entries", str(len(qa.get("entries", []))))
    console.print(table)
    console.print("[green]corpus ok[/green] — all bullet ids unique")


@app.command()
def run(
    url: str = typer.Option(..., "--url", help="Job posting URL."),
    company: str = typer.Option("", help="Override company (else inferred from the page)."),
    title: str = typer.Option("", help="Override title (else inferred from the page)."),
    submit: bool = typer.Option(
        False, "--submit/--no-submit", help="Run the submission stage."
    ),
) -> None:
    """Run one job posting through the application graph."""
    from jobgraph.graph_application import compile_application_graph
    from jobgraph.llm import TELEMETRY
    from jobgraph.run import checkpointer, new_run_id, run_one_application
    from jobgraph.schemas import JobPosting

    run_id = new_run_id()
    job = JobPosting(
        url=url,
        title=title or "(from posting)",
        company=company or "(from posting)",
    )

    console.print(f"[bold]run {run_id}[/bold]  dry_run={settings.dry_run}  submit={submit}")

    async def _go() -> dict:
        async with checkpointer() as cp:
            graph = compile_application_graph(cp, with_submission=submit)
            return await run_one_application(job, run_id=run_id, graph=graph)

    final = asyncio.run(_go())

    for note in final.get("notes", []):
        console.print(f"  [dim]·[/dim] {note}")
    for err in final.get("errors", []):
        console.print(f"  [red]![/red] {err}")

    fit = final.get("fit")
    console.print(
        f"\nstatus=[bold]{final.get('status')}[/bold]"
        + (f"  fit={fit.score} ({fit.recommendation})" if fit else "")
        + f"  revisions={final.get('revision_count', 0)}"
    )
    if final.get("run_dir"):
        console.print(f"[dim]{Path(final['run_dir']) / 'package.md'}[/dim]")

    s = TELEMETRY.summary()
    console.print(
        f"[dim]llm: {s['calls']} calls, retry_rate={s['retry_rate']}, "
        f"${s['cost_usd']}, {s['avg_duration_s']}s avg[/dim]"
    )

    if final.get("status") == "failed":
        raise typer.Exit(1)


@app.command()
def status(limit: int = typer.Option(20, help="Rows to show.")) -> None:
    """Show recent applications and today's remaining budget."""
    from jobgraph import storage

    storage.init_db()
    counts = storage.status_counts()
    rows = storage.recent_applications(limit)

    if counts:
        console.print(
            "  ".join(f"[bold]{k}[/bold]={v}" for k, v in sorted(counts.items()))
            + f"   today={storage.submissions_today()}"
            f"/{settings.daily_submission_cap}"
        )

    if not rows:
        console.print("[dim]no applications recorded yet[/dim]")
        return

    table = Table(show_lines=False)
    table.add_column("when", style="dim", no_wrap=True)
    table.add_column("company")
    table.add_column("title", max_width=38)
    table.add_column("fit", justify="right")
    table.add_column("status")
    table.add_column("ats", style="dim")
    for r in rows:
        table.add_row(
            (r["created_at"] or "")[:16].replace("T", " "),
            r["company"],
            r["title"],
            str(r["fit_score"] or "-"),
            r["status"],
            r["ats_vendor"] or "-",
        )
    console.print(table)


@app.command()
def config() -> None:
    """Print effective settings."""
    console.print_json(json.dumps(settings.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    app()
