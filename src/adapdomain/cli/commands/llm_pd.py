"""`adapdomain llm-pd`."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from adapdomain.experiments.llm_pd import (
    LlmPdInputs,
    bursty_trace,
    default_profiles,
    load_profiles_csv,
    load_trace_csv,
    run_llm_pd_experiment,
    write_results,
)

app = typer.Typer(help="Run the offline LLM prefill/decode deployment experiment.")


@app.callback(invoke_without_command=True)
def llm_pd(
    profiles_csv: Path | None = typer.Option(None, help="Measured phase profile CSV. If omitted, use built-in smoke profiles."),
    trace_csv: Path | None = typer.Option(None, help="Measured or generated phase trace CSV. If omitted, use a deterministic bursty trace."),
    out: Path = typer.Option(Path("data/results/llm_pd"), help="Output directory."),
    num_cards: int = typer.Option(4, min=1, help="Number of NPU cards."),
    precision: str | None = typer.Option(None, help="Select one precision from a mixed profile CSV."),
) -> None:
    profiles = load_profiles_csv(profiles_csv, precision=precision) if profiles_csv else default_profiles()
    trace = load_trace_csv(trace_csv) if trace_csv else bursty_trace()
    results = run_llm_pd_experiment(LlmPdInputs(profiles=profiles, trace=trace, num_cards=num_cards))
    write_results(results, out)

    table = Table(title="LLM PD deployment comparison")
    table.add_column("Policy")
    table.add_column("Goodput qps", justify="right")
    table.add_column("Violation %", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("Accepted", justify="right")
    for row in results:
        table.add_row(
            row.policy,
            f"{row.goodput_qps:.2f}",
            f"{row.violation_pct:.2f}",
            f"{row.p95_latency_ms:.2f}",
            f"{row.accepted}/{row.total}",
        )
    console = Console()
    console.print(table)
    console.print(f"[green]Wrote[/green] {out / 'llm_pd_summary.csv'}")
