"""Summarize timestamped vLLM-Ascend serving request logs."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from adapdomain.experiments.llm_requests import (
    load_request_csv,
    summarize_requests,
    write_request_summaries,
)


def llm_measure(
    requests_csv: Annotated[Path, typer.Argument(exists=True, readable=True)],
    duration_s: Annotated[float, typer.Option(min=0.001, help="Fixed offered-arrival window in seconds.")],
    out: Annotated[Path, typer.Option()] = Path("data/results/llm_request_summary.csv"),
    ttft_slo_ms: Annotated[float, typer.Option(min=0.001)] = 500.0,
    itl_slo_ms: Annotated[float, typer.Option(min=0.001)] = 40.0,
) -> None:
    """Compute request-level goodput and violations from live run timestamps."""
    summaries = summarize_requests(
        load_request_csv(requests_csv), duration_s=duration_s,
        ttft_slo_ms=ttft_slo_ms, itl_slo_ms=itl_slo_ms,
    )
    write_request_summaries(summaries, out)
    typer.echo(f"Wrote {len(summaries)} policy/trace/seed summaries to {out}")
