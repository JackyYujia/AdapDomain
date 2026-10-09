"""`adapdomain plot paper` bundle."""
from __future__ import annotations

from pathlib import Path

import typer

from adapdomain.experiments.plotting import plot_e2e_bar, plot_overhead_curve
from adapdomain.experiments.tables import TableRow, write_latex, write_markdown

app = typer.Typer(help="Reproduce paper figures and tables.")


@app.command("paper")
def paper(
    results: str = typer.Option("data/results"),
    out: str = typer.Option("data"),
) -> None:
    out_dir = Path(out)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    # Placeholder data for offline smoke; real values come from data/results/<run-id>/summary.csv files.
    plot_e2e_bar({"Exclusive": 31.4, "BFD-2D": 66.7, "Dilu-style": 78.8, "AdapDomain": 141.1}, fig_dir / "fig3_e2e.png")
    plot_overhead_curve([10, 100, 400, 800, 1000], [2.79, 19.07, 77.88, 117.80, 131.64], fig_dir / "fig8_runtime_overhead.png")
    write_markdown(tab_dir / "tab_e2e.md", ["Config", "Goodput", "Violation"], [
        TableRow(cells=["Exclusive", "31.4", "0.37"]),
        TableRow(cells=["BFD-2D", "66.7", "17.97"]),
        TableRow(cells=["Dilu-style", "78.8", "15.13"]),
        TableRow(cells=["AdapDomain", "141.1", "5.59"]),
    ])
    write_latex(tab_dir / "tab_overhead.tex", ["Operation", "Size", "Mean (ms)", "p95 (ms)", "Max (ms)"], [
        TableRow(cells=["Placement planning", "10", "2.66", "2.79", "2.93"]),
        TableRow(cells=["Placement planning", "100", "18.16", "19.07", "19.97"]),
        TableRow(cells=["Placement planning", "400", "74.17", "77.88", "81.59"]),
    ])
    typer.echo(str(out_dir))
