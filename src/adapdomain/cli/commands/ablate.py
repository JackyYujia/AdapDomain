"""`adapdomain ablate`."""
from __future__ import annotations

import typer

from adapdomain.utils.config import ExperimentAblationConfig

app = typer.Typer(help="HarmonicFit and ActiveVC ablations.")


@app.callback(invoke_without_command=True)
def ablate(config: str = typer.Option("configs/experiments/ablation.yaml")) -> None:
    cfg = ExperimentAblationConfig.from_yaml(config)
    typer.echo(
        f"running ablation with {len(cfg.harmonicfit_variants)} variants, "
        f"{len(cfg.activevc_baselines)} baselines, budgets={cfg.budgets}",
    )
