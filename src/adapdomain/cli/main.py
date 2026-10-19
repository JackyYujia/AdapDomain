"""Top-level Typer app for AdapDomain."""
from __future__ import annotations

import typer

app = typer.Typer(
    name="adapdomain",
    help="Resource-shape-aware multi-tenant inference scheduling for Ascend 910B3.",
    no_args_is_help=True,
)


@app.command("version")
def version() -> None:
    """Print the installed AdapDomain version."""
    from adapdomain import __version__

    typer.echo(__version__)


from adapdomain.cli.commands.verify_remote import app as _verify_remote_app
app.add_typer(_verify_remote_app, name="verify-remote", help="Verify CANN/ACL on the remote NPU host.")


@app.command("run-model")
def run_model_placeholder() -> None:
    """Placeholder; real implementation in M15/M17."""
    typer.echo("run-model: not implemented yet.")


@app.command("profile")
def profile_placeholder() -> None:
    """Placeholder; real implementation in M21."""
    typer.echo("profile: not implemented yet.")


@app.command("schedule")
def schedule_placeholder() -> None:
    """Placeholder; real implementation in M28."""
    typer.echo("schedule: not implemented yet.")


from adapdomain.cli.commands.replay import app as _replay_app
app.add_typer(_replay_app, name="replay")


from adapdomain.cli.commands.ablate import app as _ablate_app
app.add_typer(_ablate_app, name="ablate")


from adapdomain.cli.commands.plot import app as _plot_app
app.add_typer(_plot_app, name="plot")


from adapdomain.cli.commands.llm_pd import app as _llm_pd_app
app.add_typer(_llm_pd_app, name="llm-pd")

from adapdomain.cli.commands.llm_measure import llm_measure
app.command("llm-measure")(llm_measure)


if __name__ == "__main__":
    app()
